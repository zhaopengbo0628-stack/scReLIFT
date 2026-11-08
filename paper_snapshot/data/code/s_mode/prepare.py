"""Prepare one explicitly identified benchmark; never read another dataset's ATAC."""
import importlib.util
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse

ROOT = Path('/home/u202522802045/share/zpb')
sys.path[:0] = [str(ROOT), str(ROOT / 'main/run'), str(ROOT / 'main/run/d_mode'), str(ROOT / 'main/core/regulatory/common/legacy_support')]
import prepare_updated_b2_sources as legacy
import unblock_updated_full_b2 as reader
from main.core.regulatory.s_mode.single_source import (
    build_bs, build_field, canonical_genes, finite, normalize_counts, select_hvg)
from v5d.regulatory_graph import read_tss


def splits(key):
    directory = legacy.DATA[key] / 'processed'
    if all((directory / f'{s}_index.npy').exists() for s in ('train', 'val', 'test')):
        result = {s: np.load(directory / f'{s}_index.npy') for s in ('train', 'val', 'test')}
    else:
        with np.load(legacy.PREPARED[key] / 'metadata.npz', allow_pickle=True) as m:
            result = {s: m[f'{s}_index'].copy() for s in ('train', 'val', 'test')}
    cells = reader.barcodes(key)
    joined = np.concatenate(list(result.values()))
    if not np.array_equal(np.sort(joined), np.arange(len(cells))):
        raise ValueError(f'{key}: overlapping or incomplete frozen split')
    if len(set(cells)) != len(cells):
        raise ValueError(f'{key}: duplicated barcodes')
    return result, cells


def paired_atac(key, rows, cells):
    if key == 'D2':
        directory = legacy.DATA[key] / 'processed'
        path = directory / 'atac_counts.npz'
        peaks = np.load(directory / 'peak_names.npy').astype(str)
        matrix = sparse.load_npz(path)[rows].tocsr()
        dependencies = [path, directory / 'peak_names.npy', directory / 'barcodes.txt']
    else:
        path = next((legacy.DATA[key] / 'raw').glob('*feature_bc_matrix.h5'))
        with h5py.File(path, 'r') as h:
            m = h['matrix']
            is_peak = m['features/feature_type'][:].astype(str) == 'Peaks'
            peaks = m['features/id'][:].astype(str)[is_peak]
            lookup = {cell: i for i, cell in enumerate(m['barcodes'][:].astype(str))}
            feature_map = np.full(len(is_peak), -1, np.int64)
            feature_map[is_peak] = np.arange(is_peak.sum())
            pointers = m['indptr'][:]
            data, indices, indptr = [], [], [0]
            for cell in np.asarray(cells)[rows]:
                i = lookup[cell]
                lo, hi = int(pointers[i]), int(pointers[i+1])
                ix = m['indices'][lo:hi]
                keep = is_peak[ix]
                positions = np.flatnonzero(keep) + lo
                data.append(m['data'][positions] if len(positions) else np.array([], np.float32))
                indices.append(feature_map[ix[keep]])
                indptr.append(indptr[-1] + keep.sum())
            matrix = sparse.csr_matrix((np.concatenate(data), np.concatenate(indices), indptr),
                                       shape=(len(rows), len(peaks)))
        dependencies = [path]
    finite(matrix, 'paired ATAC')
    if (matrix.data < 0).any() or (matrix.data != np.floor(matrix.data)).any():
        raise ValueError('ATAC must be nonnegative raw counts')
    if matrix.shape[1] != len(peaks) or len(set(peaks)) != len(peaks):
        raise ValueError('Peak names duplicated or dimensions differ')
    return matrix.astype(bool).astype(np.float32), peaks, dependencies


def prepare(key, cache_root, expected_identity):
    cache = legacy.cache_dir(cache_root, key)
    manifest_path = cache / 'manifest.json'
    if manifest_path.exists():
        manifest = legacy.checked(manifest_path)
        if manifest['identity'] != expected_identity:
            raise ValueError('Existing cache identity differs')
        return cache, manifest
    if cache.exists() and any(cache.iterdir()):
        raise RuntimeError(f'Incomplete cache requires inspection: {cache}')
    started = time.perf_counter()
    identity_path = legacy.DATA[key] / 'DATASET_IDENTITY.json'
    identity = json.loads(identity_path.read_text())
    if identity['canonical_identity'] != expected_identity:
        raise ValueError(f'{key} identity mismatch: {identity["canonical_identity"]}')
    index, cells = splits(key)
    counts, ids, raw_path, provenance = reader.raw_rna(key)
    if counts.shape[0] != len(cells) or counts.shape[0] != identity['cell_count']:
        raise ValueError('Frozen cohort dimensions differ')
    audit = json.loads((ROOT / 'data/protocols/PBMC/corrected_legacy_evidence/UPDATED_FOUR_DATASET_INDEPENDENCE_AUDIT.json').read_text())
    record = audit['datasets'][key]
    if legacy.sha(raw_path) != record['rna_sha256']:
        raise ValueError('Raw RNA hash differs from corrected benchmark audit')
    counts, ids, inverse = canonical_genes(counts, ids)
    train_log = normalize_counts(counts[index['train']])
    hvg, scores = select_hvg(train_log)
    train_rna = train_log[:, hvg].toarray()
    val_rna = normalize_counts(counts[index['val']])[:, hvg].toarray()
    mean, std = train_rna.mean(0), train_rna.std(0)
    paired_rows = np.concatenate([index['train'], index['val']])
    atac, peaks, dependencies = paired_atac(key, paired_rows, cells)
    genes = ids[hvg]
    tss = read_tss(legacy.GTF)
    mapped = np.flatnonzero([str(g).split('.')[0] in tss for g in genes])
    graph = dict(candidate_index=np.zeros((2000, 50), np.int64),
                 candidate_mask=np.zeros((2000, 50), bool),
                 distance_bp=np.zeros((2000, 50), np.float32))
    small = legacy.graph_fast(genes[mapped], peaks, tss)
    for name in graph:
        graph[name][mapped] = small[name]
    graph['peak_names'] = peaks
    selector_start = time.perf_counter()
    matrix, diagnostics = build_bs(train_rna, atac[:len(index['train'])], graph)
    selector_seconds = time.perf_counter() - selector_start
    field_start = time.perf_counter()
    field, state, field_report = build_field(atac, matrix, np.arange(len(index['train'])))
    if matrix.nnz == 0:
        raise ValueError('No effective single-source regulatory evidence')
    cache.mkdir(parents=True, exist_ok=False)
    np.save(cache / 'gene_ids.npy', genes)
    np.save(cache / 'all_gene_ids.npy', ids)
    np.save(cache / 'peak_names.npy', peaks)
    np.save(cache / 'hvg_scores.npy', scores)
    np.savez_compressed(cache / 'rna_state.npz', mean=mean, std=std, hvg_index=hvg,
                        raw_gene_inverse=inverse, fit_cell_index=index['train'])
    np.savez_compressed(cache / 'field_state.npz', **state)
    np.savez_compressed(cache / 'candidate_graph.npz', **graph)
    np.savez_compressed(cache / 'regulatory_stability.npz', **diagnostics)
    sparse.save_npz(cache / 'B_S.npz', matrix)
    for split, values in [('train', train_rna), ('val', val_rna)]:
        np.save(cache / f'rna_{split}.npy', values.astype(np.float32))
        np.save(cache / f'{split}_index.npy', index[split])
    np.save(cache / 'test_index.npy', index['test'])
    np.save(cache / 'F_train.npy', field[:len(index['train'])])
    np.save(cache / 'F_val.npy', field[len(index['train']):])
    (cache / 'barcodes.txt').write_text('\n'.join(cells)+'\n')
    dependencies += [raw_path, identity_path, legacy.GTF, Path(__file__), Path(reader.__file__),
                      Path(legacy.__file__), ROOT / 'main/core/regulatory/s_mode/single_source.py']
    processed = legacy.DATA[key] / 'processed'
    dependencies += ([processed / f'{s}_index.npy' for s in index]
                     if (processed / 'train_index.npy').exists()
                     else [legacy.PREPARED[key] / 'metadata.npz'])
    files = list(cache.iterdir()) + dependencies
    manifest = dict(status='PREPARED', dataset=key, identity=expected_identity,
        cohort_cells=len(cells), split_counts={s:len(v) for s,v in index.items()},
        raw_rna_source=str(raw_path), raw_rna_provenance=provenance,
        hvg_count=2000, hvg_formula='variance/mean on full-library log1p Source train; stable ascending argsort tail',
        hvg_candidate_pool='all validated genes, including unsupported genes; no graph-based gene deletion',
        fitting_cells=index['train'].tolist(), target_used=False, labels_used=False,
        test_atac_loaded=False, selector_seed=20020130,
        B_S_shape=list(matrix.shape), candidate_edges=int(graph['candidate_mask'].sum()),
        effective_edges=int(matrix.nnz), supported_genes=2000-len(diagnostics['unsupported_genes']),
        B_S_formula='Top50%(abs(median(rho))*abs(mean(sign(rho)))); selected*median(rho)/gene-wise L1',
        regulatory_diagnostics={name:diagnostics[name].tolist() for name in
            ['zero_variance_genes','zero_variance_peaks','no_candidate_genes','unsupported_genes','zero_denominator_genes']},
        field=field_report, selector_seconds=selector_seconds,
        field_seconds=time.perf_counter()-field_start, preprocessing_seconds=time.perf_counter()-started,
        hashes={str(p):legacy.sha(p) for p in files if p.is_file()})
    legacy.write(cache / 'regulatory_manifest.json', {k:manifest[k] for k in
        ['B_S_formula','B_S_shape','candidate_edges','effective_edges','supported_genes','regulatory_diagnostics','selector_seed']})
    manifest['hashes'][str(cache/'regulatory_manifest.json')] = legacy.sha(cache/'regulatory_manifest.json')
    legacy.write(manifest_path, manifest)
    return cache, manifest
