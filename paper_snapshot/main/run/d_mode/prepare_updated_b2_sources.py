"""Source-only preparation for the corrected Full B2 quartet."""
import argparse
import ast
import gzip
import hashlib
import json
import sys
import time
from bisect import bisect_left, bisect_right
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse

ROOT = Path('/home/u202522802045/share/zpb')
OUT = ROOT / 'main/result/d_mode/corrected_pbmc_legacy'
sys.path.insert(0, str(ROOT / 'V5/v5_final_20260906'))
sys.path.insert(0, str(ROOT / 'main/core/regulatory/common/legacy_support'))
from v5d.data import V5DData
from v5d.dataset_spec import DatasetSpec
from v5d.harmonize import prepare_harmonized
from v5d.dmode_reliable_selector import DModeReliabilityConfig, build_source_only_b
from v5d.dmode_teacher import fit_dmode_field
from v5d.regulatory_graph import read_tss, parse_peak, normalize_chromosome, build_candidate_graph
from v5s.data import normalize_log1p

NAMES = {f'D{i}': f'PBMC-MO-00{i}' for i in range(1, 5)}
PREPARED = {
    'D1': ROOT / 'V5/v5_final_20260906/data/v5s_prepared/PBMC10k_model_v1',
    'D2': OUT / 'validated_prepared/D2',
    'D3': ROOT / 'main/result/s_mode/intra/S_mode_10x20_multidataset_3fullseed_20260919/dataset_3/prepared',
    'D4': ROOT / 'main/result/s_mode/intra/S_mode_10x20_multidataset_3fullseed_20260919/dataset_4/prepared',
}
DATA = {key: ROOT / 'data/benchmark/PBMC' / name / 'legacy_corrected_refs_v1/native_access'
        for key, name in NAMES.items()}
GTF = Path('/home/u202522802045/glue_156478/graph_files/gencode.v32.primary_assembly.annotation.gtf.gz')
STORAGE_MAP = ROOT / 'main/configs/storage_paths_20261022.json'
AUDIT = ROOT / 'data/protocols/PBMC/corrected_legacy_evidence'


def relocate(path):
    value = str(Path(path).absolute())
    if STORAGE_MAP.exists():
        mappings = json.loads(STORAGE_MAP.read_text()).get('mapping', {})
        for _ in range(16):
            changed = False
            for old, new in sorted(mappings.items(), key=lambda item: -len(item[0])):
                if value == old or value.startswith(old.rstrip('/') + '/'):
                    value = new + value[len(old):]
                    changed = True
                    break
            if not changed:
                break
    return Path(value)


def cache_dir(cache_root, key, version='pbmc_corrected_s10x20_v1_20261012'):
    return Path(cache_root) / NAMES[key] / version


def sha(path):
    path = relocate(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def expected_hash(hashes, path):
    current = relocate(path).resolve()
    matches = {digest for name, digest in hashes.items() if relocate(name).resolve() == current}
    if len(matches) != 1:
        raise ValueError('Missing or ambiguous historical hash: ' + str(current))
    return matches.pop()


def verify_sha(path, expected):
    current = relocate(path)
    if sha(current) == expected:
        return
    state = json.loads(STORAGE_MAP.read_text())
    record = state.get('source_backups', {}).get(str(current))
    if (record and record['before_sha256'] == expected
            and sha(record['backup']) == expected
            and sha(current) == record['after_sha256']):
        return
    raise ValueError('Historical source checksum mismatch after path migration: '+str(path))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def checked(path):
    value = json.loads(path.read_text())
    for filename, digest in value['hashes'].items():
        verify_sha(filename, digest)
    return value


def graph_fast(genes, peaks, tss, window=250000, cap=50):
    by_chrom = {}
    for i, peak in enumerate(peaks):
        chrom, start, end = parse_peak(str(peak))
        by_chrom.setdefault(chrom, []).append(((start + end) // 2, i))
    indexed = {}
    for chrom, entries in by_chrom.items():
        entries.sort()
        indexed[chrom] = (entries, [v[0] for v in entries])
    index = np.zeros((len(genes), cap), np.int64)
    mask = np.zeros_like(index, bool)
    distance = np.zeros_like(index, np.float32)
    for i, gene in enumerate(genes):
        chrom, position = tss[str(gene).split('.')[0]]
        entries, positions = indexed.get(normalize_chromosome(chrom), ([], []))
        left, right = bisect_left(positions, position-window), bisect_right(positions, position+window)
        values = sorted((abs(mid-position), j) for mid, j in entries[left:right])[:cap]
        for k, (dist, j) in enumerate(values):
            index[i, k], mask[i, k], distance[i, k] = j, True, dist
    return dict(candidate_index=index, candidate_mask=mask, distance_bp=distance)


def prepare_d2():
    destination = PREPARED['D2']
    manifest = destination / 'manifest.json'
    if manifest.exists():
        return checked(manifest)
    processed = DATA['D2'] / 'processed'
    identity = json.loads((DATA['D2'] / 'DATASET_IDENTITY.json').read_text())
    assert identity['canonical_identity'] == 'GEO:GSE196990:GSE196988+GSE196987:HDa'
    reference = next((DATA['D3'] / 'raw').glob('*feature_bc_matrix.h5'))
    with h5py.File(reference, 'r') as h:
        ft = h['matrix/features/feature_type'][:].astype(str) == 'Gene Expression'
        ids = h['matrix/features/id'][:].astype(str)[ft]
        names = h['matrix/features/name'][:].astype(str)[ft]
    seen, unique = set(), []
    for name in names:
        value, suffix = name, 1
        while value in seen:
            value = f'{name}.{suffix}'
            suffix += 1
        unique.append(value)
        seen.add(value)
    assert np.array_equal(np.load(processed / 'gene_names.npy'), np.asarray(unique))
    raw = sparse.load_npz(processed / 'rna_counts.npz').tocsr()
    atac = sparse.load_npz(processed / 'atac.npz').tocsr()
    peaks = np.load(processed / 'peak_names.npy').astype(str)
    assert raw.shape == (6317, 36601) and atac.shape[0] == 6317
    splits = {key: np.load(processed / f'{key}_index.npy') for key in ['train', 'val', 'test']}
    assert np.unique(np.concatenate(list(splits.values()))).size == 6317
    tss = read_tss(GTF)
    mapped = np.flatnonzero([g in tss for g in ids])
    graph = graph_fast(ids[mapped], peaks, tss)
    candidates = mapped[np.flatnonzero(graph['candidate_mask'].any(1))]
    original = ROOT / 'V5/v5_final_20260906/prepare_pbmc10k_model_v1.py'
    tree = ast.parse(original.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'hvg_train_indices')
    namespace = dict(np=np, sparse=sparse, normalize_log1p=normalize_log1p)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(original), 'exec'), namespace)
    selected = namespace['hvg_train_indices'](raw, splits['train'], candidates, 2000)
    final_graph = graph_fast(ids[selected], peaks, tss)
    reference_graph, _ = build_candidate_graph(ids[selected[:8]], peaks, tss, 250000, 50)
    assert all(np.array_equal(final_graph[k][:8], reference_graph[k]) for k in reference_graph)
    kept_peaks = np.unique(final_graph['candidate_index'][final_graph['candidate_mask']])
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / 'rna.npy', normalize_log1p(raw)[:, selected].toarray().astype(np.float32))
    np.save(destination / 'gene_ids.npy', ids[selected])
    np.save(destination / 'gene_names.npy', names[selected])
    np.save(destination / 'peak_names.npy', peaks[kept_peaks])
    sparse.save_npz(destination / 'atac_binary.npz', atac[:, kept_peaks].astype(bool).astype(np.float32))
    np.savez_compressed(destination / 'metadata.npz', **{f'{k}_index': v for k, v in splits.items()})
    inputs = [processed / 'rna_counts.npz', processed / 'atac.npz', processed / 'gene_names.npy',
              processed / 'peak_names.npy', DATA['D2'] / 'DATASET_IDENTITY.json', reference, original, GTF]
    inputs += [processed / f'{k}_index.npy' for k in splits]
    files = list(destination.glob('*')) + inputs
    value = dict(identity=identity['canonical_identity'], cells=6317, genes=2000,
                 preparation='frozen train-only variance/mean HVG among graph-connected genes; full-library log1p',
                 window_bp=250000, max_candidates=50, labels_loaded=False,
                 hashes={str(p): sha(p) for p in files if p.is_file()})
    write(manifest, value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', choices=list(NAMES))
    args = parser.parse_args()
    audit = json.loads((AUDIT / 'UPDATED_FOUR_DATASET_INDEPENDENCE_AUDIT.json').read_text())
    assert audit['FOUR_DATASET_INDEPENDENCE'] == 'PASS'
    for record in audit['datasets'].values():
        assert sha(record['rna_path']) == record['rna_sha256']
    for target in ([args.fold] if args.fold else list(NAMES)):
        sources = [key for key in NAMES if key != target]
        folder = OUT / 'validated_source_artifacts' / f'target_{target}'
        if (folder / 'manifest.json').exists():
            checked(folder / 'manifest.json')
            print('VERIFIED_EXISTING', target, flush=True)
            continue
        if 'D2' in sources:
            prepare_d2()
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / 'harmonized').exists():
            raise RuntimeError(f'Incomplete harmonization must be inspected: {folder}')
        started = time.perf_counter()
        specs = [DatasetSpec(NAMES[key], PREPARED[key]) for key in sources]
        prepare_harmonized(specs, folder / 'harmonized', feature_reference_names=[s.name for s in specs])
        genes = np.load(folder / 'harmonized/common_gene_ids.npy').astype(str)
        peaks = np.load(folder / 'harmonized/canonical_peak_names.npy').astype(str)
        graph = graph_fast(genes, peaks, read_tss(GTF))
        native, _ = build_candidate_graph(genes[:8], peaks, read_tss(GTF), 250000, 50)
        assert all(np.array_equal(graph[k][:8], native[k]) for k in native)
        graph['peak_names'] = peaks
        np.savez_compressed(folder / 'candidate_graph.npz', **graph)
        datasets = [V5DData(folder / 'harmonized', NAMES[key]) for key in sources]
        config = DModeReliabilityConfig()
        selection = build_source_only_b(datasets, graph, genes, config, 20020130)
        sparse.save_npz(folder / 'B_D.npz', selection['B_D'])
        for key, value in selection.items():
            if isinstance(value, np.ndarray):
                np.save(folder / f'{key}.npy', value)
            elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                np.savez_compressed(folder / f'{key}.npz', **{s.name: v for s, v in zip(datasets, value)})
        np.save(folder / 'gene_ids.npy', genes)
        b_seconds = time.perf_counter() - started
        started = time.perf_counter()
        for source in datasets:
            field, state, fm = fit_dmode_field(source.atac, selection['B_D'], source.train)
            np.save(folder / f'{source.name}_F_train.npy', field[source.train])
            np.save(folder / f'{source.name}_F_val.npy', field[source.val])
            np.savez_compressed(folder / f'{source.name}_field_state.npz', **state)
            write(folder / f'{source.name}_field_manifest.json', fm)
        code = [ROOT / 'main/core/regulatory/common/legacy_support/v5d' / name for name in ['harmonize.py', 'dmode_reliable_selector.py', 'dmode_teacher.py', 'regulatory_graph.py']]
        inputs = [p for key in sources for p in PREPARED[key].glob('*') if p.is_file()]
        outputs = [p for p in folder.rglob('*') if p.is_file()]
        write(folder / 'manifest.json', dict(status='SOURCE_ARTIFACTS_COMPLETE', target=target,
              sources=sources, gene_count=len(genes), config=asdict(config), graph_window=250000,
              graph_candidates=50, selection=selection['manifest'],
              target_atac_loaded=False, target_labels_loaded=False, target_hvg_fitted=False,
              target_pca_fitted=False, B_D_BUILD_TIME_SEC=b_seconds,
              F_BUILD_TIME_SEC=time.perf_counter()-started,
              hashes={str(p): sha(p) for p in outputs + inputs + code + [GTF, Path(__file__)]}))
        print('SOURCE_COMPLETE', target, len(genes), selection['B_D'].nnz, flush=True)


if __name__ == '__main__':
    main()
