import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import h5py
from scipy import sparse

ROOT = Path('/home/u202522802045/share/zpb')
OUT = ROOT / 'task38_corrected_pbmc002_ablation'
FOLDS = {f'target_D{i}': (i, [j for j in range(1, 5) if j != i]) for i in range(1, 5)}
PREPARED = {
    1: ROOT / 'V5/v5_final_20260906/data/v5s_prepared/PBMC10k_model_v1',
    2: ROOT / 'final_dmode_b2full_updated_pbmc002/validated_prepared/D2',
    3: ROOT / 'result_contain/S_mode_10x20_multidataset_3fullseed_20260919/dataset_3/prepared',
    4: ROOT / 'result_contain/S_mode_10x20_multidataset_3fullseed_20260919/dataset_4/prepared',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=True) as z:
        return sparse.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))


def target_h5_rna(path, cell_ids, genes):
    digest = hashlib.sha256()
    values = np.zeros((len(cell_ids), len(genes)), dtype=np.float32)
    with h5py.File(path, 'r') as f:
        matrix = f['matrix']
        features = matrix['features']
        rows = np.flatnonzero(features['feature_type'][:].astype(str) == 'Gene Expression')
        ids = [v.split('.')[0] for v in features['id'][rows].astype(str)]
        feature_row = np.full(int(matrix['shape'][0]), -1, dtype=np.int64)
        feature_row[rows] = np.arange(len(rows))
        lookup = {gene_id: i for i, gene_id in enumerate(ids)}
        wanted = np.asarray([lookup.get(g.split('.')[0], -1) for g in genes], dtype=np.int64)
        h5cells = {v: i for i, v in enumerate(matrix['barcodes'][:].astype(str))}
        columns = [h5cells[c] for c in cell_ids]
        indptr = matrix['indptr'][:]
        digest.update('\n'.join(ids).encode())
        totals = np.zeros(len(cell_ids), dtype=np.float32)
        for row, column in enumerate(columns):
            start, stop = int(indptr[column]), int(indptr[column + 1])
            feature_index = matrix['indices'][start:stop]
            positions = np.flatnonzero(feature_row[feature_index] >= 0)
            gene_rows = feature_row[feature_index[positions]]
            counts = matrix['data'][start + positions].astype(np.float32)
            totals[row] = counts.sum(dtype=np.float32)
            digest.update(cell_ids[row].encode())
            digest.update(gene_rows.tobytes())
            digest.update(counts.tobytes())
            target_cols = wanted >= 0
            row_lookup = np.zeros(len(ids), dtype=np.float32)
            row_lookup[gene_rows] = counts
            values[row, target_cols] = row_lookup[wanted[target_cols]]
    assert np.all(totals > 0)
    return np.log1p(values * (1e4 / totals[:, None])).astype(np.float32), int(np.sum(wanted < 0)), digest.hexdigest()


def dataset(i):
    matches = list((ROOT / 'DATA').glob(f'PBMC-MO-00{i}*'))
    assert len(matches) == 1, (i, matches)
    return matches[0]


def build(fold, target_i, source_ids):
    out = OUT / 'cache' / fold
    out.mkdir(parents=True, exist_ok=True)
    source = [PREPARED[i] for i in source_ids]
    ids = [np.load(p / 'gene_ids.npy', allow_pickle=True).astype(str) for p in source]
    names = [np.load(p / 'gene_names.npy', allow_pickle=True).astype(str) for p in source]
    maps = {}
    for gi, gn in zip(ids, names):
        assert len(gi) == len(gn)
        for gene_id, symbol in zip(gi, gn):
            if gene_id in maps:
                assert maps[gene_id] == symbol, (gene_id, maps[gene_id], symbol)
            maps[gene_id] = symbol
    authoritative_genes = ROOT / 'final_dmode_b2full_updated_pbmc002/validated_source_artifacts' / fold / 'gene_ids.npy'
    genes = np.load(authoritative_genes, allow_pickle=True).astype(str)
    assert len(set(genes)) == len(genes) and all(g in maps for g in genes)
    assert len(genes) > 0
    existing = out / 'manifest.json'
    if existing.exists():
        m = json.loads(existing.read_text())
        if m.get('gene_order_source') == str(authoritative_genes) and m.get('cache_sha256') and all(sha(out / k) == v for k, v in m['cache_sha256'].items()) and np.array_equal(np.load(out / 'gene_ids.npy'), genes):
            print('CACHE_VALID', fold, flush=True)
            return
    for p in (out / 'gene_ids.npy', out / 'target_rna_test.npy', out / 'target_test_index.npy', out / 'target_cells.tsv', existing):
        if p.exists():
            p.unlink()
    np.save(out / 'gene_ids.npy', genes)
    d = dataset(target_i)
    proc = d / 'processed'
    barcode_path = proc / 'barcodes.npy'
    if barcode_path.exists():
        cell_names = np.load(barcode_path, allow_pickle=True).astype(str)
    else:
        barcode_path = proc / 'barcodes.txt'
        cell_names = np.asarray(barcode_path.read_text().splitlines(), dtype=str)
    target_meta_path = PREPARED[target_i] / 'metadata.npz'
    with np.load(target_meta_path, allow_pickle=True) as meta:
        test = meta['test_index'].astype(np.int64)
    if target_i == 2:
        target_counts_path = proc / 'rna_counts.npz'
        raw = load_npz(target_counts_path)
        target_symbols_path = proc / 'gene_names.npy'
        target_symbols = np.load(target_symbols_path, allow_pickle=True).astype(str)
        assert raw.shape == (len(cell_names), len(target_symbols))
        symbol_to_col = {symbol: col for col, symbol in enumerate(target_symbols)}
        wanted = np.asarray([symbol_to_col.get(maps[g], -1) for g in genes], dtype=np.int64)
        assert np.all(wanted >= 0), (fold, int(np.sum(wanted < 0)))
        x = raw[test]
        totals = np.asarray(raw.sum(axis=1)).ravel()[test].astype(np.float32)
        selected = x[:, wanted].toarray().astype(np.float32)
        values = np.log1p(selected * (1e4 / np.maximum(totals, 1.0))[:, None]).astype(np.float32)
        missing = int(np.sum(wanted < 0))
        raw_hash = hashlib.sha256()
        raw_hash.update('\n'.join(target_symbols).encode())
        raw_hash.update(values.tobytes())
        target_hash = raw_hash.hexdigest()
        raw_path = target_counts_path
        extra_inputs = [target_symbols_path]
    else:
        raw_path = next((d / 'raw').glob('*.h5'))
        selected_cells = cell_names[test]
        values, missing, target_hash = target_h5_rna(raw_path, selected_cells, genes)
        extra_inputs = []
    assert missing == 0, (fold, missing)
    assert np.isfinite(values).all()
    np.save(out / 'target_rna_test.npy', values)
    np.save(out / 'target_test_index.npy', test)
    with (out / 'target_cells.tsv').open('w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['target_row', 'source_row', 'cell_id'])
        w.writerows((r, int(i), cell_names[i]) for r, i in enumerate(test))
    inputs = [raw_path, barcode_path, target_meta_path, authoritative_genes, *extra_inputs]
    inputs += [p / name for p in source for name in ('gene_ids.npy', 'gene_names.npy', 'rna.npy', 'metadata.npz')]
    hashes = {str(p): sha(p) for p in inputs if not (target_i != 2 and p == raw_path)}
    cache_files = ('gene_ids.npy', 'target_rna_test.npy', 'target_test_index.npy', 'target_cells.tsv')
    manifest = {
        'protocol': 'corrected-quartet strict LODO source-defined gene intersection; target RNA-only frozen inference',
        'fold': fold, 'target': f'PBMC-MO-00{target_i}', 'sources': [f'PBMC-MO-00{i}' for i in source_ids],
        'target_identity': json.loads((d / 'DATASET_IDENTITY.json').read_text())['canonical_identity'],
        'gene_universe': 'ordered intersection of the three SOURCE prepared gene_ids; target excluded',
        'gene_order_source': str(authoritative_genes),
        'gene_count': int(len(genes)), 'target_rows': int(len(test)), 'target_genuine_missing_genes': missing,
        'target_transform': 'raw RNA counts only; all-RNA library normalization to 10000 then log1p; Ensembl ID mapping (D2 uses source-defined ID-to-symbol reference)',
        'target_atac_accessed': False, 'target_labels_accessed': False,
        'target_hvg_or_target_statistics_used': False, 'target_checkpoint_or_hyperparameter_selection': False,
        'target_cell_order_sha256': sha(out / 'target_cells.tsv'),
        'input_sha256': hashes, 'target_raw_rna_sha256': target_hash,
        'target_raw_hash_scope': 'selected target-cell RNA values only; 10x H5 ATAC values not read' if target_i != 2 else 'target RNA-only CSR counts and feature reference',
        'cache_sha256': {n: sha(out / n) for n in cache_files},
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('CACHE_BUILT', fold, len(test), len(genes), missing, flush=True)


for fold, (target_i, sources) in FOLDS.items():
    build(fold, target_i, sources)
