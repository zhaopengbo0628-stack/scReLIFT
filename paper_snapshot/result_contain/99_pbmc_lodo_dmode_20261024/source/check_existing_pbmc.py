import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path('/home/u202522802045/share/zpb')
TASK = ROOT / 'result_contain/99_pbmc_lodo_dmode_20261024'
spec = importlib.util.spec_from_file_location('prep', TASK / 'source/prepare_updated_b2_sources.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)
audit = json.loads((prep.AUDIT / 'UPDATED_FOUR_DATASET_INDEPENDENCE_AUDIT.json').read_text())
processed = prep.DATA['D2'] / 'processed'
reference = next((prep.DATA['D3'] / 'raw').glob('*feature_bc_matrix.h5'))
with h5py.File(reference, 'r') as h:
    features = h['matrix/features']
    mask = features['feature_type'][:].astype(str) == 'Gene Expression'
    ids = features['id'][:].astype(str)[mask]
    names = features['name'][:].astype(str)[mask]
seen, unique = set(), []
for name in names:
    value, suffix = name, 1
    while value in seen:
        value = f'{name}.{suffix}'
        suffix += 1
    unique.append(value)
    seen.add(value)
assert np.array_equal(np.load(processed / 'gene_names.npy'), np.asarray(unique))
assert hashlib.sha256('\n'.join(ids).encode()).hexdigest() == audit['datasets']['D2']['gene_ids_sha256']
with np.load(processed / 'rna_counts.npz') as z:
    shape = tuple(z['shape'])
assert shape == (6317, len(ids))
selected = np.load(prep.PREPARED['D2'] / 'gene_ids.npy').astype(str)
selected_names = np.load(prep.PREPARED['D2'] / 'gene_names.npy').astype(str)
lookup = {g: i for i, g in enumerate(ids)}
assert np.array_equal(names[[lookup[g] for g in selected]], selected_names)
gene_id_path = TASK / 'metadata/PBMC-MO-002_gene_ids.npy'
gene_id_path.parent.mkdir(parents=True, exist_ok=True)
np.save(gene_id_path, ids)
(TASK / 'metadata/PBMC-MO-002_gene_ids.json').write_text(json.dumps({
    'dataset': 'PBMC-MO-002', 'status': 'RESTORED_FROM_VERIFIED_10X_FEATURE_REFERENCE',
    'feature_reference': str(reference), 'gene_count': int(len(ids)),
    'ordered_gene_id_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
    'gene_names_order_matches_processed': True, 'array_sha256': prep.sha(gene_id_path),
}, indent=2) + '\n')
print(json.dumps(dict(D2_gene_order='PASS', genes=int(len(ids)), counts_shape=[int(x) for x in shape],
                     historical_gene_digest='PASS', prepared_HVG_mapping='PASS', reference=str(reference))))
old = ROOT / 'main/result/d_mode/corrected_pbmc_legacy'
for i in range(1, 5):
    fold = f'target_D{i}'
    folder = old / 'validated_cache' / fold
    manifest = prep.checked(folder / 'manifest.json')
    current_genes = np.load(prep.OUT / 'validated_source_artifacts' / fold / 'gene_ids.npy')
    old_genes = np.load(old / 'validated_source_artifacts' / fold / 'gene_ids.npy')
    assert np.array_equal(current_genes, old_genes)
    x = np.load(folder / 'target_rna_test.npy')
    assert x.shape == (manifest['rows'], len(current_genes)) and np.isfinite(x).all()
    assert manifest['missing_genes'] == 0
    assert manifest['TARGET_ATAC_LOADED'] is False and manifest['TARGET_LABEL_LOADED'] is False
    cells = (prep.DATA[f'D{i}'] / 'processed/barcodes.txt').read_text().splitlines()
    index = np.load(folder / 'target_test_index.npy')
    assert (folder / 'cell_order.tsv').read_text().splitlines()[1:] == [cells[j] for j in index]
    print(json.dumps(dict(fold=fold, cache_hashes='PASS', source_gene_order='IDENTICAL',
                         rows=int(len(index)), genes=int(x.shape[1]), nonzero=int(np.count_nonzero(x)),
                         missing_genes=0, barcode_order='PASS', existing_cache=str(folder))))
status_path = TASK / 'status.json'
status = json.loads(status_path.read_text())
status.update(status='PREPARED_SMOKE_PASS_WAITING_GPU', completed_units=0, smoke='PASS',
              source_artifacts={f'target_D{i}': 'PASS' for i in range(1, 5)},
              target_rna_cache={f'target_D{i}': 'REUSED_HASH_VERIFIED' for i in range(1, 5)},
              target_atac_used=False, target_labels_before_embedding=False,
              blocker='Task98 BMMC queue owns GPU; formal PBMC units not started')
status_path.write_text(json.dumps(status, indent=2) + '\n')
plan_path = TASK / 'plan.json'
plan = json.loads(plan_path.read_text())
plan['status'] = 'PREPARED_SMOKE_PASS_WAITING_GPU'
plan_path.write_text(json.dumps(plan, indent=2) + '\n')
