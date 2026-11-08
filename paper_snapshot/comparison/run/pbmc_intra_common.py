"""Read existing corrected PBMC caches with fixed splits and RNA-only test input."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, adjusted_mutual_info_score,
    adjusted_rand_score, balanced_accuracy_score, f1_score,
    normalized_mutual_info_score, silhouette_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = ROOT / 'data/cache/s_mode/PBMC'
SEEDS = list(range(20020130, 20020135))
EVALUATOR = dict(init='k-means++', n_init=100, random_state=20020130,
                 max_iter=1000, tol=1e-4, algorithm='lloyd')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def sha(path):
    path = relocated(path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def cache_for(key, version):
    dataset = f'PBMC-MO-00{key[-1]}'
    return CACHE_ROOT / dataset / version


def benchmark_for(key):
    dataset = f'PBMC-MO-00{key[-1]}'
    return ROOT / 'data/benchmark/PBMC' / dataset / 'legacy_corrected_refs_v1/legacy_refs'


def relocated(path):
    value = str(Path(path).absolute())
    state = read(ROOT/'main/configs/storage_paths_20261022.json')
    mappings = state.get('mapping', {})
    for _ in range(16):
        changed = False
        for old, new in sorted(mappings.items(), key=lambda item: -len(item[0])):
            if value == old or value.startswith(old.rstrip('/') + '/'):
                value = new + value[len(old):]
                changed = True
                break
        if not changed:
            return Path(value)
    raise ValueError('Storage mapping cycle: '+str(path))


def expected_hash(hashes, path):
    current = relocated(path).resolve()
    matches = {digest for name,digest in hashes.items() if relocated(name).resolve() == current}
    if len(matches) != 1:
        raise ValueError('Missing or ambiguous historical hash: '+str(current))
    return matches.pop()


def verify_cache(cache):
    manifest = read(cache/'manifest.json')
    if any(manifest[k] for k in ('test_atac_loaded', 'labels_used_for_fitting', 'test_used_for_fitting')):
        raise ValueError('Invalid fitting boundary')
    for name, digest in manifest['cache_sha256'].items():
        if sha(cache/name) != digest:
            raise ValueError(f'Cache checksum mismatch: {name}')
    return manifest


def prepare(key):
    cache = cache_for(key, 'pbmc_comparison_intra_v1')
    if not (cache/'manifest.json').is_file():
        raise FileNotFoundError(f'Processed benchmark cache required: {cache}; raw-data fallback is disabled')
    manifest = verify_cache(cache)
    config = read(ROOT/'main/configs/smode_pbmc_corrected.json')
    identity_path = benchmark_for(key)/'DATASET_IDENTITY.json'
    identity = read(identity_path)
    base = cache_for(key, 'pbmc_corrected_s10x20_v1_20261012')
    base_manifest = read(base/'manifest.json')
    if any(value != config['identities'][key] for value in
           (identity['canonical_identity'], manifest['identity'], base_manifest['identity'])):
        raise ValueError(f'Processed benchmark identity mismatch: {key}')
    for path in (identity_path, base/'manifest.json'):
        if sha(path) != expected_hash(manifest['source_sha256'],path):
            raise ValueError(f'Processed benchmark provenance changed: {path}')
    for name in ('gene_ids.npy', 'peak_names.npy', 'train_index.npy', 'val_index.npy', 'test_index.npy', 'barcodes.txt'):
        path = base/name
        if sha(path) != expected_hash(base_manifest['hashes'],path):
            raise ValueError(f'Full B2 processed input changed: {path}')
        if name == 'peak_names.npy':
            columns = np.load(cache/'atac_state.npz')['peak_columns']
            np.testing.assert_array_equal(np.load(cache/'peak_ids.npy'), np.load(path)[columns])
        elif name.endswith('.npy'):
            np.testing.assert_array_equal(np.load(cache/name), np.load(path))
        elif (cache/name).read_text() != path.read_text():
            raise ValueError(f'Processed barcode order mismatch: {key}')
    return cache


def load(cache, smoke=False):
    manifest = verify_cache(cache)
    data = {}
    for split in ('train','val','test'):
        limit = {'train':128,'val':64,'test':64}[split] if smoke else None
        for modality in ('rna','rna_log'):
            data[f'{modality}_{split}'] = sparse.load_npz(cache/f'{modality}_{split}.npz')[:limit].tocsr()
        if split != 'test':
            for modality in ('atac','atac_tfidf'):
                data[f'{modality}_{split}'] = sparse.load_npz(cache/f'{modality}_{split}.npz')[:limit].tocsr()
    return data, manifest


def evaluate(unit, cache, key):
    """Posthoc clustering only; no classification/label-transfer claim."""
    prediction = unit/'rna_test_embedding.npy'
    frozen_hash = sha(prediction)
    label_path = benchmark_for(key)/'label.csv'
    with label_path.open(newline='') as handle:
        raw = np.asarray([r['cell_type'] for r in csv.DictReader(handle)])
    test = np.load(cache/'test_index.npy')
    _, labels = np.unique(raw[test], return_inverse=True)
    z = np.load(prediction)
    assert len(z) == len(labels) and np.isfinite(z).all()
    z = StandardScaler().fit_transform(z)
    k = len(np.unique(labels))
    clusters = KMeans(n_clusters=k, **EVALUATOR).fit_predict(z)
    table = np.zeros((k,k), np.int64)
    np.add.at(table,(labels,clusters),1)
    a,b = linear_sum_assignment(-table)
    mapping = dict(zip(b,a))
    matched = np.asarray([mapping[c] for c in clusters])
    neighbors = NearestNeighbors(n_neighbors=16).fit(z).kneighbors(z,return_distance=False)
    neighbors = np.asarray([row[row!=i][:15] for i,row in enumerate(neighbors)])
    acc = float(accuracy_score(labels,matched))
    metrics = dict(ari=float(adjusted_rand_score(labels,clusters)),
        nmi=float(normalized_mutual_info_score(labels,clusters)),
        ami=float(adjusted_mutual_info_score(labels,clusters)), cluster_acc=acc, accuracy=acc,
        balanced_accuracy=float(balanced_accuracy_score(labels,matched)),
        macro_f1=float(f1_score(labels,matched,average='macro',zero_division=0)),
        weighted_f1=float(f1_score(labels,matched,average='weighted',zero_division=0)),
        knn_purity_at_15=float(np.mean(labels[neighbors]==labels[:,None])),
        silhouette=float(silhouette_score(z,labels)) if 1<k<len(labels) else None,
        geometry_pearson=None, geometry_spearman=None,
        annotation_scope='Posthoc test Hungarian cluster scores; classification metrics are NOT label transfer',
        silhouette_labels='true cell types', evaluator=EVALUATOR,
        embedding_scaling='test StandardScaler for posthoc scoring only; model remains frozen',
        labels_loaded_after_prediction_frozen=True, test_atac_used=False,
        prediction_sha256=frozen_hash, label_file_sha256=sha(label_path))
    assert sha(prediction) == frozen_hash
    write(unit/'metrics.json', metrics)
    with (unit/'metrics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(metrics));writer.writeheader();writer.writerow(metrics)
    names = ['ari','nmi','ami','cluster_acc','accuracy','balanced_accuracy','macro_f1','weighted_f1','knn_purity_at_15','silhouette','geometry_spearman','geometry_pearson']
    (unit/'metrics.md').write_text('Posthoc clustering; pre_final. ATAC geometry unavailable.\n\n| '+' | '.join(names)+' |\n|'+ '---|'*len(names)+'\n| '+' | '.join('NA' if metrics[n] is None else f'{metrics[n]:.6f}' for n in names)+' |\n')
