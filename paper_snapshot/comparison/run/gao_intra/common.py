"""Frozen Gao experiment inputs and shared, posthoc-only evaluation."""
import hashlib
import importlib.util
import json
import time
from pathlib import Path

ROOT = Path('/home/u202522802045/share/zpb')
SEEDS = [20020130, 20020131, 20020132]
TIER = 'pre_final_explicit_3seeds'
NAME = 'adult_cortex_author_processed_intra_pre_final_20261023'
CACHE = ROOT/'data/cache/s_mode/Mouse-brain'/NAME/'GaoVIS'
BENCH = ROOT/'data/benchmark/Mouse-brain'/NAME/'GaoVIS'
SOURCE = ROOT/'data/benchmark/Mouse-brain/adult_cortex_author_processed_v1_20261022/GaoVIS'
METHODS = ['b2', 'rna_encoder', 'scmomer', 'scbutterfly', 'multigrate', 'cobolt', 'palette', 'eclare', 'rna_pca']
NAMES = dict(zip(METHODS, ['Full B2', 'RNA-only Encoder', 'scMomer-S', 'scButterfly-B', 'Multigrate', 'Cobolt', 'Palette-S', 'ECLARE-S', 'RNA PCA']))
SCORES = ['ARI', 'NMI', 'AMI', 'ACC', 'Macro-F1', 'Balanced ACC', 'Weighted-F1', 'Purity@15', 'Silhouette']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def sync(device):
    if str(device).startswith('cuda'):
        import torch
        torch.cuda.synchronize()


def stamp(device='cpu'):
    sync(device)
    return time.perf_counter()


def memory(device):
    if str(device).startswith('cuda'):
        import torch
        sync(device)
        return dict(allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                    reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
    return dict(allocated_GiB=None, reserved_GiB=None)


def inference(predict, x, device, batch=256):
    import numpy as np
    count = x.shape[0]
    predict(x[:min(batch, count)])
    sync(device)
    if str(device).startswith('cuda'):
        import torch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    tick = stamp(device)
    z = predict(x)
    seconds = stamp(device)-tick
    peak = memory(device)
    assert len(z) == count and np.isfinite(z).all()
    return z, dict(inference_seconds=seconds, cells_per_second=count/seconds,
                   inference_batch=batch, inference_peak=peak)


def score(unit, smoke=False):
    import numpy as np
    import pandas as pd
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
        adjusted_mutual_info_score, accuracy_score, f1_score, balanced_accuracy_score, silhouette_score)
    from sklearn.neighbors import NearestNeighbors
    path = unit/'test_embedding.npy'
    digest = sha(path)
    z = np.load(path, mmap_mode='r')
    # This is the first label read in a method process, after prediction persistence.
    meta = pd.read_csv(BENCH/'metadata.csv', keep_default_na=False)
    idx = np.load(CACHE/'test_index.npy')[:len(z)]
    labels = meta.subclass_label.to_numpy(str)[idx]
    _, y = np.unique(labels, return_inverse=True)
    if not smoke:
        assert len(z) == 5280 and len(np.unique(y)) == 25
    scaled = StandardScaler().fit_transform(z)
    clusters = KMeans(n_clusters=len(np.unique(y)), init='k-means++', n_init=100,
        random_state=20020130, max_iter=1000, tol=1e-4, algorithm='lloyd').fit_predict(scaled)
    table = np.zeros((len(np.unique(y)), len(np.unique(y))), dtype=np.int64)
    np.add.at(table, (y, clusters), 1)
    a, b = linear_sum_assignment(-table)
    mapping = dict(zip(b, a))
    matched = np.array([mapping[c] for c in clusters])
    neighbors = NearestNeighbors(n_neighbors=16).fit(scaled).kneighbors(scaled, return_distance=False)
    neighbors = np.asarray([row[row != i][:15] for i, row in enumerate(neighbors)])
    values = [adjusted_rand_score(y, clusters), normalized_mutual_info_score(y, clusters, average_method='arithmetic'),
        adjusted_mutual_info_score(y, clusters), accuracy_score(y, matched),
        f1_score(y, matched, average='macro', zero_division=0), balanced_accuracy_score(y, matched),
        f1_score(y, matched, average='weighted', zero_division=0), np.mean(y[neighbors] == y[:, None]),
        silhouette_score(scaled, y, metric='euclidean')]
    result = dict(zip(SCORES, map(float, values)))
    result.update(cells=len(y), classes=len(np.unique(y)), known_K=True,
        silhouette_labels='true author subclass', labels_loaded_after_embedding_frozen=True,
        label_sha256=hashlib.sha256('\n'.join(labels).encode()).hexdigest(),
        test_cell_ids_sha256=hashlib.sha256('\n'.join(meta.barcode.to_numpy(str)[idx]).encode()).hexdigest(),
        metadata_sha256=sha(BENCH/'metadata.csv'), embedding_sha256=digest)
    pd.DataFrame(dict(barcode=meta.barcode.to_numpy(str)[idx], author_subclass=labels,
                      cluster=clusters, matched_class=matched)).to_csv(unit/'test_cell_scores.csv', index=False)
    assert sha(path) == digest
    write(unit/'metrics.json', result)
    return result
