"""Donor-held-out AD adapter; reuse frozen S-mode preprocessing and Full B2 training."""
import argparse
import csv
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse

SEEDS = list(range(20020130, 20020135))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def donor_split(rows, seed=20020130):
    donors = {}
    for row in rows:
        donor, disease = row['donor_id'], row['disease_status']
        if disease not in ('AD', 'Ctrl') or not donor:
            raise ValueError('Missing donor or invalid disease state')
        if donor in donors and donors[donor] != disease:
            raise ValueError('Contradictory donor diagnosis')
        donors[donor] = disease
    rng = np.random.default_rng(seed)
    groups = {s: [] for s in ('train', 'val', 'test')}
    for disease, expected in [('AD', 7), ('Ctrl', 8)]:
        values = np.asarray(sorted(d for d in donors if donors[d] == disease))
        if len(values) != expected:
            raise ValueError('AD cohort donor count changed')
        values = rng.permutation(values)
        groups['test'].append(str(values[0]))
        groups['val'].append(str(values[1]))
        groups['train'].extend(values[2:].tolist())
    index = {s: np.flatnonzero([r['donor_id'] in ds for r in rows]) for s, ds in groups.items()}
    validate_split(rows, index)
    return index, groups


def validate_split(rows, index):
    if not np.array_equal(np.sort(np.concatenate(list(index.values()))), np.arange(len(rows))):
        raise ValueError('Incomplete or overlapping cell split')
    groups = []
    for s in ('train', 'val', 'test'):
        subset = [rows[int(i)] for i in index[s]]
        if {r['disease_status'] for r in subset} != {'AD', 'Ctrl'}:
            raise ValueError(f'{s} must contain both disease states')
        groups.append({r['donor_id'] for r in subset})
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Donor leakage')


def read_counts(path, rows, feature_index, binary=False):
    """Read only requested modality values of requested cells from sorted 10x CSC."""
    rows, feature_index = np.asarray(rows, np.int64), np.asarray(feature_index, np.int64)
    if not len(feature_index) or not np.all(np.diff(feature_index) == 1):
        raise ValueError('Expected contiguous modality feature block')
    blocks = []
    with h5py.File(path, 'r', rdcc_nbytes=64 << 20) as handle:
        matrix = handle['matrix']
        pointers = matrix['indptr'][:]
        if len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(rows >= len(pointers)-1):
            raise ValueError('Invalid cell indices')
        for start in range(0, len(rows), 512):
            data, indices, ptr = [], [], [0]
            for cell in rows[start:start+512]:
                lo, hi = int(pointers[cell]), int(pointers[cell+1])
                ix = matrix['indices'][lo:hi]
                left = int(np.searchsorted(ix, feature_index[0]))
                right = int(np.searchsorted(ix, feature_index[-1], side='right'))
                values = matrix['data'][lo+left:lo+right].astype(np.float32)
                if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
                    raise ValueError('Invalid raw modality counts')
                data.append((values != 0).astype(np.float32) if binary else values)
                indices.append(ix[left:right] - feature_index[0])
                ptr.append(ptr[-1] + right-left)
            blocks.append(sparse.csr_matrix((np.concatenate(data), np.concatenate(indices), ptr),
                                            shape=(len(ptr)-1, len(feature_index))))
    return sparse.vstack(blocks, format='csr')


def runtime(source):
    spec = importlib.util.spec_from_file_location('ad_frozen_smode', source / 'frozen_train.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_hashes(hashes):
    for filename, digest in hashes.items():
        if sha(filename) != digest:
            raise ValueError(f'Hash changed: {filename}')


def prepare(config, runner, split_validator=validate_split):
    from main.core.regulatory.s_mode import single_source as reg
    from v5d.regulatory_graph import read_tss
    benchmark, cache = Path(config['benchmark']), Path(config['cache']) / 'AD'
    if (cache / 'manifest.json').exists():
        manifest = runner.legacy.checked(cache / 'manifest.json')
        if manifest['protocol_sha256'] != sha(config['protocol_path']):
            raise ValueError('Cache protocol mismatch')
        check_cache(cache, manifest)
        return cache, manifest
    if cache.exists():
        raise ValueError('Incomplete cache; inspect before retry, do not overwrite')
    started = time.perf_counter()
    benchmark_manifest = json.loads((benchmark / 'manifest.json').read_text())
    verify_hashes({str(benchmark / name): digest for name, digest in benchmark_manifest['output_hashes'].items()})
    with (benchmark / 'cells.csv').open() as handle:
        cells = list(csv.DictReader(handle))
    if not np.array_equal(np.load(benchmark/'raw_cell_index.npy'), np.arange(len(cells))):
        raise ValueError('Benchmark/H5 row indices changed')
    split = {s: np.load(Path(config['split']) / f'{s}_index.npy') for s in ('train', 'val', 'test')}
    split_validator(cells, split)
    raw = benchmark / benchmark_manifest['matrix']
    rna_ix = np.load(benchmark / 'rna_feature_index.npy')
    atac_ix = np.load(benchmark / 'atac_feature_index.npy')
    with h5py.File(raw, 'r') as handle:
        matrix = handle['matrix']
        barcodes = matrix['barcodes'][:].astype(str)
        ids = matrix['features/id'][:].astype(str)[rna_ix]
        peaks = matrix['features/id'][:].astype(str)[atac_ix]
        if list(barcodes) != [c['barcode'] for c in cells] or len(set(barcodes)) != len(cells):
            raise ValueError('Barcode identity/order mismatch')
    if len(set(peaks)) != len(peaks):
        raise ValueError('Duplicate peaks')
    counts, genes, inverse = reg.canonical_genes(read_counts(raw, split['train'], rna_ix), ids)
    log = reg.normalize_counts(counts)
    hvg, scores = reg.select_hvg(log)
    train_rna = log[:, hvg].toarray().astype(np.float32)
    del counts, log
    val_counts, val_genes, _ = reg.canonical_genes(read_counts(raw, split['val'], rna_ix), ids)
    np.testing.assert_array_equal(genes, val_genes)
    val_rna = reg.normalize_counts(val_counts)[:, hvg].toarray().astype(np.float32)
    del val_counts
    selected = genes[hvg]
    mean, std = train_rna.mean(0), train_rna.std(0)
    paired = np.concatenate([split['train'], split['val']])
    if np.intersect1d(paired, split['test']).size:
        raise ValueError('Held-out ATAC leakage')
    atac = read_counts(raw, paired, atac_ix, binary=True)
    tss = read_tss(Path(config['gtf']))
    mapped = np.flatnonzero([g in tss for g in selected])
    graph = dict(candidate_index=np.zeros((2000, 50), np.int64),
                 candidate_mask=np.zeros((2000, 50), bool), distance_bp=np.zeros((2000, 50), np.float32))
    small = runner.legacy.graph_fast(selected[mapped], peaks, tss)
    for name in graph:
        graph[name][mapped] = small[name]
    graph['peak_names'] = peaks
    timer = time.perf_counter()
    matrix, diagnostic = reg.build_bs(train_rna, atac[:len(split['train'])], graph)
    selector_seconds = time.perf_counter()-timer
    if not matrix.nnz:
        raise ValueError('No valid regulatory evidence')
    timer = time.perf_counter()
    field, state, field_report = reg.build_field(atac, matrix, np.arange(len(split['train'])))
    field_seconds = time.perf_counter()-timer
    cache.mkdir(parents=True, exist_ok=False)
    np.save(cache / 'gene_ids.npy', selected)
    np.save(cache / 'all_gene_ids.npy', genes)
    np.save(cache / 'peak_names.npy', peaks)
    np.save(cache / 'hvg_scores.npy', scores)
    np.savez_compressed(cache / 'rna_state.npz', mean=mean, std=std, hvg_index=hvg,
                        raw_gene_inverse=inverse, fit_cell_index=split['train'])
    np.savez_compressed(cache / 'field_state.npz', **state, fit_cell_index=split['train'])
    np.savez_compressed(cache / 'candidate_graph.npz', **graph)
    np.savez_compressed(cache / 'regulatory_stability.npz', **diagnostic)
    sparse.save_npz(cache / 'B_S.npz', matrix)
    for s, values in [('train', train_rna), ('val', val_rna)]:
        np.save(cache / f'rna_{s}.npy', values)
    np.save(cache / 'F_train.npy', field[:len(split['train'])])
    np.save(cache / 'F_val.npy', field[len(split['train']):])
    for s, index in split.items():
        np.save(cache / f'{s}_index.npy', index)
    (cache / 'barcodes.txt').write_text('\n'.join(barcodes)+'\n')
    manifest = dict(status='PREPARED', identity='GEO:GSE214979:author_QC_105332',
        protocol_sha256=sha(config['protocol_path']), benchmark_manifest_sha256=sha(benchmark/'manifest.json'),
        split_counts={s: len(ix) for s, ix in split.items()}, fitting_cells=split['train'].tolist(),
        target_used=False, labels_used=False, test_atac_loaded=False, hvg_count=2000,
        hvg_formula='frozen single-Source variance/mean; full-library 10000/log1p; Source train only',
        B_S_formula='score=abs(median(rho))*abs(mean(sign(rho))); per-gene Top50%; signed median / gene L1',
        selector_seed=SEEDS[0], tss_mapped_genes=len(mapped), candidate_edges=int(graph['candidate_mask'].sum()),
        effective_edges=int(matrix.nnz), supported_genes=2000-len(diagnostic['unsupported_genes']),
        field=field_report, selector_seconds=selector_seconds, field_seconds=field_seconds,
        preprocessing_seconds=time.perf_counter()-started, research_diagnostic_policy='WARNING_ONLY',
        hashes={str(p): sha(p) for p in cache.iterdir() if p.is_file()})
    manifest['hashes'].update(config['code_hashes'])
    manifest['hashes'].update({str(raw): sha(raw), str(benchmark/'manifest.json'): sha(benchmark/'manifest.json'),
                               config['gtf']: sha(config['gtf']), config['protocol_path']: sha(config['protocol_path'])})
    for s in split:
        p = Path(config['split']) / f'{s}_index.npy'
        manifest['hashes'][str(p)] = sha(p)
    write(cache / 'manifest.json', manifest)
    check_cache(cache, manifest)
    return cache, manifest


def check_cache(cache, manifest):
    from main.core.regulatory.s_mode.single_source import finite
    split = {s: np.load(cache / f'{s}_index.npy') for s in ('train', 'val', 'test')}
    rna = np.load(cache / 'rna_state.npz')
    field = np.load(cache / 'field_state.npz')
    np.testing.assert_array_equal(rna['fit_cell_index'], split['train'])
    np.testing.assert_array_equal(field['fit_cell_index'], split['train'])
    genes = np.load(cache / 'gene_ids.npy')
    if len(genes) != 2000 or len(set(genes)) != 2000 or manifest['test_atac_loaded']:
        raise ValueError('Invalid feature space/isolation')
    matrix = sparse.load_npz(cache / 'B_S.npz')
    finite(matrix, 'B_S')
    sums = np.asarray(abs(matrix).sum(0)).ravel()
    np.testing.assert_allclose(sums[sums > 0], 1, atol=1e-5)
    for s in ('train', 'val'):
        a, b = np.load(cache / f'rna_{s}.npy', mmap_mode='r'), np.load(cache / f'F_{s}.npy', mmap_mode='r')
        if a.shape != b.shape or a.shape != (len(split[s]), 2000):
            raise ValueError('Cache row/feature dimensions differ')
        finite(a, 'RNA'); finite(b, 'F')
    a = np.load(cache / 'rna_train.npy', mmap_mode='r')
    np.testing.assert_allclose(rna['mean'], a.mean(0), atol=1e-6)
    np.testing.assert_allclose(rna['std'], a.std(0), atol=1e-6)


def evaluate(config, runner, unit, source_key, target_key, student, cache, mean, std, seed, device):
    import torch
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    from main.core.regulatory.s_mode.single_source import map_rna, finite
    destination = unit / 'evaluation/AD_to_AD'
    if (destination / 'manifest.json').exists():
        return runner.legacy.checked(destination/'manifest.json')
    evaluation_started = time.perf_counter()
    benchmark = Path(config['benchmark'])
    raw = benchmark / 'GSE214979_filtered_feature_bc_matrix.h5'
    index = np.load(cache/'test_index.npy')
    with h5py.File(raw, 'r') as handle:
        ids = handle['matrix/features/id'][:].astype(str)[np.load(benchmark/'rna_feature_index.npy')]
    counts = read_counts(raw, index, np.load(benchmark/'rna_feature_index.npy'))
    x, hit = map_rna(counts, ids, np.load(cache/'gene_ids.npy'), mean, std)
    if not hit.all():
        raise ValueError('Intra-dataset selected genes missing')
    destination.mkdir(parents=True, exist_ok=True)
    torch.cuda.synchronize(); started = time.perf_counter()
    z = runner.b.forward_numpy(student, x, device)
    torch.cuda.synchronize(); seconds = time.perf_counter()-started
    finite(z, 'Student embedding')
    np.save(destination/'embedding.npy', z)
    digest = sha(destination/'embedding.npy')
    frozen = runner.b.now()
    # Only post-hoc evaluation reads cell-type annotations or test disease labels.
    with (benchmark/'cells.csv').open() as handle:
        cells = list(csv.DictReader(handle))
    bars = (cache/'barcodes.txt').read_text().splitlines()
    if bars != [r['barcode'] for r in cells]:
        raise ValueError('Label/barcode misalignment')
    scaled = runner.b.StandardScaler().fit_transform(z)
    neighbors = NearestNeighbors(n_neighbors=16).fit(scaled).kneighbors(scaled, return_distance=False)
    silhouette_index = np.sort(np.random.default_rng(SEEDS[0]).choice(len(z), min(5000, len(z)), replace=False))
    np.save(destination/'silhouette_index.npy', silhouette_index)
    metrics = {}
    for name, column in [('disease', 'disease_status'), ('author_cell_type', 'author_predicted_id')]:
        labels = np.asarray([cells[int(i)][column] for i in index])
        names, y = np.unique(labels, return_inverse=True)
        if name == 'disease' and names.tolist() != ['AD', 'Ctrl']:
            raise ValueError('Held-out disease classes differ')
        values = runner.b.evaluate(z, y)
        prediction = KMeans(n_clusters=len(names), **runner.b.EVALUATOR).fit_predict(scaled)
        np.save(destination/f'{name}_clusters.npy', prediction)
        np.save(destination/f'{name}_labels.npy', y)
        values['label_names'] = names.tolist()
        values['knn_purity_at_15'] = float(np.mean([np.mean(y[row[row != i][:15]] == y[i])
                                                   for i, row in enumerate(neighbors)]))
        sample_clusters = prediction[silhouette_index]
        values['silhouette'] = (float(silhouette_score(scaled[silhouette_index], sample_clusters))
            if 1 < len(np.unique(sample_clusters)) < len(sample_clusters) else None)
        values['silhouette_scope'] = 'predicted KMeans clusters; fixed <=5000 test-cell subsample'
        metrics[name] = values
    if sha(destination/'embedding.npy') != digest:
        raise ValueError('Frozen output changed during evaluation')
    np.save(destination/'test_index.npy', index)
    (destination/'cell_order.tsv').write_text('barcode\n'+'\n'.join(np.asarray(bars)[index])+'\n')
    training = json.loads((unit/'training_manifest.json').read_text())
    resources = [training['stages'][s]['resources'] for s in ('teacher', 'student')]
    preprocessing = json.loads((cache/'manifest.json').read_text())['preprocessing_seconds']
    efficiency = dict(Method='Full B2 S-mode', Total_time=preprocessing+training['wall_seconds']+time.perf_counter()-evaluation_started,
        Trainable_params=resources[1]['parameters'], Teacher_params=resources[0]['parameters'],
        Peak_VRAM=max(torch.cuda.max_memory_allocated(), max(r['peak_allocated_bytes'] for r in resources)),
        peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in resources),
        Inference_time=seconds, Cells_sec=len(index)/seconds, GPU=resources[1]['gpu'],
        total_time_scope='shared preprocessing charged once per standalone seed + training + posthoc evaluation; queue wait excluded',
        evaluation_posthoc_seconds=time.perf_counter()-started-seconds)
    result = dict(status='COMPLETE', seed=seed, run_tier='pre_final', metrics=metrics, efficiency=efficiency,
        embedding_sha256=digest, embedding_frozen_at=frozen, label_read_after_embedding_hash=True,
        cache_manifest_sha256=sha(cache/'manifest.json'), test_atac_used=False, target_fit=False,
        adaptation=False, protocol_sha256=sha(config['protocol_path']),
        annotation_scope='cell-weighted held-out donor clustering; NOT clinical classification',
        hashes={str(p): sha(p) for p in destination.iterdir() if p.is_file()})
    write(destination/'manifest.json', result)
    return result


def summarize(out, cache_manifest):
    columns = ['Seed', 'ARI', 'NMI', 'ACC', 'Macro-F1', 'Balanced ACC', 'Purity@15', 'Silhouette',
               'Method', 'Total time', 'Trainable params', 'Peak VRAM', 'Inference time', 'Cells/sec', 'GPU',
               'AMI', 'Accuracy', 'Weighted-F1']
    out.joinpath('metrics').mkdir(exist_ok=True)
    all_rows = {}
    for scope in ('disease', 'author_cell_type'):
        rows = []
        for seed in SEEDS:
            path = out/'models/AD'/f'seed_{seed}'/'evaluation/AD_to_AD/manifest.json'
            if not path.exists():
                continue
            value = json.loads(path.read_text()); verify_hashes(value['hashes'])
            m, e = value['metrics'][scope], value['efficiency']
            rows.append(dict(zip(columns, [seed, m['ari'], m['nmi'], m['cluster_acc'], m['macro_f1'],
                m['balanced_accuracy'], m['knn_purity_at_15'], m['silhouette'], e['Method'],
                e['Total_time'], e['Trainable_params'], e['Peak_VRAM']/(1 << 30), e['Inference_time'], e['Cells_sec'], e['GPU'],
                m['ami'], m['accuracy'], m['weighted_f1']])))
        with (out/'metrics'/f'{scope}_per_seed.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, columns); writer.writeheader(); writer.writerows(rows)
        summary = {c: dict(mean=float(np.mean([r[c] for r in rows])),
                          std=float(np.std([r[c] for r in rows], ddof=1)) if len(rows)>1 else None)
                   for c in columns[1:] if rows and all(isinstance(r[c], (float, int)) for r in rows)}
        all_rows[scope] = dict(completed_seeds=[r['Seed'] for r in rows], rows=rows, summary=summary)
        lines = ['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
        lines += ['| '+' | '.join(str(row[c]) for c in columns)+' |' for row in rows]
        lines.append('\nFive-seed pre_final; cell-level clustering. Time: seconds; peak VRAM: GiB. No supervised disease classifier.\n')
        for c, v in summary.items():
            lines.append(f"{c}: {v['mean']:.6g} ± {v['std']}\n")
        (out/'metrics'/f'{scope}_summary.md').write_text('\n'.join(lines))
    graph = {k: cache_manifest[k] for k in ('candidate_edges', 'effective_edges', 'supported_genes',
             'selector_seconds', 'field_seconds', 'preprocessing_seconds')}
    graph.update(dataset='GSE214979', candidate_genes=2000,
                 candidate_peaks=cache_manifest['field']['peaks'],
                 nondegenerate_F_genes=cache_manifest['field']['nondegenerate_genes'])
    with (out/'metrics/graph_stats.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, list(graph)); writer.writeheader(); writer.writerow(graph)
    (out/'metrics/graph_stats.md').write_text('| '+' | '.join(graph)+' |\n| '+
        ' | '.join(['---']*len(graph))+' |\n| '+' | '.join(map(str, graph.values()))+' |\n')
    write(out/'metrics/summary.json', dict(run_tier='pre_final', results=all_rows,
        shared_preprocessing_seconds=cache_manifest['preprocessing_seconds'], regulatory=cache_manifest))
    return all(len(v['completed_seeds']) == 5 for v in all_rows.values())


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(); config = json.loads(args.config.read_text())
    verify_hashes(config['code_hashes'])
    if config['seeds'] != SEEDS:
        raise ValueError('Seed protocol differs')
    runner = runtime(args.config.parent)
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(2)
    with threadpool_limits(2):
        cache, manifest = prepare(config, runner)
        callback = lambda *values: evaluate(config, runner, *values)
        runner.train('AD', SEEDS[0], Path(config['result']), cache, manifest, smoke=True,
                     evaluation_targets=('AD',), evaluate_fn=callback)
        write(Path(config['result'])/'verification.json', dict(cache_check='PASS', smoke='PASS',
            source_only=True, protocol_sha256=sha(config['protocol_path']), research_diagnostics='WARNING_ONLY'))
        try:
            for seed in SEEDS:
                runner.train('AD', seed, Path(config['result']), cache, manifest,
                             evaluation_targets=('AD',), evaluate_fn=callback)
                summarize(Path(config['result']), manifest)
        finally:
            summarize(Path(config['result']), manifest)
        if not summarize(Path(config['result']), manifest):
            raise RuntimeError('Incomplete seed metrics')


if __name__ == '__main__':
    main()
