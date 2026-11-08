"""Re-score frozen Task45 outputs and time inference using processed RNA only."""
import argparse
import csv
import fcntl
import importlib.util
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path('/home/u202522802045/share/zpb')
sys.path[:0] = [str(ROOT), str(ROOT/'main/run/d_mode'), str(ROOT/'main/core/regulatory/common/legacy_support')]
from main.core.regulatory.s_mode.single_source import map_rna
import prepare_updated_b2_sources as storage

REFERENCE = ROOT/'main/core/run_b2_strict_lodo.py'
spec = importlib.util.spec_from_file_location('frozen_b2_audit', REFERENCE)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
b.sha = storage.sha
OLD = ROOT/'main/result/s_mode/intra/45_smode_full_b2_corrected_pbmc_20261012'
CACHE = ROOT/'data/cache/s_mode/PBMC'
COMPARISON = CACHE
BENCH = ROOT/'data/benchmark/PBMC'
PROTOCOL = ROOT/'comparison/docs/PBMC_SMODE_SCORING_TIMING_V1.json'
KEYS = ['D1', 'D2', 'D3', 'D4']
SEEDS = list(range(20020130, 20020135))
SCORES = {'ARI': 'ari', 'NMI': 'nmi', 'ACC': 'cluster_acc', 'Macro-F1': 'macro_f1',
          'Balanced ACC': 'balanced_accuracy', 'Purity@15': 'knn_purity_at_15', 'Silhouette': 'silhouette'}


def cache_for(root, key, version):
    dataset = f'PBMC-MO-00{key[-1]}'
    return root/dataset/version


def benchmark_for(key):
    dataset = f'PBMC-MO-00{key[-1]}'
    return BENCH/dataset/'legacy_corrected_refs_v1/legacy_refs'


def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    b.write(path.with_suffix('.json'), rows)
    columns = list(rows[0])
    text = '| ' + ' | '.join(columns) + ' |\n|' + '---|'*len(columns) + '\n'
    for row in rows:
        values = ['NA' if row[c] is None else f'{row[c]:.6f}' if isinstance(row[c], float) else str(row[c]) for c in columns]
        text += '| ' + ' | '.join(values) + ' |\n'
    path.with_suffix('.md').write_text(text)


def summarize(out, protocol):
    records = [b.read(p) for p in sorted((out/'evaluations').glob('D*/seed_*/*/manifest.json'))]
    rows = [dict(Family=r['family'], Source=r['source'], Target=r['target'], Seed=r['seed'],
                 **r['display']) for r in records]
    if rows:
        table(out/'metrics/per_seed', rows)
    summaries, deviations = [], []
    for family, count in [('intra', 4), ('inter', 12)]:
        selected = [r for r in records if r['family']==family]
        mean = dict.fromkeys(protocol['display_columns'])
        sd = dict.fromkeys(protocol['display_columns'])
        gpu = ', '.join(sorted({r['display']['GPU'] for r in selected})) or None
        mean.update(Method='Full B2-S', GPU=gpu)
        sd.update(Method='Full B2-S', GPU=gpu)
        if len(selected)==count*len(SEEDS):
            for column in protocol['display_columns'][1:-1]:
                seed_means = []
                for seed in SEEDS:
                    values = [r['display'][column] for r in selected if r['seed']==seed]
                    assert len(values)==count
                    if all(v is not None for v in values):
                        seed_means.append(float(np.mean(values)))
                if len(seed_means)==len(SEEDS):
                    mean[column] = float(np.mean(seed_means))
                    sd[column] = float(np.std(seed_means, ddof=1))
        table(out/f'metrics/{family}_mean', [mean])
        table(out/f'metrics/{family}_sample_sd', [sd])
        summaries.append(dict(Family=family, Evaluations=len(selected), Expected=count*len(SEEDS),
                              Timed=sum(r['inference_seconds'] is not None for r in selected), **mean))
        deviations.append(dict(Family=family, **sd))
    table(out/'metrics/summary_mean', summaries)
    table(out/'metrics/summary_sample_sd', deviations)
    return records


def processed_inputs(source):
    """No raw H5 access and no fallback to legacy D1 counts."""
    cache = cache_for(CACHE, source, 'pbmc_corrected_s10x20_v1_20261012')
    state = np.load(cache/'rna_state.npz')
    genes = np.load(cache/'gene_ids.npy')
    values, evidence = {}, {}
    for target in KEYS:
        if target=='D1':
            if source!='D1':
                values[target] = None
                evidence[target] = dict(reason='Processed D1 RNA lacks the full gene panel required by other Sources; raw-data fallback disabled.')
                continue
            folder = cache_for(COMPARISON, target, 'pbmc_comparison_intra_v1')
            manifest = b.read(folder/'manifest.json')
            files = [folder/n for n in ['rna_log_test.npz', 'gene_ids.npy', 'test_index.npy', 'barcodes.txt']]
            for path in files:
                assert b.sha(path)==manifest['cache_sha256'][path.name]
            np.testing.assert_array_equal(genes, np.load(folder/'gene_ids.npy'))
            np.testing.assert_array_equal(np.load(cache/'test_index.npy'), np.load(folder/'test_index.npy'))
            log = sparse.load_npz(folder/'rna_log_test.npz').toarray()
            x = ((log-state['mean'])/(state['std']+1e-6)).astype(np.float32)
        else:
            folder = benchmark_for(target)
            processed = (folder/'barcodes.txt').resolve().parent
            source_cache = cache_for(CACHE, target, 'pbmc_corrected_s10x20_v1_20261012')
            index = np.load(source_cache/'test_index.npy')
            files = [processed/'rna_counts.npz' if target=='D2' else processed/'rna.npz',
                     processed/'barcodes.txt', source_cache/'test_index.npy']
            if target=='D2':
                counts = sparse.load_npz(files[0])
                files.append(source_cache/'all_gene_ids.npy')
                ids = np.load(files[-1])
            else:
                with np.load(files[0]) as archive:
                    counts = sparse.csr_matrix((archive['data'], archive['indices'], archive['indptr']), shape=tuple(archive['shape']))
                    ids = archive['gene_ids'].copy()
            assert counts.shape[0]==len((source_cache/'barcodes.txt').read_text().splitlines())
            assert files[1].read_text()==(source_cache/'barcodes.txt').read_text()
            x, hit = map_rna(counts[index], ids, genes, state['mean'], state['std'])
            old = b.read(OLD/'models'/source/'seed_20020130/evaluation'/f'{source}_to_{target}'/'manifest.json')
            old_mask = OLD/'models'/source/'seed_20020130/evaluation'/f'{source}_to_{target}'/'gene_hit_mask.npy'
            np.testing.assert_array_equal(hit, np.load(old_mask))
        assert np.isfinite(x).all()
        values[target] = x
        evidence[target] = dict(files={str(f): b.sha(f) for f in files}, shape=list(x.shape))
    evidence['source_state'] = {str(cache/n): b.sha(cache/n) for n in ['rna_state.npz', 'gene_ids.npy', 'manifest.json']}
    return values, evidence


@torch.no_grad()
def infer(model, x):
    chunks = [model(torch.from_numpy(x[i:i+256]).to('cuda')).cpu().numpy() for i in range(0, len(x), 256)]
    return np.concatenate(chunks)


def run(out, smoke):
    protocol = b.read(PROTOCOL)
    original = b.read(OLD/'manifest.json')
    assert b.sha(REFERENCE)=='3de1ad3642e7af276658d9b239ccaf9457adf7f8e6078b95a237cf4605f2c7e1'
    assert b.EVALUATOR==protocol['scoring']['kmeans']
    assert original['status']=='COMPLETE' and original['seeds']==SEEDS
    for item in protocol['labels']:
        assert b.sha(item['path'])==item['sha256']
    source_hashes = {str(p): b.sha(p) for p in [REFERENCE, Path(__file__), PROTOCOL,
                     ROOT/'main/core/regulatory/s_mode/single_source.py']}
    progress = dict(status='RUNNING', protocol_id=protocol['protocol_id'], run_tier='smoke' if smoke else 'pre_final',
                    source_task=45, source_hashes=source_hashes, seeds=SEEDS[:1] if smoke else SEEDS,
                    expected_evaluations=4 if smoke else 80,
                    retraining=False, total_time_seconds=None, total_time_reason='Checkpoint reuse; complete historical workflow not timed.',
                    scoring_scope='frozen historical embeddings; harmonized labels; no target fitting',
                    expected_timed_evaluations=4 if smoke else 65, expected_timing_gaps=0 if smoke else 15)
    b.write(out/'manifest.json', progress)
    for source in KEYS[:1] if smoke else KEYS:
        inputs, input_evidence = processed_inputs(source)
        b.write(out/'audits'/f'{source}_processed_inputs.json', input_evidence)
        for seed in SEEDS[:1] if smoke else SEEDS:
            unit = OLD/'models'/source/f'seed_{seed}'
            trained = b.read(unit/'training_manifest.json')
            checkpoint = unit/'checkpoints/student/best.pt'
            teacher = unit/'checkpoints/teacher/best.pt'
            for path in [checkpoint, teacher]:
                assert b.sha(path)==storage.expected_hash(trained['hashes'],path)
            copy = out/'checkpoints'/source/f'seed_{seed}'
            copy.mkdir(parents=True, exist_ok=True)
            for name, path in [('student_best.pt', checkpoint), ('teacher_best.pt', teacher), ('training_manifest.json', unit/'training_manifest.json')]:
                dest = copy/name
                if not dest.exists():
                    shutil.copy2(path, dest)
                assert b.sha(dest)==b.sha(path)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = b.Encoder(2000)
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
            model.eval().requires_grad_(False).cuda()
            params = sum(r['resources']['parameters'] for r in trained['stages'].values())
            for target in KEYS:
                dest = out/'evaluations'/source/f'seed_{seed}'/target
                old_dir = unit/'evaluation'/f'{source}_to_{target}'
                old_manifest = b.read(old_dir/'manifest.json')
                assert old_manifest['status']=='COMPLETE'
                assert old_manifest['student_checkpoint_sha256']==b.sha(checkpoint)
                old_z = old_dir/'embedding.npy'
                assert b.sha(old_z)==storage.expected_hash(old_manifest['hashes'],old_z)
                if (dest/'manifest.json').exists():
                    done = b.read(dest/'manifest.json')
                    assert done['status']=='COMPLETE'
                    assert done['protocol_sha256']==b.sha(PROTOCOL) and done['checkpoint_sha256']==b.sha(checkpoint)
                    assert b.sha(dest/'embedding.npy')==done['prediction_sha256']
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                started = time.perf_counter()
                z = np.load(old_z)
                seconds, inference_peak, inference_reserved, difference = None, None, None, None
                x = inputs[target]
                if x is not None:
                    infer(model, x[:256])
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    tick = time.perf_counter()
                    predicted = infer(model, x)
                    torch.cuda.synchronize()
                    seconds = time.perf_counter()-tick
                    inference_peak = torch.cuda.max_memory_allocated()
                    inference_reserved = torch.cuda.max_memory_reserved()
                    np.testing.assert_allclose(predicted, z, rtol=1e-4, atol=1e-5)
                    difference = float(np.max(np.abs(predicted-z)))
                    np.save(dest/'retimed_embedding.npy', predicted)
                shutil.copy2(old_z, dest/'embedding.npy')
                frozen_hash = b.sha(dest/'embedding.npy')
                index = np.load(old_dir/'test_index.npy')
                np.testing.assert_array_equal(index, np.load(CACHE/target/'test_index.npy'))
                label_path = benchmark_for(target)/'label_harmonized.csv'
                with label_path.open() as handle:
                    labels = list(csv.DictReader(handle))
                cells = (cache_for(CACHE, target, 'pbmc_corrected_s10x20_v1_20261012')/'barcodes.txt').read_text().splitlines()
                assert [row['barcode'] for row in labels]==cells
                y = np.unique(np.asarray([r['cell_type'] for r in labels])[index], return_inverse=True)[1]
                assert len(y)==len(z) and np.isfinite(z).all()
                tick = time.perf_counter()
                metrics = b.evaluate(z, y)
                scaled = StandardScaler().fit_transform(z)
                neighbors = NearestNeighbors(n_neighbors=16, metric='euclidean').fit(scaled).kneighbors(scaled, return_distance=False)
                neighbors = np.asarray([row[row!=i][:15] for i, row in enumerate(neighbors)])
                metrics['knn_purity_at_15'] = float(np.mean(y[neighbors]==y[:,None]))
                metrics['silhouette'] = float(silhouette_score(scaled, y, metric='euclidean'))
                score_seconds = time.perf_counter()-tick
                for key in ['ari', 'nmi', 'cluster_acc', 'macro_f1', 'balanced_accuracy', 'knn_purity_at_15']:
                    assert abs(metrics[key]-old_manifest['metrics'][key])<1e-6, (source, target, seed, key)
                resources = [s['resources'] for s in trained['stages'].values()]
                training_peak = max(s['peak_allocated_bytes'] for s in resources)
                combined_peak = max(training_peak, inference_peak) if inference_peak is not None else None
                display = dict(Method='Full B2-S', **{c: metrics[k] for c, k in SCORES.items()},
                               **{'Total time (s)': None, 'Trainable params': params,
                                  'Peak VRAM (GiB)': combined_peak/(1024**3) if combined_peak is not None else None,
                                  'Inference time (s)': seconds, 'Cells/sec': len(z)/seconds if seconds is not None else None,
                                  'GPU': torch.cuda.get_device_name()})
                record = dict(status='COMPLETE', protocol_id=protocol['protocol_id'], protocol_sha256=b.sha(PROTOCOL),
                              source=source, target=target, seed=seed, family='intra' if source==target else 'inter',
                              display=display, metrics=metrics, checkpoint_path=str(copy/'student_best.pt'),
                              checkpoint_sha256=b.sha(checkpoint), label_sha256=b.sha(label_path),
                              prediction_sha256=frozen_hash, historical_prediction=str(old_z),
                              historical_prediction_sha256=b.sha(old_z), inference_seconds=seconds,
                              inference_peak_allocated_bytes=inference_peak, inference_peak_reserved_bytes=inference_reserved,
                              source_training_peak_allocated_bytes=training_peak, peak_allocated_bytes=combined_peak,
                              source_training_gpu=trained['environment']['gpu'], scoring_seconds=score_seconds,
                              evaluation_wall_seconds=time.perf_counter()-started, batch_size=256, warmup_batches=1,
                              max_abs_prediction_difference=difference, test_cells=len(z), label_count=len(np.unique(y)),
                              input_evidence=input_evidence.get(target), input_hashes=input_evidence['source_state'],
                              total_time_seconds=None, historical_training_seconds=sum(s['training_seconds'] for s in resources),
                              inference_parameters=sum(p.numel() for p in model.parameters()),
                              target_atac_used=False, target_fit=False, labels_used_for_fitting=False,
                              geometry_pearson=None, geometry_spearman=None,
                              input_timing_missing_reason=input_evidence[target].get('reason'),
                              environment=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda))
                np.save(dest/'test_index.npy', index)
                b.write(dest/'metrics.json', metrics)
                assert b.sha(dest/'embedding.npy')==frozen_hash
                assert b.sha(checkpoint)==storage.expected_hash(trained['hashes'],checkpoint)
                record['evaluation_wall_seconds']=time.perf_counter()-started
                b.write(dest/'manifest.json', record)
                print('COMPLETE', source, seed, target, 'timed', seconds is not None, flush=True)
            del model
            torch.cuda.empty_cache()
            summarize(out, protocol)
    records = summarize(out, protocol)
    assert len(records)==progress['expected_evaluations']
    assert sum(r['inference_seconds'] is not None for r in records)==progress['expected_timed_evaluations']
    assert all(b.sha(p)==digest for p, digest in source_hashes.items())
    progress.update(status='SMOKE_PASS' if smoke else 'SCORED_COMPLETE_TIMING_PARTIAL',
                    completed_evaluations=len(records), timed_evaluations=sum(r['inference_seconds'] is not None for r in records))
    b.write(out/'manifest.json', progress)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    with (ROOT/'.benchmark_a30.lock').open('a') as lock:
        print('WAIT_GPU_LOCK', flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        torch.set_num_threads(2)
        with threadpool_limits(2):
            run(args.output, args.smoke)
