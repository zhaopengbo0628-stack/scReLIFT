"""Thin wrapper over the frozen Task35 model and training stages."""
import argparse
import copy
import csv
import fcntl
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, silhouette_score
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import prepare_updated_b2_sources as prep

REFERENCE = prep.ROOT / 'main/result/d_mode/35_b2_shared_regulatory_teacher_dense_kd_20261003/source/run_b2_strict_lodo.py'
spec = importlib.util.spec_from_file_location('frozen_b2', REFERENCE)
b2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b2)
OUT = prep.OUT
SEEDS = (20020130, 20020131, 20020132)


def load_sources(fold):
    folder = OUT / 'validated_source_artifacts' / fold
    manifest = prep.checked(folder / 'manifest.json')
    sources = []
    for key in manifest['sources']:
        directory = folder / 'harmonized' / prep.NAMES[key]
        with np.load(directory / 'metadata.npz') as meta:
            train, val = meta['train_index'], meta['val_index']
        x = np.load(directory / 'rna.npy')
        assert np.isfinite(x).all() and not np.intersect1d(train, val).size
        sources.append(dict(name=prep.NAMES[key], key=key, raw=x, train_index=train, val_index=val))
    blocks = [s['raw'][s['train_index']] for s in sources]
    means = np.stack([x.mean(0) for x in blocks])
    mean = means.mean(0).astype(np.float32)
    std = np.sqrt(np.mean(np.stack([x.var(0) for x in blocks])+(means-mean)**2, axis=0)).astype(np.float32)
    for s in sources:
        for split in ['train', 'val']:
            x = s['raw'][s[f'{split}_index']]
            f = np.load(folder / f"{s['name']}_F_{split}.npy")
            assert f.shape == x.shape and np.isfinite(f).all()
            s[split] = dict(rna=((x-mean)/(std+1e-6)).astype(np.float32), raw_nonzero=x!=0, field=f)
    return sources, mean, std, manifest


def target_cache(fold):
    target = fold.split('_')[1]
    directory = OUT / 'validated_cache' / fold
    if (directory / 'manifest.json').exists():
        return prep.checked(directory / 'manifest.json')
    import unblock_updated_full_b2 as raw_reader
    genes = np.load(OUT / 'validated_source_artifacts' / fold / 'gene_ids.npy')
    if target == 'D2':
        index = np.load(prep.DATA[target] / 'processed/test_index.npy')
    else:
        with np.load(prep.PREPARED[target] / 'metadata.npz', allow_pickle=True) as meta:
            index = meta['test_index']
    counts, ids, raw_path, provenance = raw_reader.raw_rna(target)
    lookup = {str(g).split('.')[0]: i for i, g in enumerate(ids)}
    wanted = np.asarray([lookup.get(str(g), -1) for g in genes])
    assert (wanted >= 0).all()
    totals = np.asarray(counts[index].sum(axis=1)).ravel().astype(np.float32)
    values = counts[index][:, wanted].toarray().astype(np.float32)
    x = np.log1p(values*(1e4/np.maximum(totals, 1.0))[:, None]).astype(np.float32)
    assert np.isfinite(x).all()
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / 'target_rna_test.npy', x)
    np.save(directory / 'target_test_index.npy', index)
    cells = raw_reader.barcodes(target)
    (directory / 'cell_order.tsv').write_text('cell_id\n'+'\n'.join(cells[i] for i in index)+'\n')
    dependency_files = [raw_path, prep.DATA[target] / 'processed/barcodes.txt',
                        OUT / 'validated_source_artifacts' / fold / 'gene_ids.npy', Path(raw_reader.__file__)]
    value = dict(target=target, rows=len(index), genes=len(genes), missing_genes=0,
                 raw_source=str(raw_path), provenance=provenance,
                 TARGET_ATAC_LOADED=False, TARGET_LABEL_LOADED=False, TARGET_HVG_FITTED=False,
                 TARGET_PCA_FITTED=False, target_population_statistics_fitted=False,
                 hashes={str(p): prep.sha(p) for p in list(directory.iterdir()) + dependency_files if p.is_file()})
    prep.write(directory / 'manifest.json', value)
    return value


def preflight():
    audit = json.loads((prep.ROOT / 'data/protocols/PBMC/corrected_legacy_evidence/UPDATED_FOUR_DATASET_INDEPENDENCE_AUDIT.json').read_text())
    assert audit['FOUR_DATASET_INDEPENDENCE'] == 'PASS'
    for record in audit['datasets'].values():
        assert prep.sha(record['rna_path']) == record['rna_sha256']
    rows = []
    for fold in [f'target_D{i}' for i in range(1, 5)]:
        sources, mean, std, sm = load_sources(fold)
        target = target_cache(fold)
        b2.seed_all(20020130)
        model = b2.Teacher(len(mean), len(mean))
        x = torch.from_numpy(sources[0]['train']['rna'][:8])
        f = torch.from_numpy(sources[0]['train']['field'][:8])
        losses = b2.teacher_losses(model, x, f)
        total = sum(losses)
        total.backward()
        assert torch.isfinite(total) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        student = copy.deepcopy(model.rna_encoder).eval()
        assert torch.equal(student(x), model.rna_encoder.eval()(x))
        kd = torch.nn.functional.mse_loss(student(x), model(x, f)[2].detach())
        kd.backward()
        assert torch.isfinite(kd)
        stats = OUT / 'validated_cache' / fold / 'source_rna_stats.npz'
        np.savez_compressed(stats, mean=mean, std=std)
        rows.append(dict(fold=fold, genes=len(mean), sources=sm['sources'], target=target,
                         teacher_params=sum(p.numel() for p in model.parameters()),
                         student_params=sum(p.numel() for p in student.parameters()), stats_sha256=prep.sha(stats)))
    gate = dict(status='PASS', FOUR_DATASET_INDEPENDENCE='PASS', NEW_D2_PROVENANCE='PASS',
                SOURCE_ARTIFACT_REBUILD='PASS', TARGET_RNA_ONLY_DRYRUN='PASS', FULL_B2_METHOD_DRIFT='NO',
                UPDATED_RUNNER_READY='YES', FORMAL_RUN_STATUS='NOT_STARTED', run_tier='pre_final_explicit_3seeds',
                frozen_reference_sha256=prep.sha(REFERENCE), wrapper_sha256=prep.sha(__file__),
                unchanged_functions=['Encoder', 'Teacher', 'batches', 'teacher_losses', 'train_stage', 'rng_state', 'restore_rng'],
                cpu_forward_backward_only=True, optimizer_step=False, rows=rows,
                registry_modified=False, historical_results_modified=False)
    prep.write(OUT / '_benchmark_meta/VALIDATED_STEP1_GATE.json', gate)
    print('STEP1_PASS', flush=True)


def cell_order(sources, split, path):
    lines = ['dataset\tcell_id']
    for s in sources:
        cells = (prep.DATA[s['key']] / 'processed/barcodes.txt').read_text().splitlines()
        lines += [f"{s['key']}\t{cells[i]}" for i in s[f'{split}_index']]
    path.write_text('\n'.join(lines)+'\n')


def f_retention_gate(model, sources, device):
    rows = []
    for s in sources:
        train = b2.forward_numpy(model, s['train']['rna'], device, s['train']['field'])[2]
        val = b2.forward_numpy(model, s['val']['rna'], device, s['val']['field'])[2]
        x_scaler = StandardScaler().fit(train)
        pred = Ridge(alpha=1.0).fit(x_scaler.transform(train), s['train']['field']).predict(x_scaler.transform(val))
        truth = s['val']['field'].astype(np.float64)
        pred = pred.astype(np.float64)
        variance = truth.var(axis=0)
        mse = np.mean((truth - pred) ** 2, axis=0)
        active = variance > 0
        r2 = np.zeros_like(variance)
        r2[active] = 1.0 - mse[active] / variance[active]
        total_variance = variance.sum()
        weighted = float(np.sum(variance * r2) / total_variance) if total_variance > 0 else float('-inf')
        pooled = float(1.0 - np.sum(variance * (1.0 - r2)) / total_variance) if total_variance > 0 else float('-inf')
        negative = r2 < 0
        rows.append(dict(
            source=s['name'], validation_cells=len(truth), output_directions=len(variance),
            exact_zero_variance_directions=int((~active).sum()),
            UNIFORM_MEAN_R2=float(r2.mean()), MEDIAN_R2=float(np.median(r2)),
            FRACTION_R2_POSITIVE=float(np.mean(r2 > 0)), VARIANCE_WEIGHTED_R2=weighted,
            POOLED_R2=pooled, NEGATIVE_R2_DIRECTION_COUNT=int(negative.sum()),
            NEGATIVE_DIRECTION_VARIANCE_FRACTION=float(variance[negative].sum() / total_variance) if total_variance > 0 else 0.0,
            passed=bool(weighted > 0),
        ))
    return rows


def save_gate(unit, rows, checkpoint_sha, seed, fold, inputs):
    evaluation = unit / 'evaluation'
    evaluation.mkdir(parents=True, exist_ok=True)
    columns = ['source', 'validation_cells', 'output_directions', 'exact_zero_variance_directions',
               'UNIFORM_MEAN_R2', 'MEDIAN_R2', 'FRACTION_R2_POSITIVE', 'VARIANCE_WEIGHTED_R2',
               'POOLED_R2', 'NEGATIVE_R2_DIRECTION_COUNT', 'NEGATIVE_DIRECTION_VARIANCE_FRACTION', 'passed']
    with (evaluation / 'teacher_source_gate_matrix.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    decision = dict(
        fold=fold, seed=seed, status='RECORDED',
        TEACHER_F_RETENTION_VALID='DIAGNOSTIC_ONLY',
        warning=any(not r['passed'] for r in rows),
        gate='VARIANCE_WEIGHTED_R2 > 0', threshold=0.0, source_count=len(rows), source_rows=rows,
        source_validation_only=True, target_metrics_used=False, target_labels_used=False, target_atac_used=False,
        teacher_checkpoint_sha256=checkpoint_sha, source_input_hashes=inputs,
    )
    prep.write(evaluation / 'teacher_gate_decision.json', decision)
    return decision


def validate_reusable_teacher(unit, fold, seed, inputs):
    unit_config = json.loads((unit / 'config.json').read_text())
    assert unit_config['fold'] == fold and unit_config['seed'] == seed
    directory = unit / 'checkpoints/teacher'
    config_path, data_path = directory / 'config.json', directory / 'data_manifest.json'
    manifest_path = directory / 'manifest.json'
    config = json.loads(config_path.read_text())
    assert config == dict(**b2.CONFIG, stage='teacher', seed=seed)
    manifest = json.loads(manifest_path.read_text())
    assert manifest['status'] == 'TRAINING_COMPLETE'
    assert manifest['input_hashes'] == inputs
    assert manifest['config_hash'] == prep.sha(config_path)
    assert json.loads(data_path.read_text())['input_hashes'] == inputs
    for name, expected_hash in [('best.pt', manifest['best_checkpoint_sha256']),
                                ('latest.pt', manifest['latest_checkpoint_sha256'])]:
        path = directory / name
        assert prep.sha(path) == expected_hash, f'Fatal checkpoint hash mismatch: {path}'
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert state['config_hash'] == prep.sha(config_path) and state['input_hashes'] == inputs
        assert all(torch.isfinite(value).all() for value in state['model'].values()
                   if torch.is_tensor(value)), f'Fatal non-finite checkpoint tensor: {path}'


def fatal_latent_check(model, sources, device):
    for source in sources:
        for split in ['train', 'val']:
            values = b2.forward_numpy(model, source[split]['rna'], device, source[split]['field'])
            for name, latent in zip(['z_R', 'z_F', 'z_T'], values[:3]):
                assert latent.ndim == 2 and latent.shape == (len(source[split]['rna']), 64)
                assert np.isfinite(latent).all(), f'Fatal non-finite {name}: {source["name"]}/{split}'
                assert np.any(np.var(latent, axis=0) > 0), f'Fatal constant {name}: {source["name"]}/{split}'


def record_research_warnings(audit, gate_rows, fold, seed):
    threshold = b2.CONFIG['teacher_gate']['min_rank']
    for row, f_row in zip(audit['source_rows'], gate_rows):
        ranks = {key: float(row[key]['effective_rank']) for key in ['z_R', 'z_F', 'z_T']}
        row['WARNING_EFFECTIVE_RANK_LOW'] = any(value < threshold for value in ranks.values())
        row['effective_rank_threshold'] = threshold
        row['effective_rank_values'] = ranks
        row['F_retention_warning'] = not f_row['passed']
        row['research_diagnostic_warnings'] = [key for key, value in row['checks'].items()
                                               if key != 'noncollapsed' and not value]
        if row['WARNING_EFFECTIVE_RANK_LOW']:
            print(f'WARNING_EFFECTIVE_RANK_LOW {fold} {seed} {row["source"]} {ranks} threshold={threshold}', flush=True)
        if row['F_retention_warning']:
            print(f'WARNING_F_RETENTION {fold} {seed} {row["source"]} weighted_r2={f_row["VARIANCE_WEIGHTED_R2"]}', flush=True)
        for key in row['research_diagnostic_warnings']:
            print(f'WARNING_RESEARCH_DIAGNOSTIC {fold} {seed} {row["source"]} {key}', flush=True)
    audit['RESEARCH_GATES_MODE'] = 'WARNING'
    audit['FATAL_GATES_UNCHANGED'] = True
    audit['TEACHER_SHARED_ALIGNMENT_VALID'] = 'DIAGNOSTIC_ONLY'


def audit_existing():
    for fold in [f'target_D{i}' for i in range(1, 5)]:
        sources, _, _, sm = load_sources(fold)
        inputs = sm['hashes'].copy()
        inputs[str(REFERENCE)] = prep.sha(REFERENCE)
        for seed in SEEDS:
            unit = OUT / fold / f'seed_{seed}'
            checkpoint = unit / 'checkpoints/teacher/best.pt'
            if not checkpoint.exists():
                print('TEACHER_MISSING', fold, seed, flush=True)
                continue
            model = b2.Teacher(sources[0]['train']['rna'].shape[1], sources[0]['train']['field'].shape[1])
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if 'config_hash' in state:
                cfg = unit / 'checkpoints/teacher/config.json'
                manifest = unit / 'checkpoints/teacher/data_manifest.json'
                assert prep.sha(cfg) == state['config_hash']
                assert state['input_hashes'] == inputs
                assert json.loads(manifest.read_text())['input_hashes'] == inputs
            model.load_state_dict(state['model'])
            rows = f_retention_gate(model, sources, torch.device('cpu'))
            decision = save_gate(unit, rows, prep.sha(checkpoint), seed, fold, inputs)
            print('TEACHER_GATE', fold, seed, decision['status'], flush=True)


def run(fold, seed, unit):
    assert Path('/proc/self/fd/9').resolve() == prep.ROOT / '.benchmark_a30.lock'
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gate = json.loads((OUT / '_benchmark_meta/VALIDATED_STEP1_GATE.json').read_text())
    assert gate['status'] == 'PASS' and gate['frozen_reference_sha256'] == prep.sha(REFERENCE)
    assert gate['wrapper_sha256'] == prep.sha(__file__)
    sources, mean, std, sm = load_sources(fold)
    inputs = sm['hashes'].copy()
    inputs[str(REFERENCE)] = prep.sha(REFERENCE)
    if (unit / 'manifest.json').exists():
        record = json.loads((unit / 'manifest.json').read_text())
        if record['status'] == 'COMPLETE':
            for filename, digest in record['artifact_hashes'].items():
                assert prep.sha(unit / filename) == digest
            expected_config = dict(**b2.CONFIG, seed=seed, fold=fold, independent_teacher_per_seed=True)
            assert json.loads((unit / 'config.json').read_text()) == expected_config
            data_record = json.loads((unit / 'data_manifest.json').read_text())
            assert data_record['source_input_hashes'] == inputs
            checkpoint = unit / 'checkpoints/teacher_best.pt'
            model = b2.Teacher(len(mean), len(mean))
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
            decision = save_gate(unit, f_retention_gate(model, sources, torch.device('cpu')),
                                 prep.sha(checkpoint), seed, fold, inputs)
            assert decision['status'] == 'PASS', 'Scientific blocker: source F-retention gate failed'
            print('ALREADY_COMPLETE', fold, seed, flush=True)
            return
    device = torch.device('cuda')
    unit.mkdir(parents=True, exist_ok=True)
    for name in ['checkpoints', 'embeddings', 'preprocessing', 'evaluation', 'logs']:
        (unit / name).mkdir(exist_ok=True)
    prep.write(unit / 'config.json', dict(**b2.CONFIG, seed=seed, fold=fold, independent_teacher_per_seed=True))
    prep.write(unit / 'data_manifest.json', dict(source_manifest=sm, target='RNA_ONLY', source_input_hashes=inputs))
    np.savez_compressed(unit / 'preprocessing/source_rna_stats.npz', mean=mean, std=std)
    started = time.perf_counter()
    b2.seed_all(seed)
    teacher = b2.Teacher(len(mean), len(mean)).to(device)
    teacher_dir = unit / 'checkpoints/teacher'
    if (teacher_dir / 'manifest.json').exists():
        validate_reusable_teacher(unit, fold, seed, inputs)
    teacher = b2.train_stage(teacher, sources, teacher_dir, seed, device, True, inputs)
    audit = b2.teacher_audit(teacher, sources, device)
    fatal_latent_check(teacher, sources, device)
    gate_rows = f_retention_gate(teacher, sources, device)
    for row, gate_row in zip(audit['source_rows'], gate_rows):
        row['F_retention_gate'] = gate_row
        row['checks']['F_preserved'] = gate_row['passed']
        row['passed'] = all(row['checks'].values())
    audit['F_RETENTION_GATE'] = 'VARIANCE_WEIGHTED_R2 > 0'
    audit['TEACHER_F_RETENTION_VALID'] = 'DIAGNOSTIC_ONLY'
    record_research_warnings(audit, gate_rows, fold, seed)
    decision = save_gate(unit, gate_rows, prep.sha(teacher_dir / 'best.pt'), seed, fold, inputs)
    prep.write(unit / 'evaluation/source_teacher_audit.json', audit)
    teacher.eval().requires_grad_(False)
    for split in ['train', 'val']:
        for s in sources:
            zr, zf, zt, _, _ = b2.forward_numpy(teacher, s[split]['rna'], device, s[split]['field'])
            s[split]['teacher_latent'] = zt
            for name, values in [('z_R', zr), ('z_F', zf), ('z_T', zt)]:
                np.save(unit / 'embeddings' / f"teacher_{s['key']}_{split}_{name}.npy", values)
    mu = np.mean([s['train']['teacher_latent'].mean(0) for s in sources], axis=0)
    variance = np.mean([s['train']['teacher_latent'].var(0)+(s['train']['teacher_latent'].mean(0)-mu)**2 for s in sources], axis=0)
    sigma = np.sqrt(variance).astype(np.float32)
    for s in sources:
        s.update(latent_mean=mu, latent_std=sigma, raw_zero_standardized=-mean/(std+1e-6))
    np.savez_compressed(unit / 'preprocessing/kd_stats.npz', mean=mu, std=sigma)
    student_inputs = dict(inputs)
    student_inputs[str(teacher_dir / 'best.pt')] = prep.sha(teacher_dir / 'best.pt')
    student_inputs.update({str(p): prep.sha(p) for p in (unit / 'embeddings').glob('teacher*')})
    b2.seed_all(seed)
    student = copy.deepcopy(teacher.rna_encoder).requires_grad_(True).to(device)
    student_dir = unit / 'checkpoints/student'
    student = b2.train_stage(student, sources, student_dir, seed, device, False, student_inputs)
    for split in ['train', 'val']:
        z = np.concatenate([b2.forward_numpy(student, s[split]['rna'], device) for s in sources])
        np.save(unit / 'embeddings' / f'source_{split}_embedding_raw.npy', z)
        cell_order(sources, split, unit / 'embeddings' / f'source_{split}_cell_order.tsv')
    target_cache(fold)
    torch.cuda.synchronize()
    start = time.perf_counter()
    raw = np.load(OUT / 'validated_cache' / fold / 'target_rna_test.npy')
    target = b2.forward_numpy(student, ((raw-mean)/(std+1e-6)).astype(np.float32), device)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter()-start
    target_path = unit / 'embeddings/target_embedding_raw.npy'
    np.save(target_path, target)
    frozen_hash = prep.sha(target_path)
    prep.write(unit / 'embedding_manifest.json', dict(target_sha256=frozen_hash, frozen_before_labels=True))
    shutil.copy2(OUT / 'validated_cache' / fold / 'cell_order.tsv', unit / 'embeddings/target_cell_order.tsv')
    key = fold.split('_')[1]
    with (prep.DATA[key] / 'processed/label_harmonized.csv').open() as handle:
        labels = np.asarray([row['cell_type'] for row in csv.DictReader(handle)])
    index = np.load(OUT / 'validated_cache' / fold / 'target_test_index.npy')
    labels = np.unique(labels[index], return_inverse=True)[1]
    assert prep.sha(target_path) == frozen_hash
    scaled = StandardScaler().fit_transform(target)
    np.save(unit / 'embeddings/target_embedding_standardized.npy', scaled)
    metrics = b2.evaluate(target, labels)
    from sklearn.cluster import KMeans
    from scipy.optimize import linear_sum_assignment
    clusters = KMeans(n_clusters=len(np.unique(labels)), **b2.EVALUATOR).fit_predict(scaled)
    table = np.zeros((len(np.unique(labels)), len(np.unique(labels))), np.int64)
    np.add.at(table, (labels, clusters), 1)
    a, c = linear_sum_assignment(-table)
    mapping = dict(zip(c, a))
    mapped = np.asarray([mapping[i] for i in clusters])
    precision, recall, f1, support = precision_recall_fscore_support(labels, mapped, zero_division=0)
    metrics.update(macro_precision=float(precision.mean()), macro_recall=float(recall.mean()),
                   per_class=[dict(label=int(i), precision=float(p), recall=float(r), f1=float(f), support=int(n))
                              for i, (p, r, f, n) in enumerate(zip(precision, recall, f1, support))])
    neighbors = NearestNeighbors(n_neighbors=min(16, len(labels))).fit(scaled).kneighbors(return_distance=False)
    metrics['knn_purity_at_15'] = float((labels[neighbors[:, :15]] == labels[:, None]).mean())
    metrics['cell_type_silhouette'] = float(silhouette_score(scaled, labels))
    prep.write(unit / 'metrics.json', metrics)
    for stage, directory in [('teacher', teacher_dir), ('student', student_dir)]:
        for suffix, filename in [('best', 'best.pt'), ('final', 'latest.pt')]:
            shutil.copy2(directory / filename, unit / 'checkpoints' / f'{stage}_{suffix}.pt')
    tm = json.loads((teacher_dir / 'manifest.json').read_text())['resources']
    st = json.loads((student_dir / 'manifest.json').read_text())['resources']
    environment = dict(python=platform.python_version(), torch=torch.__version__, CUDA_VERSION=torch.version.cuda,
                       GPU_MODEL=torch.cuda.get_device_name(), GPU_VRAM=torch.cuda.get_device_properties(0).total_memory,
                       DRIVER_VERSION=subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip(),
                       PRECISION_MODE='float32; TF32 disabled')
    (unit / 'environment.txt').write_text(json.dumps(environment, indent=2)+'\n')
    (unit / 'git_info.txt').write_text(f'Frozen source SHA256: {prep.sha(REFERENCE)}\nWrapper SHA256: {prep.sha(__file__)}\n')
    timing = dict(TEACHER_TRAIN_TIME_SEC=tm['training_seconds'], STUDENT_TRAIN_TIME_SEC=st['training_seconds'],
                  TOTAL_MODEL_TRAIN_TIME_SEC=tm['training_seconds']+st['training_seconds'],
                  TARGET_INFERENCE_TIME_SEC=inference_seconds, TOTAL_WALL_TIME_SEC=time.perf_counter()-started,
                  PEAK_GPU_MEMORY_MB=max(tm['peak_allocated_bytes'], st['peak_allocated_bytes'])/2**20,
                  GPU_HOURS=(tm['training_seconds']+st['training_seconds'])/3600,
                  B_D_BUILD_TIME_SEC=sm['B_D_BUILD_TIME_SEC'], F_BUILD_TIME_SEC=sm['F_BUILD_TIME_SEC'])
    prep.write(unit / 'timing.json', timing)
    prep.write(unit / 'parameter_count.json', dict(Teacher=tm['parameters'], Student=st['parameters'], TOTAL_PARAMS=tm['parameters']+st['parameters']))
    prep.write(unit / 'checkpoint_manifest.json', dict(hashes={str(p.relative_to(unit)): prep.sha(p) for p in (unit / 'checkpoints').rglob('*.pt')}, optimizer_and_rng_preserved=True, scheduler='NONE_IN_FROZEN_REFERENCE'))
    prep.write(unit / 'leakage_audit.json', dict(target_atac_loaded=False, target_labels_loaded_after_embedding_hash=True,
               target_hvg_fitted=False, target_pca_fitted=False, target_population_fit_before_embedding=False,
               EVALUATION_TIME_UNSUPERVISED_SCALING=True, KNOWN_K_EVALUATION=True,
               target_checkpoint_selection=False, source_only_training=True))
    prep.write(unit / 'manifest.json', dict(status='COMPLETE', fold=fold, seed=seed, run_tier='pre_final_explicit_3seeds',
               artifact_hashes={str(p.relative_to(unit)): prep.sha(p) for p in unit.rglob('*') if p.is_file() and p.name!='manifest.json'}))
    print('RUN_COMPLETE', fold, seed, metrics['ari'], metrics['nmi'], flush=True)


def summarize():
    rows = []
    for fold in [f'target_D{i}' for i in range(1, 5)]:
        for seed in SEEDS:
            unit = OUT / fold / f'seed_{seed}'
            manifest = json.loads((unit / 'manifest.json').read_text())
            assert manifest['status'] == 'COMPLETE'
            for filename, digest in manifest['artifact_hashes'].items():
                assert prep.sha(unit / filename) == digest, str(unit / filename)
            metrics = json.loads((unit / 'metrics.json').read_text())
            timing = json.loads((unit / 'timing.json').read_text())
            params = json.loads((unit / 'parameter_count.json').read_text())
            rows.append(dict(fold=fold, seed=seed, **{k: v for k, v in metrics.items() if isinstance(v, (float, int))}, **timing, **params))
    metric_names = ['ari', 'nmi', 'cluster_acc', 'macro_f1', 'balanced_accuracy', 'knn_purity_at_15', 'cell_type_silhouette']
    per_fold, overall = {}, {}
    for fold in [f'target_D{i}' for i in range(1, 5)]:
        per_fold[fold] = {k: dict(mean=float(np.mean([r[k] for r in rows if r['fold'] == fold])),
                                 std=float(np.std([r[k] for r in rows if r['fold'] == fold], ddof=1))) for k in metric_names}
    seed_means = [{k: float(np.mean([r[k] for r in rows if r['seed'] == seed])) for k in metric_names} for seed in SEEDS]
    overall = {k: dict(mean=float(np.mean([r[k] for r in seed_means])), std=float(np.std([r[k] for r in seed_means], ddof=1))) for k in metric_names}
    result = dict(status='COMPLETE', run_tier='pre_final_explicit_3seeds', seeds_completed=list(SEEDS),
                  folds_completed=list(per_fold), per_fold=per_fold, per_seed=seed_means,
                  overall=overall, rows=rows,
                  std_scope='three seed-specific four-fold means; ddof=1', FULL_B2_METHOD_DRIFT='NO',
                  checkpoint_count=sum(len(list((OUT/r['fold']/f"seed_{r['seed']}"/'checkpoints').rglob('*.pt'))) for r in rows),
                  target_embedding_count=12, UMAP_READY=True, historical_reference='descriptive only; dataset changed',
                  registry_modified=False, ablations_launched=False)
    warning_path = warning_matrix()
    result.update(RESEARCH_GATES_MODE='WARNING', FATAL_GATES_UNCHANGED='YES',
                  warning_table=str(warning_path), valid_run_count=len(rows))
    prep.write(OUT / 'UPDATED_B2FULL_3SEED_REPORT.json', result)
    with (OUT / 'UPDATED_B2FULL_PER_RUN.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metric_columns = [('ari', 'ARI'), ('nmi', 'NMI'), ('cluster_acc', 'ACC'),
                      ('macro_f1', 'Macro-F1'), ('balanced_accuracy', 'Balanced accuracy'),
                      ('knn_purity_at_15', 'kNN Purity@15'), ('cell_type_silhouette', 'Cell-type Silhouette')]
    header = '| Fold | ' + ' | '.join(f'{label} mean +/- std' for _, label in metric_columns) + ' |'
    divider = '| --- | ' + ' | '.join('---' for _ in metric_columns) + ' |'
    lines = ['# Updated Full B2 D-mode', '',
             'Status: COMPLETE; explicit three-seed pre_final, not final 15-seed evidence.',
             'RESEARCH_GATES_MODE = WARNING; FATAL_GATES_UNCHANGED = YES.', '', header, divider]
    for fold, values in per_fold.items():
        lines.append(f"| {fold} | " + ' | '.join(
            f"{values[key]['mean']:.6f} +/- {values[key]['std']:.6f}" for key, _ in metric_columns) + ' |')
    lines += ['', '## Overall 3-seed metrics', '',
              'Each seed is first averaged across four folds; mean and sample SD are across the three seed means.', '',
              '| Metric | Mean +/- std |', '| --- | --- |']
    lines.extend(f'| {label} | {overall[key]["mean"]:.6f} +/- {overall[key]["std"]:.6f} |'
                 for key, label in metric_columns)
    lines += ['', '| Seed | ' + ' | '.join(label for _, label in metric_columns) + ' |',
              '| --- | ' + ' | '.join('---' for _ in metric_columns) + ' |']
    for seed, values in zip(SEEDS, seed_means):
        lines.append(f'| {seed} | ' + ' | '.join(f'{values[key]:.6f}' for key, _ in metric_columns) + ' |')
    resource_names = [('TOTAL_PARAMS', 'Total Params'), ('TEACHER_TRAIN_TIME_SEC', 'Teacher train time (s)'),
                      ('STUDENT_TRAIN_TIME_SEC', 'Student train time (s)'),
                      ('TOTAL_MODEL_TRAIN_TIME_SEC', 'Total train time (s)'),
                      ('TARGET_INFERENCE_TIME_SEC', 'Target inference time (s)'),
                      ('PEAK_GPU_MEMORY_MB', 'Peak VRAM (MB)'), ('GPU_HOURS', 'GPU hours')]
    lines += ['', '## Efficiency', '', '| Metric | Mean across 12 runs |', '| --- | --- |']
    lines.extend(f'| {label} | {np.mean([row[key] for row in rows]):.4f} |'
                 for key, label in resource_names)
    gpu_names = sorted({json.loads((OUT / row['fold'] / f"seed_{row['seed']}" / 'environment.txt').read_text())['GPU_MODEL'] for row in rows})
    lines += [f'| GPU model | {", ".join(gpu_names)} |', '',
              'Research-quality gates are diagnostic warnings only for this formal benchmark; thresholds were not tuned to individual runs.',
              'Only control-flow semantics changed. Model, architecture, data, losses, optimizer, schedule, masking, and checkpoint rule are unchanged.',
              'Fatal numerical, shape, source-data/hash, checkpoint-integrity, and target-leakage checks remain hard stops.',
              'Warning table: _benchmark_meta/TEACHER_RESEARCH_WARNING_MATRIX.csv.', '',
              'All 12 units use independent Teacher+Student training; only deterministic source artifacts shared.',
              'Target labels loaded only after target embedding save and hash. Evaluation scaling and known K are posthoc.',
              'Inherited paired cohorts and historical SOURCE split caveat retained. No donor independence claim.',
              'Regulatory preprocessing time is recorded separately from model training time. No ablation launched.']
    (OUT / 'UPDATED_B2FULL_3SEED_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('BENCHMARK_COMPLETE', json.dumps(overall), flush=True)


def warning_matrix():
    rows = []
    threshold = b2.CONFIG['teacher_gate']['min_rank']
    for fold in [f'target_D{i}' for i in range(1, 5)]:
        sources, _, _, _ = load_sources(fold)
        for seed in SEEDS:
            unit = OUT / fold / f'seed_{seed}'
            leakage = json.loads((unit / 'leakage_audit.json').read_text())
            assert leakage['target_atac_loaded'] is False
            assert leakage['target_labels_loaded_after_embedding_hash'] is True
            assert leakage['target_hvg_fitted'] is False and leakage['target_pca_fitted'] is False
            assert leakage['target_population_fit_before_embedding'] is False
            raw_target = np.load(unit / 'embeddings/target_embedding_raw.npy')
            assert raw_target.ndim == 2 and np.isfinite(raw_target).all()
            assert np.any(np.var(raw_target, axis=0) > 0), f'Fatal constant target embedding: {unit}'
            audit = json.loads((unit / 'evaluation/source_teacher_audit.json').read_text())
            gate = json.loads((unit / 'evaluation/teacher_gate_decision.json').read_text())
            gate_by_source = {row['source']: row for row in gate['source_rows']}
            for source_row in audit['source_rows']:
                ranks = [float(source_row[name]['effective_rank']) for name in ['z_R', 'z_F', 'z_T']]
                f_row = gate_by_source[source_row['source']]
                rows.append(dict(fold=fold, seed=seed, source=source_row['source'],
                    effective_rank_zR=ranks[0], effective_rank_zF=ranks[1], effective_rank_zT=ranks[2],
                    historical_effective_rank_threshold=threshold,
                    WARNING_EFFECTIVE_RANK_LOW='YES' if min(ranks) < threshold else 'NO',
                    variance_weighted_F_retention_R2=f_row['VARIANCE_WEIGHTED_R2'],
                    historical_F_retention_threshold=0.0,
                    WARNING_F_RETENTION='YES' if not f_row['passed'] else 'NO',
                    fatal_check_status='PASS',
                    diagnostic_only=True))
    path = OUT / '_benchmark_meta/TEACHER_RESEARCH_WARNING_MATRIX.csv'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', choices=[f'target_D{i}' for i in range(1, 5)])
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--audit-existing', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.audit_existing:
            audit_existing()
        elif args.summarize:
            summarize()
        elif args.dry_run:
            preflight()
        else:
            assert args.fold and args.seed and args.output_dir
            assert args.output_dir.resolve() == (OUT / args.fold / f'seed_{args.seed}').resolve()
            run(args.fold, args.seed, args.output_dir)


if __name__ == '__main__':
    main()
