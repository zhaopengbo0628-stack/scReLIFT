from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import fields, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import run_strict_lodo_two_step as strict
from v5d_student.student_encoder import RNAScaffoldVectorNormResidualEncoder

parent = strict.parent
ROOT = Path('/home/u202522802045/share/zpb')
GOAL1 = ROOT / 'result_contain/strict_lodo_two_step_20261002'
GATES = {
    'source_global_pearson_mean_gain': 0.02,
    'source_non_degrading_folds': 3,
    'source_recall_mean_regression_tolerance': 0.005,
    'target_ari_mean_gain': 0.01,
    'target_non_degrading_folds': 3,
    'target_nmi_mean_regression_tolerance': 0.005,
    'target_nmi_single_fold_regression_tolerance': 0.02,
}


def read(path):
    return json.loads(Path(path).read_text())


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def goal1_args():
    return argparse.Namespace(root=ROOT, out=GOAL1)


def dependencies():
    manifest = read(GOAL1 / 'COMPLETION_MANIFEST.json')
    decision = read(GOAL1 / 'FINAL_ADJUDICATION.json')
    assert manifest['status'] == 'COMPLETE' and len(manifest['units']) == 16
    assert decision['CAPACITY_BRANCH_AUTHORIZED'] == 'YES'
    for path, digest in manifest['source_files_sha256'].items():
        assert strict.sha(path) == digest, path
    for fold in parent.FOLDS:
        strict.verify_cache(GOAL1 / 'cache' / fold)
        for seed in strict.SEEDS:
            assert (GOAL1 / 'rna' / fold / f'seed_{seed}/target_metrics.json').is_file()
        assert (GOAL1 / 'parent' / fold / 'seed_20020130/diagnostics.json').is_file()
    return manifest, decision


@contextmanager
def gpu_lock():
    with (ROOT / '.goal2_gpu_training.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
        others = [x for x in active.splitlines() if x.strip() and x.strip() != str(os.getpid())]
        if others:
            raise RuntimeError(f'GPU already occupied: {others}')
        yield


def load_fold(fold):
    sources = strict.sources_for(goal1_args(), fold)
    saved = torch.load(GOAL1 / 'rna' / fold / 'seed_20020130/shared_rna_best.pt', map_location='cpu', weights_only=False)
    from v5s.denoise import DenoisingAutoencoder
    dae = DenoisingAutoencoder(saved['feature_count'], 512, 64)
    dae.load_state_dict(saved['model_state_dict'])
    old_config = read(GOAL1 / 'parent' / fold / 'seed_20020130/config.json')
    config = strict.V5DStudentConfig(**{f.name: old_config[f.name] for f in fields(strict.V5DStudentConfig) if f.name in old_config})
    train_banks, val_banks, train_targets, val_targets, hashes = {}, {}, {}, {}, {}
    for source in sources:
        directory = ROOT / 'result_contain/26_dmode_rna_scaffold_bounded_residual_20260929' / fold / 'pair_banks'
        tp, vp = directory / f'{source.name}_train.npz', directory / f'{source.name}_val.npz'
        train_banks[source.name], val_banks[source.name] = parent._project_topology_embeddings(strict.load_pair_bank(tp), strict.load_pair_bank(vp), 64)
        hashes.update({str(tp): strict.sha(tp), str(vp): strict.sha(vp)})
        for split, store in [('train', train_targets), ('val', val_targets)]:
            p = ROOT / 'result_contain' / strict.OLD_PARENT / fold / 'global_targets' / f'{source.name}_{split}.npz'
            store[source.name] = strict.load_global(p)
            hashes[str(p)] = strict.sha(p)
    wp = ROOT / 'result_contain' / strict.SOURCE_ROOT / fold / 'dataset_weights.npz'
    weights = {s.name: float(w) for s, w in zip(sources, np.load(wp)['omega'])}
    hashes[str(wp)] = strict.sha(wp)
    return sources, dae.encoder, saved['rna_mean'], saved['rna_std'], config, old_config, train_banks, val_banks, train_targets, val_targets, weights, hashes


def prepare(out):
    manifest, decision = dependencies()
    ratios, weights, records = [], [], []
    for fold in parent.FOLDS:
        sources, backbone, mean, std, *_ = load_fold(fold)
        old = parent.RNAScaffoldBoundedResidualEncoder(backbone, 64, 128, 0.10)
        checkpoint = GOAL1 / 'parent' / fold / 'seed_20020130/best_by_topology.pt'
        old.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state_dict'])
        old.eval()
        for source in sources:
            cap, actual = [], []
            for start in range(0, len(source.val), 512):
                x = torch.from_numpy(strict.normalized_rows(source, source.val[start:start + 512], mean, std))
                with torch.no_grad():
                    r = old.backbone(x)
                    delta = 0.10 * torch.tanh(old.adapter(r))
                    rn = r.norm(dim=1).clamp_min(1e-8)
                cap.extend((0.8 / rn).tolist())
                actual.extend((delta.norm(dim=1) / rn).tolist())
            ratios.extend(cap)
            weights.extend([1.0 / (12 * len(cap))] * len(cap))
            records.append({'fold': fold, 'source': source.name, 'old_hard_cap_relative': strict.distribution(cap), 'old_realized_relative': strict.distribution(actual), 'checkpoint_sha256': strict.sha(checkpoint)})
    ordered = np.argsort(ratios, kind='stable')
    cumulative = np.cumsum(np.asarray(weights)[ordered])
    rho = float(np.asarray(ratios)[ordered[np.searchsorted(cumulative, 0.5)]])
    derivation = {
        'rho': rho, 'eps': 1e-8, 'selection': 'one SOURCE-validation weighted median of old L2 ceiling / scaffold norm; each of 12 fold/source units has equal mass',
        'old': 'Delta=0.10*tanh(u); L2 ceiling=0.10*sqrt(64)=0.8; relative ceiling=0.8/||R||; derivative at u=0 is 0.10*I',
        'new': 'budget=rho*||R||; Delta=budget*u/sqrt(||u||^2+budget^2+eps); ||Delta||<budget; relative ceiling=rho',
        'scale_caveat': 'Matches the source-weighted median old relative hard budget, not every cell and not the realized mean. Fixed rho necessarily redistributes absolute budgets with scaffold norm.',
        'initial_gradient_gain': 'New Jacobian at zero is budget/sqrt(budget^2+eps)*I, approximately I, versus old 0.10*I. This approximately 10x initial gain is inherent to the requested formula, not an optimizer/lr change.',
        'no_rho_sweep': True, 'target_metrics_used': False, 'source_validation_records': records,
        'goal1_manifest_sha256': strict.sha(GOAL1 / 'COMPLETION_MANIFEST.json'), 'promotion_gates_predeclared': GATES,
        'confirmation_rule': 'seed31 only after seed30 passes all promising gates; seed32 only after seed31 independently passes the same fixed gates; compare to fixed strict parent seed30; shared source backbone remains fixed seed30',
    }
    strict.write(out / 'CAPACITY_SCALE_DERIVATION.json', derivation)
    strict.write(out / 'DEPENDENCY_LOCK.json', {'status': 'PASS', 'goal1_decisions': {k: v for k, v in decision.items() if k.isupper()}, 'source_hashes_verified': len(manifest['source_files_sha256']), 'strict_protocol_version': strict.sha(GOAL1 / 'COMPLETION_MANIFEST.json')})
    print('PREPARED_RHO', rho, flush=True)


def reserve(out, branch):
    assignment = out / 'TASK_ASSIGNMENTS.json'
    with (ROOT / 'result_contain/task_registry.csv.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assignments = read(assignment) if assignment.exists() else {}
        if branch in assignments:
            return Path(assignments[branch]['result_path']), assignments[branch]['task_id']
        ids = set()
        for directory in [ROOT / 'slum', ROOT / 'result_contain']:
            for p in directory.rglob('*'):
                match = re.match(r'^(?:task)?(\d+)(?:[_\-.]|$)', p.name)
                if match:
                    ids.add(int(match[1]))
        registry = ROOT / 'result_contain/task_registry.csv'
        with registry.open(newline='') as handle:
            reader = csv.DictReader(handle)
            columns, rows = reader.fieldnames, list(reader)
        ids.update(int(r['task_id']) for r in rows)
        task_id = max(ids) + 1
        directory = ROOT / 'result_contain' / f'{task_id}_strict_d_{branch}_goal2_20261002'
        directory.mkdir()
        now = datetime.now(timezone.utc).isoformat()
        row = dict.fromkeys(columns, '')
        row.update(task_id=str(task_id), status='RUNNING', experiment_name=f'strict_d_{branch}_goal2', model_version=f'Strict-Task31-{branch}', result_path=str(directory), created_at=now, updated_at=now, datasets='D1,D2,D3,D4', seeds='20020130', pid=str(os.getpid()), launch_time=now, launcher='direct_ssh', scheduler='none', launch_method='foreground_ssh')
        with registry.with_suffix('.csv.goal2.tmp').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows + [row])
        registry.with_suffix('.csv.goal2.tmp').replace(registry)
        assignments[branch] = {'task_id': task_id, 'result_path': str(directory)}
        strict.write(assignment, assignments)
        state = read(ROOT / 'CURRENT_RESEARCH_STATE.json')
        state['current_max_task_id'] = task_id
        state['goal2'] = {'status': 'RUNNING', 'preparation_path': str(out), 'assignments': assignments, 'combined_branch_authorized': False}
        strict.write(ROOT / 'CURRENT_RESEARCH_STATE.json', state)
        return directory, task_id


def checkpoint(path, model, optimizer, epoch, config_hash, rng, mean, std):
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': None, 'epoch': epoch, 'config_hash': config_hash, 'rna_mean': mean, 'rna_std': std, 'rng_state': {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all(), 'anchor_generator': rng.bit_generator.state}}, path)


def train_capacity(data, seed, rho, directory, smoke=False):
    sources, backbone, mean, std, config, original_config, tb, vb, tt, vt, weights, hashes = data
    seed_all(seed)
    model = RNAScaffoldVectorNormResidualEncoder(backbone, 64, 128, rho).cuda()
    rng = np.random.default_rng(seed)
    initial_local, initial_global = [], []
    with torch.no_grad():
        for source in sources:
            anchors = np.arange(min(config.batch_anchor, tb[source.name].anchors), dtype=np.int64)
            local, global_loss = parent._batch_losses(model, source, tb[source.name], tt[source.name], anchors, mean, std, config, torch.device('cuda'))
            initial_local.append(float(local.cpu()))
            initial_global.append(float(global_loss.cpu()))
    weighted = [weights[s.name] for s in sources]
    local_scale = max(float(np.average(initial_local, weights=weighted)), 1e-8)
    global_scale = max(float(np.average(initial_global, weights=weighted)), 1e-8)
    run_config = {**original_config, 'task_id': None if smoke else read(directory.parents[1] / 'TASK.json')['task_id'], 'branch': 'capacity', 'seed': seed, 'student_parameterization': 'R + budget*u/sqrt(||u||^2+budget^2+eps)', 'rho': rho, 'shared_backbone_seed': 20020130, 'scientific_changes': ['residual_parameterization'], 'smoke': smoke}
    strict.write(directory / 'config.json', run_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    steps = max(math.ceil(source.train.size / config.batch_anchor) for source in sources)
    best_topology, best_epoch, best_total = -float('inf'), 0, float('inf')
    best_state, best_total_state, history = None, None, []
    scaffold_hash = strict.snapshot(model.backbone)
    epochs = 1 if smoke else config.max_epochs
    for epoch in range(1, epochs + 1):
        model.train()
        train_local, train_global = [], []
        for _ in range(steps):
            losses = []
            for source in sources:
                anchors = rng.integers(0, tb[source.name].anchors, size=min(config.batch_anchor, tb[source.name].anchors))
                local, global_loss = parent._batch_losses(model, source, tb[source.name], tt[source.name], anchors, mean, std, config, torch.device('cuda'))
                losses.append((local / local_scale + global_loss / global_scale) * weights[source.name])
                train_local.append(float(local.detach().cpu()))
                train_global.append(float(global_loss.detach().cpu()))
            loss = torch.stack(losses).sum() / sum(weights.values())
            if not torch.isfinite(loss):
                raise FloatingPointError('Capacity loss is not finite')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if smoke:
                assert all(torch.isfinite(p.grad).all() for p in model.adapter.parameters() if p.grad is not None)
                assert float(model.adapter[-1].weight.grad.norm()) > 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        val_local, val_global, topology = [], [], []
        for source in sources:
            z = parent._encode_rows(model, source, np.arange(source.val.size), 'val', mean, std, torch.device('cuda')).detach().cpu().numpy()
            local, global_loss = parent._validation_losses(model, source, vb[source.name], vt[source.name], mean, std, config, torch.device('cuda'))
            val_local.append(local)
            val_global.append(global_loss)
            topology.append(parent._topology_score(z, vb[source.name]))
        vl, vg = float(np.average(val_local, weights=weighted)), float(np.average(val_global, weights=weighted))
        score = float(np.average([x['teacher_recall_at_10'] for x in topology], weights=weighted))
        total = vl / local_scale + vg / global_scale
        record = {'epoch': epoch, 'train_local_kl': float(np.mean(train_local)), 'train_global_landmark_mse': float(np.mean(train_global)), 'val_local_kl': vl, 'val_global_landmark_mse': vg, 'val_total_normalized': total, 'val_topology_score_teacher_recall_at_10': score, 'val_topology_by_dataset': {s.name: t for s, t in zip(sources, topology)}}
        history.append(record)
        if total < best_total:
            best_total, best_total_state = total, copy.deepcopy(model.state_dict())
        if score > best_topology:
            best_topology, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
            checkpoint(directory / 'best_by_topology.pt', model, optimizer, epoch, strict.sha(directory / 'config.json'), rng, mean, std)
        checkpoint(directory / 'latest.pt', model, optimizer, epoch, strict.sha(directory / 'config.json'), rng, mean, std)
        strict.write(directory / 'history.json', history)
        print('CAPACITY_EPOCH', directory.parent.name, seed, epoch, total, score, flush=True)
    assert scaffold_hash == strict.snapshot(model.backbone)
    torch.save({'model_state_dict': best_total_state, 'config': run_config, 'rna_mean': mean, 'rna_std': std}, directory / 'best_by_total_normalized_loss.pt')
    model.load_state_dict(best_state)
    model.eval()
    return model, {'history': history, 'best_topology_epoch': best_epoch, 'best_topology_score': best_topology, 'best_total_normalized': best_total, 'local_initial_scale': local_scale, 'global_initial_scale': global_scale}


def diagnostics(model, data, training):
    sources, _, mean, std, config, _, tb, vb, tt, vt, *_ = data
    before = strict.snapshot(model)
    relative, norm, cosine, fraction, tangential, radial = [], [], [], [], [], []
    gradients, fidelity = [], []
    params = [p for p in model.adapter.parameters() if p.requires_grad]
    for source in sources:
        for rows in [source.train, source.val]:
            for start in range(0, len(rows), 512):
                x = torch.from_numpy(strict.normalized_rows(source, rows[start:start + 512], mean, std)).cuda()
                with torch.no_grad():
                    r, delta, z = model.decompose_raw(x)
                    u = model.adapter(r)
                    rn = r.norm(dim=1).clamp_min(1e-8)
                    budget = model.rho * rn
                    denominator = (u.square().sum(dim=1) + budget.square() + model.eps).sqrt()
                    assert torch.isfinite(z).all() and torch.isfinite(delta).all()
                    assert (delta.norm(dim=1) <= budget + 1e-6).all()
                    relative.extend((delta.norm(dim=1) / rn).cpu().tolist())
                    norm.extend(delta.norm(dim=1).cpu().tolist())
                    cosine.extend(F.cosine_similarity(r, z).cpu().tolist())
                    fraction.extend((delta.norm(dim=1) / budget).cpu().tolist())
                    tangential.extend((budget / denominator).cpu().tolist())
                    radial.extend((budget * (budget.square() + model.eps) / denominator.pow(3)).cpu().tolist())
        z = strict.infer(model, source.rna[source.val], mean, std, raw=True)
        fidelity.append({'source': source.name, 'topology': parent._topology_score(z, vb[source.name]), 'global_teacher_fidelity': parent._global_metric(z, vt[source.name])})
        rng = np.random.default_rng(20020130 + int(source.name[-3:]))
        for _ in range(4):
            anchors = rng.integers(0, tb[source.name].anchors, size=min(config.batch_anchor, tb[source.name].anchors))
            local, global_loss = parent._batch_losses(model, source, tb[source.name], tt[source.name], anchors, mean, std, config, torch.device('cuda'))
            gl = torch.cat([g.flatten() for g in torch.autograd.grad(local, params, retain_graph=True)])
            gg = torch.cat([g.flatten() for g in torch.autograd.grad(global_loss, params)])
            nl, ng = float(gl.norm()), float(gg.norm())
            gradients.append({'source': source.name, 'cosine': float(F.cosine_similarity(gl[None], gg[None])), 'local_norm': nl, 'global_norm': ng, 'normalized_global_over_local_norm': (ng / training['global_initial_scale']) / max(nl / training['local_initial_scale'], 1e-12)})
    assert before == strict.snapshot(model)
    return {'parameter_updates': False, 'model_sha256_before_and_after': before, 'elementwise_tanh_present': False, 'delta_over_scaffold_norm': strict.distribution(relative), 'residual_norm': strict.distribution(norm), 'RNA_Z_cosine': strict.distribution(cosine), 'vector_budget_fraction': strict.distribution(fraction), 'vector_near_budget_fraction': float(np.mean(np.asarray(fraction) >= 0.9)), 'vector_budget_hit_fraction': float(np.mean(np.asarray(fraction) >= 0.99)), 'tangential_gradient_gain': strict.distribution(tangential), 'radial_gradient_gain': strict.distribution(radial), 'gradient_records': gradients, 'source_metrics': fidelity}


def run_capacity(out, seed, smoke):
    dependencies()
    rho = read(out / 'CAPACITY_SCALE_DERIVATION.json')['rho']
    if smoke:
        directory, task_id = out / 'capacity_smoke', None
    else:
        assert read(out / 'capacity_smoke/SMOKE_GATE.json')['status'] == 'PASS'
        if seed != 20020130:
            previous = read(out / 'TASK_ASSIGNMENTS.json')['capacity']['result_path']
            assert read(Path(previous) / f'ADJUDICATION_seed_{seed - 1}.json')['CAPACITY_BRANCH_PROMISING'] == 'YES'
        directory, task_id = reserve(out, 'capacity')
        strict.write(directory / 'TASK.json', {'task_id': task_id, 'branch': 'capacity', 'preparation_path': str(out), 'rho': rho})
    with gpu_lock():
        for fold in list(parent.FOLDS)[:1] if smoke else parent.FOLDS:
            unit = directory / fold / f'seed_{seed}'
            if (unit / 'manifest.json').exists():
                continue
            unit.mkdir(parents=True, exist_ok=True)
            data = load_fold(fold)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            model, training = train_capacity(data, seed, rho, unit, smoke)
            strict.write(unit / 'training.json', training)
            strict.write(unit / 'diagnostics.json', diagnostics(model, data, training))
            if not smoke:
                target = np.load(GOAL1 / 'cache' / fold / 'target_rna_test.npy', mmap_mode='r')
                inference_start = time.perf_counter()
                np.save(unit / 'z_target_test.npy', strict.infer(model, target, data[2], data[3], raw=True))
                torch.cuda.synchronize()
                inference_time = time.perf_counter() - inference_start
            else:
                inference_time, target = 0.0, []
            strict.write(unit / 'data_manifest.json', {'goal1_cache_manifest': read(GOAL1 / 'cache' / fold / 'manifest.json'), 'source_artifact_hashes': data[-1], 'target_labels_loaded': False, 'target_atac_loaded': False})
            (unit / 'environment.txt').write_text(f'Python {sys.version}\nTorch {torch.__version__}\nGPU {torch.cuda.get_device_name()}\n')
            strict.write(unit / 'manifest.json', {'status': 'SMOKE_COMPLETE' if smoke else 'EMBEDDING_FROZEN', 'seed': seed, 'fold': fold, 'task_id': task_id, 'strict_protocol_version': strict.sha(GOAL1 / 'COMPLETION_MANIFEST.json'), 'target_labels_used_for_training_or_selection': False, 'target_atac_accessed': False, 'scientific_changes': ['residual_parameterization'], 'runner_sha256': strict.sha(__file__), 'checkpoint_sha256': strict.sha(unit / 'best_by_topology.pt'), 'embedding_sha256': None if smoke else strict.sha(unit / 'z_target_test.npy'), 'resources': strict.resources(start, inference_time, len(target), model)})
            del model, data
            torch.cuda.empty_cache()
    if smoke:
        strict.write(directory / 'SMOKE_GATE.json', {'status': 'PASS', 'finite_delta_Z_losses': True, 'gradient_reaches_Z': True, 'scaffold_parameters_unchanged': True, 'vector_bound_checked': True, 'target_data_used': False, 'one_smoke_only': True})
    print('CAPACITY_COMPLETE', seed, 'smoke', smoke, flush=True)


def evaluate_capacity(out, seed):
    directory = Path(read(out / 'TASK_ASSIGNMENTS.json')['capacity']['result_path'])
    target_rows, comparisons = [], []
    for fold in parent.FOLDS:
        unit = directory / fold / f'seed_{seed}'
        manifest = read(unit / 'manifest.json')
        assert strict.sha(unit / 'z_target_test.npy') == manifest['embedding_sha256']
        embedding = np.load(unit / 'z_target_test.npy')
        cm = read(GOAL1 / 'cache' / fold / 'manifest.json')
        raw = Path(cm['raw_target_source'])
        processed = raw.parent if raw.suffix == '.npz' else raw.parent.parent / 'processed'
        test = np.load(GOAL1 / 'cache' / fold / 'target_test_index.npy')
        with (processed / 'label.csv').open(newline='') as handle:
            labels = np.asarray([x['cell_type'] for x in csv.DictReader(handle)])[test]
        reference = np.load(GOAL1 / 'rna' / fold / 'seed_20020130/z_target_test.npy')
        metrics = parent.evaluate_geometry(embedding, reference, labels, seed)
        strict.write(unit / 'metrics.json', {**metrics, 'labels_loaded_after_embedding_frozen': True, 'EVALUATION_TIME_UNSUPERVISED_SCALING': True, 'KNOWN_K_EVALUATION': True, 'label_file_sha256': strict.sha(processed / 'label.csv')})
        old_unit = GOAL1 / 'parent' / fold / 'seed_20020130'
        old_metrics = read(old_unit / 'target_metrics.json')
        diag = read(unit / 'diagnostics.json')
        new_sources, old_sources = diag['source_metrics'], read(old_unit / 'source_validation_metrics.json')
        pearson = lambda values: float(np.mean([x['global_teacher_fidelity']['normalized_distance_pearson'] for x in values]))
        recall = lambda values: float(np.mean([x['topology']['teacher_recall_at_10'] for x in values]))
        comparisons.append({'fold': fold, 'ari_delta': metrics['ari'] - old_metrics['ari'], 'nmi_delta': metrics['nmi'] - old_metrics['nmi'], 'source_pearson_delta': pearson(new_sources) - pearson(old_sources), 'source_recall_delta': recall(new_sources) - recall(old_sources)})
        target_rows.append({'fold': fold, 'seed': seed, 'ari': metrics['ari'], 'nmi': metrics['nmi']})
    means = {k: float(np.mean([x[k] for x in comparisons])) for k in comparisons[0] if k != 'fold'}
    fit = means['source_pearson_delta'] >= GATES['source_global_pearson_mean_gain'] and sum(x['source_pearson_delta'] >= 0 for x in comparisons) >= 3 and means['source_recall_delta'] >= -0.005
    generalization = means['ari_delta'] >= 0.01 and sum(x['ari_delta'] >= 0 for x in comparisons) >= 3 and means['nmi_delta'] >= -0.005 and min(x['nmi_delta'] for x in comparisons) >= -0.02
    result = {'seed': seed, 'target_metrics': target_rows, 'mean_ari': float(np.mean([x['ari'] for x in target_rows])), 'mean_nmi': float(np.mean([x['nmi'] for x in target_rows])), 'comparisons': comparisons, 'mean_deltas': means, 'ELEMENTWISE_SATURATION_FIXED': 'YES', 'GLOBAL_TEACHER_FIT_IMPROVED': 'YES' if fit else ('MIXED' if means['source_pearson_delta'] > 0 else 'NO'), 'TARGET_GENERALIZATION_IMPROVED': 'YES' if generalization else ('MIXED' if means['ari_delta'] > 0 else 'NO'), 'CAPACITY_BRANCH_PROMISING': 'YES' if fit and generalization else 'NO', 'gate_predeclared': GATES, 'parent_reference_seed': 20020130, 'next_seed': seed + 1 if fit and generalization and seed < 20020132 else None}
    strict.write(directory / f'ADJUDICATION_seed_{seed}.json', result)
    print('CAPACITY_ADJUDICATION', json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=['prepare', 'smoke', 'capacity', 'evaluate-capacity'], required=True)
    parser.add_argument('--seed', type=int, choices=strict.SEEDS, default=20020130)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / 'V5/Teacher-S10x20_Student-64D-KL/source'))
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'prepare':
        prepare(args.out)
    elif args.stage in ['smoke', 'capacity']:
        run_capacity(args.out, args.seed, args.stage == 'smoke')
    else:
        evaluate_capacity(args.out, args.seed)


if __name__ == '__main__':
    main()
