"""Matched Direct100 versus SOURCE-only pure-order curriculum; fixed Teacher30."""
import argparse
import copy
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

ROOT = Path('/home/u202522802045/share/zpb')
RESULTS = ROOT / 'result_contain'
PARENT = RESULTS / '35_b2_shared_regulatory_teacher_dense_kd_20261003'
PROBES = RESULTS / '37_b2_distillability_20261003'
OUT = RESULTS / '39_b2_matched_direct100_curriculum_20261004'
spec = importlib.util.spec_from_file_location('historical_b2', PARENT / 'source/run_b2_strict_lodo.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
GROUPS = ['HIGH_UTILITY_EASY', 'HIGH_UTILITY_HARD', 'LOW_UTILITY_REMAINING']
METHODS = ['DIRECT_100', 'CURRICULUM_20_20_60', 'RANDOM', 'REVERSE']
PROTOCOL = dict(task_id=39, epochs=100, seeds=b.SEEDS, controls_seed=20020130,
    phases=[20, 20, 60], checkpoint='best original full-zT SOURCE validation, epochs1-100',
    normalization='fixed original full-zT SOURCE-train dataset-balanced mu/sigma',
    utility='median SOURCE-val F-decoder MSE(zR)-MSE(zR+single raw Delta projection)',
    ease='median existing Task37 pseudo-held SOURCE D_skill; no probe retraining',
    target_schedule_only_change=True, teacher_retrained=False, evaluator=b.EVALUATOR,
    rng='historical NumPy default_rng sampler and Torch mask/dropout stream; reseed each paired run',
    rng_rationale='separating Torch streams would break historical direct trajectory',
    history_atol=1e-7, history_rtol=1e-5, parameter_atol=2e-6, parameter_rtol=1e-5,
    mechanism_rule='equal-source/fold/seed recovery-skill delta >= -0.005 for both highU groups',
    mechanism_stability_tolerance=0.005, ari_gain=0.005, strong_ari_gain=0.010,
    nonnegative_folds=3, nmi_delta_min=-0.005, controls_must_be_strictly_beaten=True,
    speed='first epoch reaching 90/95/99% of epoch0-to-best Direct100 full-val improvement',
    run_tier='pre_final_user_explicit_3seeds', cohort_caveat=b.CONFIG['cohort_caveat'])


class ProtocolMismatch(RuntimeError):
    pass


def status(value, detail):
    path = RESULTS / 'task_registry.csv'
    with path.open('r+') as h:
        fcntl.flock(h, fcntl.LOCK_EX)
        reader = csv.DictReader(h)
        columns, rows = reader.fieldnames, list(reader)
        match = [r for r in rows if r['task_id'] == '39']
        if not match:
            assert max(int(r['task_id']) for r in rows) == 38
            row = dict.fromkeys(columns, '')
            row.update(task_id='39', experiment_name=OUT.name, model_version='B2-pure-order-matched100',
                       result_path=str(OUT), created_at=b.now(), datasets='D1,D2,D3,D4')
            rows.append(row)
        else:
            assert len(match) == 1 and match[0]['result_path'] == str(OUT)
            row = match[0]
        row.update(status=value, updated_at=b.now(), pid=str(os.getpid()),
                   seeds=','.join(map(str, b.SEEDS)), run_script_path=str(Path(__file__).resolve()),
                   stdout_path=str(OUT/'stdout.log'), stderr_path=str(OUT/'stdout.log'))
        if value.startswith('COMPLETED'):
            row['completed_at'] = b.now()
        h.seek(0)
        csv.DictWriter(h, columns).writeheader()
        csv.DictWriter(h, columns).writerows(rows)
        h.truncate()
        h.flush()
        os.fsync(h.fileno())
    state = b.read(ROOT/'CURRENT_RESEARCH_STATE.json')
    state.update(current_max_task_id=max(39, state.get('current_max_task_id', 0)),
                 B2_CURRICULUM_Task39=dict(status=value, result_path=str(OUT), detail=detail),
                 next_action=detail)
    b.write(ROOT/'CURRENT_RESEARCH_STATE.json', state)
    with (ROOT/'RESEARCH_DECISION_LEDGER.md').open('a') as h:
        h.write(f'\n\n## Task39 pure-order curriculum {b.now()}\n\n- {value}: {detail}\n'
                '- Fixed Task35 Teacher30; matched100 epochs; full-zT SOURCE-only selection.\n')
    b.write(OUT/'manifest.json', dict(task_id=39, status=value, detail=detail,
            updated_at=b.now(), runner_sha256=b.sha(__file__), protocol_sha256=b.sha(OUT/'protocol.json')))


def load_fold(fold, device):
    sources, mean, std, inputs = b.load_sources(fold, PROBES)
    unit = PARENT/'teacher'/fold/'seed_20020130'
    teacher = b.Teacher(sources[0]['train']['rna'].shape[1], sources[0]['train']['field'].shape[1]).to(device)
    teacher.load_state_dict(torch.load(unit/'best.pt', map_location=device, weights_only=False)['model'])
    teacher.eval().requires_grad_(False)
    inputs[str(unit/'best.pt')] = b.sha(unit/'best.pt')
    cm = b.read(PARENT/'teacher_cache'/fold/'manifest.json')
    assert cm['teacher_checkpoint_sha256'] == inputs[str(unit/'best.pt')]
    audit = b.read(PROBES/'step1'/fold/'adjudication.json')['artifact_hashes']
    basis = PROBES/'step1'/fold/'basis.npz'
    assert b.sha(basis) == audit[str(basis)]
    inputs[str(basis)] = b.sha(basis)
    V = np.load(basis)['V']
    assert V.shape == (64, 64) and np.allclose(V.T@V, np.eye(64), atol=1e-5)
    for s in sources:
        for split in ['train', 'val']:
            zpath = PARENT/'teacher_cache'/fold/f"{s['name']}_{split}_zT.npy"
            assert b.sha(zpath) == cm['hashes'][str(zpath)]
            s[split]['full'] = np.load(zpath)
            inputs[str(zpath)] = b.sha(zpath)
            for kind in ['z_R', 'z_T']:
                p = PROBES/'step1'/fold/'teacher_cache'/f"{s['name']}_{split}_{kind}.npy"
                assert b.sha(p) == audit[str(p)]
                inputs[str(p)] = b.sha(p)
                if kind == 'z_R':
                    s[split]['zR'] = np.load(p)
                else:
                    assert np.array_equal(np.load(p), s[split]['full'])
            s[split]['delta'] = s[split]['full'] - s[split]['zR']
        p = PROBES/'step1'/fold/'probes'/s['name']/'direction_scores.npz'
        assert b.sha(p) == audit[str(p)]
        s['ease'] = np.load(p)['D']
        inputs[str(p)] = b.sha(p)
    mu = np.mean([s['train']['full'].mean(0) for s in sources], axis=0)
    var = np.mean([s['train']['full'].var(0)+(s['train']['full'].mean(0)-mu)**2 for s in sources], axis=0)
    sigma = np.sqrt(var).astype(np.float32)
    assert np.array_equal(mu, np.asarray(cm['latent_mean'], np.float32))
    assert np.array_equal(sigma, np.asarray(cm['latent_std'], np.float32))
    for s in sources:
        s.update(latent_mean=mu, latent_std=sigma, raw_zero_standardized=-mean/(std+1e-6))
    return sources, teacher, V, mean, std, inputs


def masks_for(groups, method):
    g = [np.asarray(groups[name], np.int64) for name in GROUPS]
    if method == 'DIRECT_100':
        ids = [np.arange(64)]*3
    elif method == 'CURRICULUM_20_20_60':
        ids = [g[0], np.concatenate(g[:2]), np.arange(64)]
    elif method == 'REVERSE':
        ids = [g[2], np.concatenate([g[2], g[1]]), np.arange(64)]
    else:
        order = np.random.default_rng(20020130).permutation(64)
        ids = [order[:len(g[0])], order[:len(g[0])+len(g[1])], np.arange(64)]
    return [np.isin(np.arange(64), i) for i in ids]


def targets(sources, V, mask):
    for s in sources:
        for split in ['train', 'val']:
            d = s[split]
            d['teacher_latent'] = d['full'] if mask.all() else (
                d['zR'] + (d['delta']@V[:, mask])@V[:, mask].T).astype(np.float32)


def prepare(device):
    if (OUT/'protocol.json').exists():
        assert b.read(OUT/'protocol.json') == PROTOCOL
        assert b.read(OUT/'STEP1_GATE.json')['CURRICULUM_VALID'] == 'YES'
        return
    used = set()
    for d in [ROOT/'slum', RESULTS]:
        for p in d.iterdir():
            match = re.match(r'(?:task)?(\d+)(?:_|\b)', p.name, re.I)
            if match:
                used.add(int(match.group(1)))
    with (RESULTS/'task_registry.csv').open() as h:
        used.update(int(r['task_id']) for r in csv.DictReader(h))
    assert max(used) == 38 and 39 not in used
    assert b.sha(PARENT/'source/run_b2_strict_lodo.py') == '3de1ad3642e7af276658d9b239ccaf9457adf7f8e6078b95a237cf4605f2c7e1'
    OUT.mkdir()
    b.write(OUT/'protocol.json', PROTOCOL)
    reuse = dict(global_used_ids=sorted(used), input_hashes={}, folds=[], target_read=False)
    for fold in b.FOLDS:
        sources, teacher, V, _, _, inputs = load_fold(fold, device)
        utility = []
        for s in sources:
            d = s['val']
            base = b.forward_numpy(teacher.field_decoder, d['zR'], device)
            baseline = np.mean((base-d['field'])**2)
            coeff = d['delta']@V
            scores = []
            for m in range(64):
                added = (d['zR']+coeff[:, m:m+1]*V[:, m]).astype(np.float32)
                pred = b.forward_numpy(teacher.field_decoder, added, device)
                scores.append(float(baseline-np.mean((pred-d['field'])**2)))
            utility.append(scores)
        U = np.median(utility, axis=0)
        E = np.median([s['ease'] for s in sources], axis=0)
        assert np.isfinite(U).all() and np.isfinite(E).all()
        high = U >= np.median(U)
        easy = high & (E >= np.median(E[high]))
        groups = dict(zip(GROUPS, [np.flatnonzero(easy).tolist(),
            np.flatnonzero(high & ~easy).tolist(), np.flatnonzero(~high).tolist()]))
        assert all(groups.values()) and sorted(sum(groups.values(), [])) == list(range(64))
        schedules, energy = {}, {}
        for method in METHODS:
            masks = masks_for(groups, method)
            schedules[method] = [m.astype(int).tolist() for m in masks]
            values = []
            for mask in masks:
                targets(sources, V, mask)
                values.append(float(np.mean([np.sum((s['train']['teacher_latent']-s['train']['zR'])**2)/
                    np.sum(s['train']['delta']**2) for s in sources])))
                if mask.all():
                    assert all(np.array_equal(s['train']['teacher_latent'], s['train']['full']) for s in sources)
            assert abs(values[-1]-1) < 1e-7
            energy[method] = values
        record = dict(fold=fold, groups=groups, U=U.tolist(), E=E.tolist(), utility_by_source=utility,
            ease_by_source=[s['ease'].tolist() for s in sources], sources=[s['name'] for s in sources],
            masks=schedules, phase_energy=energy, input_hashes=inputs,
            utility_proxy_not_biological_ground_truth=True, projection='raw Delta @ V; no mean correction outside groups')
        b.write(OUT/'step1'/fold/'groups.json', record)
        reuse['input_hashes'].update(inputs)
        reuse['folds'].append(dict(fold=fold, sizes={k:len(v) for k,v in groups.items()}, phase_energy=energy))
        batch = next(b.batches(sources, np.random.default_rng(20020130)))
        b.seed_all(20020130)
        student = copy.deepcopy(teacher.rna_encoder).requires_grad_(True)
        x = b.batch_tensor(batch, 'rna', device)
        before = {k:v.clone() for k,v in student.state_dict().items()}
        loss = F.mse_loss(student(x), b.batch_tensor(batch, 'full', device))
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in student.parameters())
        assert all(torch.equal(v, before[k]) for k,v in student.state_dict().items())
        del teacher, student
    b.write(OUT/'reuse_audit.json', reuse)
    b.write(OUT/'STEP1_GATE.json', dict(CURRICULUM_VALID='YES', PAIRING_VALID='PENDING_HISTORICAL_REPRODUCTION',
        smoke='finite forward/backward, no optimizer step or parameter update', source_only=True))
    (OUT/'source').mkdir()
    shutil.copy2(__file__, OUT/'source'/Path(__file__).name)
    shutil.copy2(PARENT/'source/run_b2_strict_lodo.py', OUT/'source/historical_b2.py')
    status('PREPARED', 'SOURCE groups frozen; Direct100 historical prefix must pass before curriculum')
    print('STEP1_PASS', reuse['folds'], flush=True)


def model_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def validate(model, sources, device):
    model.eval()
    mu = torch.as_tensor(sources[0]['latent_mean'], device=device)
    std = torch.as_tensor(sources[0]['latent_std'], device=device)
    partial, full = [], []
    for s in sources:
        current, original, counts = [], [], []
        for k in range(0, len(s['val']['rna']), 512):
            x = torch.from_numpy(s['val']['rna'][k:k+512]).to(device)
            pred = model(x)
            t = torch.from_numpy(s['val']['teacher_latent'][k:k+512]).to(device)
            z = torch.from_numpy(s['val']['full'][k:k+512]).to(device)
            current.append([F.mse_loss((pred-mu)/(std+1e-6), (t-mu)/(std+1e-6)).item()])
            original.append([F.mse_loss((pred-mu)/(std+1e-6), (z-mu)/(std+1e-6)).item()])
            counts.append(len(x))
        partial.append(np.average(current, axis=0, weights=counts).tolist())
        full.append(np.average(original, axis=0, weights=counts).tolist())
    return float(np.mean(partial)), float(np.mean(full)), full


def historical_compare(model, record, old, epoch, historical_unit):
    original = old[epoch-1]
    for key in ['train', 'val', 'val_by_source']:
        a = record['full_val_by_source'] if key == 'val_by_source' else (
            [record['full_teacher_val']] if key == 'val' else record['train'])
        if not np.allclose(a, original[key], atol=PROTOCOL['history_atol'], rtol=PROTOCOL['history_rtol']):
            raise ProtocolMismatch(f'historical loss mismatch {historical_unit} epoch{epoch} {key}: {a} vs {original[key]}')
    comparisons = []
    for name in ['best.pt', 'latest.pt']:
        checkpoint = torch.load(historical_unit/name, map_location='cpu', weights_only=False)
        if epoch == checkpoint['epoch']:
            errors = []
            for key, value in model.state_dict().items():
                a, z = value.detach().cpu(), checkpoint['model'][key]
                errors.append(float((a-z).abs().max()))
                if not torch.allclose(a, z, atol=PROTOCOL['parameter_atol'], rtol=PROTOCOL['parameter_rtol']):
                    raise ProtocolMismatch(f'historical parameter mismatch {historical_unit} epoch{epoch} {name} {key}')
            comparisons.append(dict(checkpoint=name, epoch=epoch, max_parameter_abs_error=max(errors)))
    return comparisons


def train(fold, seed, method, sources, teacher, V, inputs, device):
    unit = OUT/'student'/method/fold/f'seed_{seed}'
    if (unit/'manifest.json').exists():
        manifest = b.read(unit/'manifest.json')
        assert manifest['completed_epochs'] == 100 and b.sha(unit/'best.pt') == manifest['best_checkpoint_sha256']
        return
    unit.mkdir(parents=True, exist_ok=True)
    groups_path = OUT/'step1'/fold/'groups.json'
    g = b.read(groups_path)
    masks = [np.asarray(m, bool) for m in g['masks'][method]]
    inputs = dict(inputs, **{str(groups_path):b.sha(groups_path)})
    b.write(unit/'config.json', dict(protocol=PROTOCOL, historical_student=b.CONFIG, fold=fold, seed=seed, method=method))
    b.write(unit/'data_manifest.json', dict(input_hashes=inputs, target_read=False))
    b.seed_all(seed)
    model = copy.deepcopy(teacher.rna_encoder).requires_grad_(True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=b.CONFIG['student_lr'], weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    targets(sources, V, np.ones(64, bool))
    baseline = validate(model, sources, device)[1]
    initialization = model_digest(model)
    initial = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=0,
                   rng=b.rng_state(rng), input_hashes=inputs, initial_model_sha256=initialization,
                   epoch0_full_teacher_val=baseline, protocol_sha256=b.sha(OUT/'protocol.json'))
    history, best, start = [], float('inf'), 0
    if (unit/'latest.pt').exists():
        cp = torch.load(unit/'latest.pt', map_location=device, weights_only=False)
        assert cp['input_hashes'] == inputs and cp['protocol_sha256'] == initial['protocol_sha256']
        model.load_state_dict(cp['model'])
        optimizer.load_state_dict(cp['optimizer'])
        b.restore_rng(cp['rng'], rng)
        history, best, start = cp['history'], cp['best_score'], cp['epoch']
    else:
        b.save_checkpoint(unit/'initial.pt', initial)
    mu = torch.as_tensor(sources[0]['latent_mean'], device=device)
    std = torch.as_tensor(sources[0]['latent_std'], device=device)
    zero = torch.as_tensor(sources[0]['raw_zero_standardized'], device=device)
    historical_unit = PARENT/'student'/fold/f'seed_{seed}'
    old = b.read(historical_unit/'history.json')
    parameter_checks = history[-1].get('historical_parameter_checks', []) if history else []
    direct_unit = OUT/'student'/'DIRECT_100'/fold/f'seed_{seed}'
    direct_history = b.read(direct_unit/'history.json') if method != 'DIRECT_100' else None
    if direct_history is not None:
        direct_manifest = b.read(direct_unit/'manifest.json')
        assert direct_manifest['initial_model_sha256'] == initialization
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start+1, 101):
        phase = 0 if epoch <= 20 else (1 if epoch <= 40 else 2)
        targets(sources, V, masks[phase])
        model.train()
        losses = []
        sampler_hash, masking_hash, torch_hash = (hashlib.sha256() for _ in range(3))
        for batch in b.batches(sources, rng):
            for s, indices in batch:
                sampler_hash.update(s['name'].encode())
                sampler_hash.update(indices.tobytes())
            torch_hash.update(torch.get_rng_state().numpy().tobytes())
            torch_hash.update(torch.cuda.get_rng_state().cpu().numpy().tobytes())
            x = b.batch_tensor(batch, 'rna', device)
            nonzero = b.batch_tensor(batch, 'raw_nonzero', device)
            mask = (torch.rand_like(x)<b.CONFIG['mask_rate']) & nonzero
            masking_hash.update(mask.cpu().numpy().tobytes())
            x = torch.where(mask, zero, x)
            target = b.batch_tensor(batch, 'teacher_latent', device)
            pred = model(x)
            total = F.mse_loss((pred-mu)/(std+1e-6), (target-mu)/(std+1e-6))
            assert torch.isfinite(total)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert torch.isfinite(norm)
            optimizer.step()
            losses.append([total.item()])
        current, score, by_source = validate(model, sources, device)
        record = dict(epoch=epoch, train=np.mean(losses, axis=0).tolist(), phase=phase+1,
            current_phase_val=current, full_teacher_val=score, full_val_by_source=by_source,
            curriculum_mask=masks[phase].astype(int).tolist(),
            sampler_sha256=sampler_hash.hexdigest(), mask_sha256=masking_hash.hexdigest(),
            torch_rng_sha256=torch_hash.hexdigest(), elapsed_seconds=time.perf_counter()-started)
        if method == 'DIRECT_100' and epoch <= len(old):
            parameter_checks.extend(historical_compare(model, record, old, epoch, historical_unit))
        record['historical_parameter_checks'] = list(parameter_checks)
        if direct_history is not None:
            d = direct_history[epoch-1]
            for key in ['sampler_sha256', 'mask_sha256', 'torch_rng_sha256']:
                if record[key] != d[key]:
                    raise ProtocolMismatch(f'paired RNG mismatch {method} {fold} {seed} epoch{epoch} {key}')
        history.append(record)
        improved = score < best
        if improved:
            best = score
        cp = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=b.rng_state(rng),
            epoch=epoch, best_score=best, history=history, input_hashes=inputs,
            initial_model_sha256=initialization, epoch0_full_teacher_val=baseline,
            protocol_sha256=b.sha(OUT/'protocol.json'))
        b.save_checkpoint(unit/'latest.pt', cp)
        if improved:
            b.save_checkpoint(unit/'best.pt', cp)
            b.save_checkpoint(unit/f'new_best_epoch_{epoch:03}.pt', cp)
        if epoch % 10 == 0:
            b.save_checkpoint(unit/f'epoch_{epoch:03}.pt', cp)
        b.write(unit/'history.json', history)
        if epoch % 10 == 0 or epoch == 1:
            print('EPOCH', method, fold, seed, epoch, score, 'best', best, flush=True)
    best_cp = torch.load(unit/'best.pt', map_location='cpu', weights_only=False)
    manifest = dict(status='TRAINING_COMPLETE', completed_epochs=100, best_epoch=best_cp['epoch'],
        best_source_val_score=best, epoch0_full_teacher_val=baseline,
        best_checkpoint_sha256=b.sha(unit/'best.pt'), epoch100_checkpoint_sha256=b.sha(unit/'epoch_100.pt'),
        initial_model_sha256=initialization, input_hashes=inputs, seed=seed, fold=fold, method=method,
        historical_direct_trajectory_match='YES' if method == 'DIRECT_100' else 'NA',
        historical_overlap_epochs=len(old) if method == 'DIRECT_100' else None,
        historical_parameter_checks=parameter_checks,
        resources=dict(training_seconds=time.perf_counter()-started, parameters=sum(p.numel() for p in model.parameters()),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu=torch.cuda.get_device_name()), target_used_for_selection=False)
    b.write(unit/'manifest.json', manifest)
    print('TRAIN_COMPLETE', method, fold, seed, 'best_epoch', best_cp['epoch'], flush=True)


def source_mechanism(model, sources, V, groups, device):
    rows = []
    for s in sources:
        d = s['val']
        pred = b.forward_numpy(model, d['rna'], device)
        expected = d['delta']@V
        recovered = (pred-d['zR'])@V
        group_rows = {}
        for name in GROUPS:
            ids = groups[name]
            energy = np.sum(expected[:, ids]**2)
            error = np.sum((recovered[:, ids]-expected[:, ids])**2)
            group_rows[name] = dict(mse=float(np.mean((recovered[:, ids]-expected[:, ids])**2)),
                recovery_skill=float(1-error/max(energy, 1e-12)),
                amplitude_ratio=float(np.sqrt(np.sum(recovered[:, ids]**2)/max(energy, 1e-12))))
        rows.append(dict(source=s['name'], full=b.geometry(pred, d['full']), groups=group_rows))
    return rows


def posthoc(device):
    rows = []
    for fold in b.FOLDS:
        sources, teacher, V, mean, std, _ = load_fold(fold, device)
        groups = b.read(OUT/'step1'/fold/'groups.json')['groups']
        for method in METHODS:
            for seed in (b.SEEDS if method in METHODS[:2] else [20020130]):
                unit = OUT/'student'/method/fold/f'seed_{seed}'
                model = copy.deepcopy(teacher.rna_encoder).requires_grad_(False)
                model.load_state_dict(torch.load(unit/'best.pt', map_location=device, weights_only=False)['model'])
                mechanism = source_mechanism(model, sources, V, groups, device)
                b.write(unit/'source_mechanism.json', mechanism)
                cm = b.read(b.GOAL1/'cache'/fold/'manifest.json')
                p = b.GOAL1/'cache'/fold/'target_rna_test.npy'
                assert b.sha(p) == cm['cache_sha256']['target_rna_test.npy']
                x = ((np.load(p)-mean)/(std+1e-6)).astype(np.float32)
                started = time.perf_counter()
                z = b.forward_numpy(model, x, device)
                elapsed = time.perf_counter()-started
                digest = b.save_array(unit/'z_target_test.npy', z)
                frozen = dict(status='EMBEDDING_FROZEN', embedding_sha256=digest,
                    target_rna_sha256=b.sha(p), frozen_at=b.now(), target_atac_used=False,
                    target_statistics_fitted=False, target_labels_used_for_selection=False,
                    inference_seconds=elapsed, cells_per_second=len(z)/max(elapsed, 1e-9))
                b.write(unit/'frozen_embedding_manifest.json', frozen)
                labels = b.labels_after_frozen(fold)
                metrics = dict(fold=fold, seed=seed, method=method, **frozen, **b.evaluate(z, labels))
                from sklearn.neighbors import NearestNeighbors
                from scipy.spatial.distance import pdist
                from scipy.stats import spearmanr
                count = min(16, len(z))
                neighbors = NearestNeighbors(n_neighbors=count).fit(z).kneighbors(z, return_distance=False)
                purity = np.mean([np.mean(labels[np.asarray([j for j in ids if j != i][:15])]==labels[i])
                                  for i, ids in enumerate(neighbors)])
                idx = np.arange(min(1024, len(z)))
                dist = pdist(z[idx])
                different = (labels[idx, None] != labels[idx][None, :])[np.triu_indices(len(idx), 1)]
                same = float(dist[~different].mean())
                other = float(dist[different].mean())
                metrics['topology'] = dict(knn_purity_at15=float(purity),
                    geometry_spearman=float(spearmanr(dist, different.astype(float)).statistic),
                    geometry_pearson=float(np.corrcoef(dist, different.astype(float))[0, 1]),
                    geometry_reference='posthoc label dissimilarity, not ATAC geometry',
                    same_type_distance=same, different_type_distance=other, separation_ratio=other/max(same, 1e-12),
                    target_ATAC_geometry='NA')
                assert b.sha(unit/'z_target_test.npy') == digest
                b.write(unit/'metrics.json', metrics)
                rows.append(metrics)
                print('TARGET', method, fold, seed, metrics['ari'], metrics['nmi'], flush=True)
        del teacher
        torch.cuda.empty_cache()
    return rows


def summarize(rows):
    per_seed = [dict(seed=seed, **{m:float(np.mean([r[m] for r in rows if r['seed']==seed]))
                for m in ['ari', 'nmi']}) for seed in sorted({r['seed'] for r in rows})]
    return dict(per_seed=per_seed,
        mean={m:float(np.mean([r[m] for r in per_seed])) for m in ['ari', 'nmi']},
        std={m:float(np.std([r[m] for r in per_seed], ddof=1)) if len(per_seed)>1 else None for m in ['ari', 'nmi']},
        std_scope='sample std across seed-specific four-fold means')


def tables(rows, name):
    keys = list(rows[0])
    with (OUT/f'{name}.csv').open('w') as h:
        writer = csv.DictWriter(h, keys)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['| '+' | '.join(keys)+' |', '| '+' | '.join(['---']*len(keys))+' |']
    for row in rows:
        lines.append('| '+' | '.join(str(row[k]) for k in keys)+' |')
    (OUT/f'{name}.md').write_text('\n'.join(lines)+'\n')


def adjudicate(rows):
    summary = {m:summarize([r for r in rows if r['method']==m]) for m in METHODS}
    historical = []
    for fold in b.FOLDS:
        for seed in b.SEEDS:
            historical.append(b.read(PARENT/'student'/fold/f'seed_{seed}'/'metrics.json'))
    historical_summary = summarize(historical)
    per_seed = [dict(seed=a['seed'], delta_ari=a['ari']-d['ari'], delta_nmi=a['nmi']-d['nmi'])
                for a, d in zip(summary[METHODS[1]]['per_seed'], summary[METHODS[0]]['per_seed'])]
    per_fold = []
    for fold in b.FOLDS:
        curr = [r for r in rows if r['fold']==fold and r['method']==METHODS[1]]
        direct = [r for r in rows if r['fold']==fold and r['method']==METHODS[0]]
        per_fold.append(dict(fold=fold, **{f'delta_{m}':float(np.mean([r[m] for r in curr])-np.mean([r[m] for r in direct]))
                                        for m in ['ari', 'nmi']}))
    mechanism, speeds, pairing = [], [], []
    for fold in b.FOLDS:
        for seed in b.SEEDS:
            units = [OUT/'student'/m/fold/f'seed_{seed}' for m in METHODS[:2]]
            manifests = [b.read(u/'manifest.json') for u in units]
            histories = [b.read(u/'history.json') for u in units]
            mechanisms = [b.read(u/'source_mechanism.json') for u in units]
            for a, d in zip(mechanisms[1], mechanisms[0]):
                for g in GROUPS:
                    mechanism.append(dict(fold=fold, seed=seed, source=a['source'], group=g,
                        direct_recovery=d['groups'][g]['recovery_skill'], curriculum_recovery=a['groups'][g]['recovery_skill'],
                        delta_recovery=a['groups'][g]['recovery_skill']-d['groups'][g]['recovery_skill'],
                        direct_mse=d['groups'][g]['mse'], curriculum_mse=a['groups'][g]['mse'],
                        direct_amplitude=d['groups'][g]['amplitude_ratio'], curriculum_amplitude=a['groups'][g]['amplitude_ratio']))
            baseline = manifests[0]['epoch0_full_teacher_val']
            best = min(h['full_teacher_val'] for h in histories[0])
            final = histories[0][-1]['full_teacher_val']
            for method, history in zip(METHODS[:2], histories):
                for reference, value in [('best', best), ('final', final)]:
                    for fraction in [.90, .95, .99]:
                        threshold = baseline-fraction*(baseline-value)
                        hits = [h['epoch'] for h in history if h['full_teacher_val'] <= threshold]
                        speeds.append(dict(fold=fold, seed=seed, method=method, reference=reference,
                            fraction=fraction, threshold=threshold, epoch=hits[0] if hits else None))
            pairing.append(dict(fold=fold, seed=seed, initialization_sha256=manifests[0]['initial_model_sha256'],
                methods=METHODS if seed==20020130 else METHODS[:2], RNG_epoch_digests_match=True,
                historical_overlap_epochs=manifests[0]['historical_overlap_epochs'],
                historical_parameter_checks=manifests[0]['historical_parameter_checks']))
    recovery = {g:dict(direct=float(np.mean([r['direct_recovery'] for r in mechanism if r['group']==g])),
        curriculum=float(np.mean([r['curriculum_recovery'] for r in mechanism if r['group']==g])),
        delta=float(np.mean([r['delta_recovery'] for r in mechanism if r['group']==g]))) for g in GROUPS}
    gain = summary[METHODS[1]]['mean']['ari']-summary[METHODS[0]]['mean']['ari']
    nmi = summary[METHODS[1]]['mean']['nmi']-summary[METHODS[0]]['mean']['nmi']
    curr30 = summary[METHODS[1]]['per_seed'][0]['ari']
    checks = dict(mean_gain_positive=gain>0, minimum_gain=gain>=.005,
        nonnegative_folds=sum(r['delta_ari']>=0 for r in per_fold)>=3, nmi=nmi>=-.005,
        random=curr30>summary['RANDOM']['mean']['ari'], reverse=curr30>summary['REVERSE']['mean']['ari'],
        source_hard=recovery[GROUPS[1]]['delta']>=-.005, source_easy=recovery[GROUPS[0]]['delta']>=-.005)
    supported = all(checks.values())
    adopt = supported and summary[METHODS[1]]['mean']['ari']>=historical_summary['mean']['ari']
    result = dict(TASK_ID=39, CURRICULUM_VALID='YES', PAIRING_VALID='YES', HISTORICAL_DIRECT_TRAJECTORY_MATCH='YES',
        MATCHED_DIRECT100_3SEED=summary[METHODS[0]], CURRICULUM_3SEED=summary[METHODS[1]],
        HISTORICAL_FULLB2_3SEED=historical_summary, DELTA_CURRICULUM_VS_DIRECT100_PER_SEED=per_seed,
        DELTA_CURRICULUM_VS_DIRECT100_PER_FOLD=per_fold,
        DELTA_DIRECT100_VS_HISTORICAL_FULLB2={m:summary[METHODS[0]]['mean'][m]-historical_summary['mean'][m] for m in ['ari','nmi']},
        RANDOM_CURRICULUM_SEED30=summary['RANDOM'], REVERSE_CURRICULUM_SEED30=summary['REVERSE'],
        SOURCE_HIGH_UTILITY_EASY_RECOVERY=recovery[GROUPS[0]], SOURCE_HIGH_UTILITY_HARD_RECOVERY=recovery[GROUPS[1]],
        SOURCE_LOW_UTILITY_RECOVERY=recovery[GROUPS[2]], success_checks=checks,
        CURRICULUM_EFFECT_SUPPORTED='YES' if supported else 'NO',
        CURRICULUM_EFFECT=('STRONG' if gain>=.010 else 'MODEST_BUT_SUPPORTED') if supported else 'NOT_SUPPORTED',
        ADOPT_CURRICULUM_AS_MAIN='YES' if adopt else 'NO', METHOD_FREEZE='B2_CURRICULUM' if adopt else 'FULL_B2',
        INDEPENDENT_TEACHER_REPLICATION_RECOMMENDED='YES' if adopt else 'NO', EXTERNAL_VALIDATION_REQUIRED='YES',
        mean_ari_gain=gain, mean_nmi_gain=nmi, run_tier=PROTOCOL['run_tier'], next_action='STOP',
        target_used_for_selection=False, teacher_retrained=False, cohort_caveat=PROTOCOL['cohort_caveat'])
    b.write(OUT/'ADJUDICATION.json', result)
    b.write(OUT/'paired_run_mapping.json', pairing)
    b.write(OUT/'optimization_speed.json', speeds)
    tables(mechanism, 'source_group_recovery')
    tables([{k:r[k] for k in ['method','fold','seed','ari','nmi','ami','cluster_acc','accuracy',
        'balanced_accuracy','macro_f1','weighted_f1']} for r in rows], 'target_metrics')
    resources = []
    for r in rows:
        m = b.read(OUT/'student'/r['method']/r['fold']/f"seed_{r['seed']}"/'manifest.json')
        res = m['resources']
        resources.append(dict(Method=r['method'], Dataset=r['fold'], seed=r['seed'],
            total_wall_seconds=res['training_seconds']+r['inference_seconds'],
            trainable_parameters=res['parameters'], peak_GPU_bytes=res['peak_allocated_bytes'],
            inference_seconds=r['inference_seconds'], cells_per_second=r['cells_per_second'], GPU_model=res['gpu']))
    tables(resources, 'efficiency')
    tables([dict(method=m, **{f'{k}_mean':summary[m]['mean'][k] for k in ['ari','nmi']},
                 **{f'{k}_std':summary[m]['std'][k] for k in ['ari','nmi']}) for m in METHODS], 'summary')
    (OUT/'README.md').write_text('# Task39 Matched Direct100 Curriculum\n\n'+json.dumps(result, indent=2)+'\n')
    status('COMPLETED' if supported else 'COMPLETED_REJECT', 'Adjudicated; '+result['METHOD_FREEZE']+' frozen; STOP; no tuning/replication/external launch')
    state = b.read(ROOT/'CURRENT_RESEARCH_STATE.json')
    state['B2_CURRICULUM_Task39']['adjudication'] = result
    b.write(ROOT/'CURRENT_RESEARCH_STATE.json', state)
    print('FINAL', json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'run'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device('cuda')
    with threadpool_limits(limits=2), (ROOT/'.b2_gpu_training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            prepare(device)
            if args.stage == 'prepare':
                return
            assert b.read(OUT/'STEP1_GATE.json')['CURRICULUM_VALID'] == 'YES'
            status('RUNNING', 'Matched Direct100 historical trajectory audit; no curriculum until gate passes')
            for fold in b.FOLDS:
                sources, teacher, V, _, _, inputs = load_fold(fold, device)
                for seed in b.SEEDS:
                    train(fold, seed, 'DIRECT_100', sources, teacher, V, inputs, device)
                del teacher
                torch.cuda.empty_cache()
            b.write(OUT/'HISTORICAL_DIRECT_GATE.json', dict(HISTORICAL_DIRECT_TRAJECTORY_MATCH='YES',
                PAIRING_VALID='YES', units=12, criterion=PROTOCOL, target_read=False))
            status('RUNNING', '12 Direct100 historical overlaps pass; run paired curriculum plus random/reverse seed30')
            for fold in b.FOLDS:
                sources, teacher, V, _, _, inputs = load_fold(fold, device)
                for method in METHODS[1:]:
                    for seed in (b.SEEDS if method==METHODS[1] else [20020130]):
                        train(fold, seed, method, sources, teacher, V, inputs, device)
                del teacher
                torch.cuda.empty_cache()
            adjudicate(posthoc(device))
        except Exception as exc:
            if OUT.exists() and (OUT/'protocol.json').exists():
                b.write(OUT/'ERROR.json', dict(error=str(exc), traceback=traceback.format_exc(), time=b.now()))
                status('BLOCKED_PAIRING_MISMATCH' if isinstance(exc, ProtocolMismatch) else 'FAILED', str(exc))
            raise


if __name__ == '__main__':
    main()
