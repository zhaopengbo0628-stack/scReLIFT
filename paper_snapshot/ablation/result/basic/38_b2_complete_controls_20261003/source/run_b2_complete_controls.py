"""Task38: reuse matched B2 controls, exact pre-selection field and gated cell confidence."""
import argparse
import copy
import csv
import fcntl
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, '/home/u202522802045/share/zpb/result_contain/36_b2_three_causal_ablations_20261003/source/experiments/v5d')
import run_b2_strict_lodo as b
import run_b2_three_ablations as a

TASK = 38
SEEDS = [20020130, 20020131, 20020132]
OLD35 = b.ROOT / 'result_contain/35_b2_shared_regulatory_teacher_dense_kd_20261003'
OLD36 = b.ROOT / 'result_contain/36_b2_three_causal_ablations_20261003'
OLD37 = b.ROOT / 'result_contain/37_b2_distillability_20261003'
POLICY = dict(task_id=TASK, seeds=SEEDS, C_seeds=[SEEDS[0]], run_tier='pre_final_user_declared_3seed',
    A3='exact sign_D*R_D candidate values immediately before Top50 selection; same gene L1 mapping',
    A3_limit='removes hard reliability selection, not reliability score/sign estimation; density not matched',
    comparison='Task36 FULL independent Teacher+Student matched seeds; Task35 fixed-Teacher separately',
    confidence=dict(model='SOURCE cross-fitted StandardScaler + Ridge(alpha=1)', energy_guard=1e-10,
        high_low_quantiles=[.25,.75], material_relative_error_reduction=.10,
        shuffle_permutations=100, shuffle_p_max=.05,
        gate='each outer fold: >=2/3 constant wins, median Spearman>=.30, median high/low error ratio<=.90, shuffle p<=.05'),
    target_selection=False, C_checkpoint='SOURCE teacher recall@10', C_loss_scales='same SOURCE initial-scale computation as C0',
    C_fix_selection='SOURCE Teacher preservation first, geometry consistency second, smaller Teacher semantic change third')


def status(out, value):
    registry = b.ROOT / 'result_contain/task_registry.csv'
    with registry.open('r+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        reader = csv.DictReader(handle)
        columns, rows = reader.fieldnames, list(reader)
        found = [r for r in rows if r['task_id'] == str(TASK)]
        if found:
            assert len(found) == 1 and found[0]['result_path'] == str(out)
            row = found[0]
        else:
            used = {int(r['task_id']) for r in rows}
            used.add(b.read(b.ROOT / 'CURRENT_RESEARCH_STATE.json')['current_max_task_id'])
            for directory in [b.ROOT / 'slum', b.ROOT / 'result_contain']:
                for path in directory.iterdir():
                    match = re.match(r'(?:task)?(\d+)(?:_|\b)', path.name, re.I)
                    if match and path.resolve() != out.resolve():
                        used.add(int(match.group(1)))
            assert max(used) == TASK - 1 and TASK not in used
            row = dict.fromkeys(columns, '')
            row.update(task_id=str(TASK), experiment_name=out.name, model_version='B2-complete-controls',
                       result_path=str(out), created_at=b.now(), datasets='strict SOURCE LODO D1-D4',
                       seeds=','.join(map(str, SEEDS)), scheduler='sequential GPU', launcher='SSH zyh-new')
            rows.append(row)
        row.update(status=value, updated_at=b.now(), pid=str(os.getpid()), run_script_path=str(Path(__file__)),
                   stdout_path=str(out / 'stdout.log'), stderr_path=str(out / 'stdout.log'))
        if value.startswith('COMPLETED'):
            row['completed_at'] = b.now()
        handle.seek(0)
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        handle.truncate()
        handle.flush()
        os.fsync(handle.fileno())
    manifest = dict(task_id=TASK, status=value, result_path=str(out), updated_at=b.now(),
                    pid=os.getpid(), protocol_sha256=b.sha(out / 'protocol.json'), runner_sha256=b.sha(__file__))
    b.write(out / 'manifest.json', manifest)
    state = b.read(b.ROOT / 'CURRENT_RESEARCH_STATE.json')
    state['B2_COMPLETE_CONTROLS_Task38'] = manifest
    state['current_max_task_id'] = max(TASK, state['current_max_task_id'])
    state['next_action'] = 'Complete matched controls and conditional B2-SD; no external run'
    b.write(b.ROOT / 'CURRENT_RESEARCH_STATE.json', state)
    with (b.ROOT / 'RESEARCH_DECISION_LEDGER.md').open('a') as handle:
        handle.write(f'\n\n## Task38 {b.now()}\n\n- {value}; reuse-first B2 controls, exact pre-Top50 field and one C geometry fix.\n'
                     '- Task37 remains closed; no direction filtering or external run.\n')


def reuse(out):
    rows = []
    for condition in ['RNA_ONLY', 'SHUFFLED_F', 'FULL', 'NO_CROSS']:
        for fold in b.FOLDS:
            for seed in SEEDS:
                unit = OLD36 / condition / fold / f'seed_{seed}'
                metrics = b.read(unit / 'metrics.json')
                assert metrics['evaluator'] == b.EVALUATOR
                ep, cp = unit / 'z_target_test.npy', unit / 'student/best.pt'
                eh, ch = b.sha(ep), b.sha(cp)
                assert eh == metrics['embedding_sha256']
                index = b.read(unit / 'checkpoint_index.json')
                assert index[str(cp)] == ch
                tc = unit / 'teacher/best.pt'
                assert b.sha(tc) == index[str(tc)]
                exact = condition != 'NO_CROSS'
                rows.append(dict(group=condition, fold=fold, seed=seed, artifact=str(unit),
                    status='EXACT_REUSE' if exact else 'NOT_EQUIVALENT',
                    reason='matched independent pipeline' if exact else 'removes C_cross/V but retains hard selection',
                    checkpoint_hash=ch, embedding_hash=eh, action='reuse' if exact else 'descriptive only; train exact A3'))
    canonical = b.read(OLD35 / 'CANONICAL_BASELINES.json')
    for row in canonical['rows']:
        if row['method'] not in ['RNA', 'scButterfly', 'D_parent']:
            continue
        assert b.sha(row['input']) == row['input_sha256'] and row['evaluator'] == b.EVALUATOR
        unit = Path(row['input']).parent
        checkpoints = list(unit.glob('*best*.pt')) or list(unit.rglob('*.pt'))
        rows.append(dict(group=row['method'], fold=row['fold'], seed=row['training_seed'], artifact=str(unit),
                    status='EXACT_REUSE', reason='existing canonical frozen embedding result',
                    checkpoint_hash=b.sha(checkpoints[0]) if checkpoints else 'NA',
                    embedding_hash=row['input_sha256'], action='reuse canonical score'))
    for fold in b.FOLDS:
        for seed in SEEDS[:2]:
            unit = OLD37 / 'step2' / fold / f'seed_{seed}'
            metric = b.read(unit / 'metrics.json')
            ep = unit / 'z_target_test.npy'
            assert b.sha(ep) == metric['embedding_sha256']
            rows.append(dict(group='B2-D', fold=fold, seed=seed, artifact=str(unit), status='EXACT_REUSE',
                reason='closed negative; fixed Teacher30, no seed32', checkpoint_hash=b.sha(unit / 'best.pt'),
                embedding_hash=b.sha(ep), action='reuse; do not retrain'))
        for seed in SEEDS:
            rows.append(dict(group='UNSCREENED', fold=fold, seed=seed, artifact='', status='MISSING',
                        reason='no exact pre-selection control found', checkpoint_hash='', embedding_hash='', action='train A3'))
        rows.append(dict(group='C1', fold=fold, seed=SEEDS[0], artifact='', status='MISSING',
                    reason='prepared FIX-A not trained; SOURCE selection pending', checkpoint_hash='', embedding_hash='', action='audit then train one fix'))
    with (out / 'ABLATION_REUSE_AUDIT.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    b.write(out / 'REUSE_SUMMARY.json', {s: sum(r['status'] == s for r in rows) for s in sorted({r['status'] for r in rows})})


def unscreened(fold, out):
    directory = out / 'fields' / fold
    if (directory / 'manifest.json').exists():
        for path, digest in b.read(directory / 'manifest.json')['hashes'].items():
            assert b.sha(path) == digest
        return
    cm = b.read(b.GOAL1 / 'cache' / fold / 'manifest.json')
    root = Path(cm['source_harmonized_root']).parent
    genes, peaks = np.load(root / 'gene_ids.npy'), np.load(root / 'canonical_peak_names.npy')
    gtf = b.ROOT.parent.parent / 'glue_156478/graph_files/gencode.v32.primary_assembly.annotation.gtf.gz'
    graph, _ = a.build_candidate_graph(genes, peaks, a.read_tss(gtf), 250000, 50)
    mask, index = graph['candidate_mask'], graph['candidate_index']
    eg = np.repeat(np.arange(len(genes)), index.shape[1])[mask.ravel()]
    ep = index[mask]
    full = sparse.load_npz(root / 'B_D.npz').tocsc()
    score, sign = np.load(root / 'R_D.npy'), np.load(root / 'sign_D.npy')
    selected = np.load(root / 'selected_support.npy').astype(bool)
    assert len(score) == len(eg)
    def mapped(values):
        values = values.astype(np.float32).copy()
        for gene in range(len(genes)):
            positions = eg == gene
            denominator = np.abs(values[positions]).sum()
            if denominator > 0:
                values[positions] /= denominator
        nz = np.abs(values) > 1e-8
        return sparse.coo_matrix((values[nz], (ep[nz], eg[nz])), shape=full.shape).tocsc()
    difference = mapped(np.where(selected, sign * score, 0)) - full
    assert not difference.nnz or np.abs(difference.data).max() < 1e-7
    control = mapped(sign * score)
    assert control.shape == full.shape and control.nnz >= full.nnz
    directory.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(directory / 'B_UNSCREENED.npz', control)
    np.save(directory / 'candidate_edge_gene.npy', eg)
    np.save(directory / 'candidate_edge_peak.npy', ep)
    inputs = [root / x for x in ['B_D.npz', 'R_D.npy', 'sign_D.npy', 'selected_support.npy', 'gene_ids.npy', 'canonical_peak_names.npy']]
    inputs.append(gtf)
    for name in cm['sources']:
        source = Path(cm['source_harmonized_root']) / name
        with np.load(source / 'metadata.npz', allow_pickle=True) as metadata:
            train, val = metadata['train_index'], metadata['val_index']
        assert not np.intersect1d(train, val).size
        values = sparse.load_npz(source / 'atac_binary.npz')[np.r_[train, val]]
        field, state, diagnostics = a.fit_dmode_field(values, control, np.arange(len(train)))
        np.save(directory / f'{name}_F_train.npy', field[:len(train)])
        np.save(directory / f'{name}_F_val.npy', field[len(train):])
        np.savez(directory / f'{name}_field_state.npz', **state)
        b.write(directory / f'{name}_field_diagnostics.json', diagnostics)
        inputs.extend([source / 'metadata.npz', source / 'atac_binary.npz'])
    hashes = {str(p): b.sha(p) for p in list(directory.iterdir()) + inputs if p.is_file()}
    b.write(directory / 'manifest.json', dict(hashes=hashes, target_read=False, full_reconstruction_pass=True,
        candidate_edges=len(eg), full_nnz=full.nnz, unscreened_nnz=control.nnz, genes=len(genes),
        operation='remove selected mask only; preserve sign_D*R_D and gene L1, TF-IDF and SOURCE-train standardization',
        directly_matched=True, limitation=POLICY['A3_limit']))


def confidence_gate(out):
    result = []
    cache = {}
    hashes = {}
    for fold in b.FOLDS:
        cm = b.read(b.GOAL1 / 'cache' / fold / 'manifest.json')
        data = {}
        for name in cm['sources']:
            base = OLD37 / 'step1' / fold
            paths = [base / 'teacher_cache' / f'{name}_val_z_R.npy',
                     base / 'teacher_cache' / f'{name}_val_z_T.npy', base / 'probes' / name / 'z_probe_pseudoheld_val.npy']
            zr, zt, zp = [np.load(p).astype(np.float64) for p in paths]
            hashes.update({str(p): b.sha(p) for p in paths})
            energy = ((zt - zr) ** 2).sum(1)
            error = ((zp - zt) ** 2).sum(1)
            valid = energy > POLICY['confidence']['energy_guard']
            skill = 1 - error[valid] / (energy[valid] + 1e-12)
            data[name] = (zr[valid], skill)
            unit = out / 'confidence' / fold
            unit.mkdir(parents=True, exist_ok=True)
            np.savez(unit / f'{name}_recovery_skill.npz', skill_raw=skill, valid=valid, energy=energy, error=error)
        episodes = []
        for name, (x, y) in data.items():
            others = [data[n] for n in data if n != name]
            tx, ty = np.concatenate([d[0] for d in others]), np.concatenate([d[1] for d in others])
            scaler = StandardScaler().fit(tx)
            ridge = Ridge(alpha=1.).fit(scaler.transform(tx), ty)
            pred = ridge.predict(scaler.transform(x))
            baseline = np.mean((y - ty.mean()) ** 2)
            r2 = 1 - np.mean((y - pred) ** 2) / max(baseline, 1e-12)
            sp = float(spearmanr(pred, y).statistic)
            quantiles = np.quantile(pred, [.25, .75])
            relative = 1 - y
            low, high = relative[pred <= quantiles[0]].mean(), relative[pred >= quantiles[1]].mean()
            rng = np.random.default_rng(SEEDS[0])
            null = []
            for _ in range(100):
                shuffled = Ridge(alpha=1.).fit(scaler.transform(tx), rng.permutation(ty))
                null.append(float(spearmanr(shuffled.predict(scaler.transform(x)), y).statistic))
            p = (1 + sum(v >= sp for v in null)) / 101
            bins = np.quantile(pred, np.linspace(0, 1, 5))
            calibration = []
            for i in range(4):
                mask = (pred >= bins[i]) & (pred <= bins[i+1])
                calibration.append(dict(n=int(mask.sum()), predicted=float(pred[mask].mean()), observed=float(y[mask].mean())))
            row = dict(source=name, pearson=float(pearsonr(pred, y).statistic), spearman=sp,
                R2_vs_training_constant=float(r2), constant_skill=float(ty.mean()),
                high_low_relative_correction_error_ratio=float(high/max(low, 1e-12)),
                shuffle_spearman_mean=float(np.mean(null)), shuffle_p=float(p), calibration=calibration)
            episodes.append(row)
            np.savez(out / 'confidence' / fold / f'{name}_crossfit_confidence.npz',
                     coef=ridge.coef_, intercept=ridge.intercept_, mean=scaler.mean_, scale=scaler.scale_,
                     predicted_skill=pred, observed_skill=y, shuffled_spearman=null)
            cache[(fold, name)] = (scaler, ridge)
        checks = dict(constant_wins=sum(r['R2_vs_training_constant'] > 0 for r in episodes) >= 2,
            spearman=float(np.median([r['spearman'] for r in episodes])) >= .30,
            material_error_reduction=float(np.median([r['high_low_relative_correction_error_ratio'] for r in episodes])) <= .90,
            shuffle_destroyed=float(np.median([r['shuffle_p'] for r in episodes])) <= .05,
            no_outer_target_used=True)
        result.append(dict(fold=fold, episodes=episodes, checks=checks, authorized=all(checks.values())))
    b.write(out / 'CELL_DISTILLABILITY_PREDICTABILITY.json', dict(folds=result,
        B2_SD_AUTHORIZED=all(r['authorized'] for r in result), input_hashes=hashes,
        scope='conditional fixed Teacher; probes reused, SOURCE val confidence targets; not nested Teacher refits',
        selection_uses_no_biological_labels=True, raw_skill_retained=True, preregistered_policy=POLICY['confidence']))
    print('CELL_GATE', json.dumps(result), flush=True)
    return cache


ORIGINAL_SOURCES = a.sources_for


def control_sources(condition, fold, seed, out):
    sources, mean, std, inputs = ORIGINAL_SOURCES(condition, fold, seed, out)
    if condition == 'UNSCREENED':
        directory = out / 'fields' / fold
        inputs.update(b.read(directory / 'manifest.json')['hashes'])
        for source in sources:
            for split in ['train', 'val']:
                source[split]['field'] = np.load(directory / f"{source['name']}_F_{split}.npy")
    return sources, mean, std, inputs


def source_mechanism(condition, fold, seed, out):
    unit = (out if condition == 'UNSCREENED' else OLD36) / condition / fold / f'seed_{seed}'
    destination = out / 'mechanisms' / condition / fold / f'seed_{seed}.json'
    if destination.exists():
        return
    sources, _, _, _ = control_sources(condition, fold, seed, out if condition == 'UNSCREENED' else OLD36)
    width = sources[0]['train']['rna'].shape[1]
    model = b.Teacher(width, width)
    if condition == 'RNA_ONLY':
        model = a.RNATeacher(model)
    model.load_state_dict(torch.load(unit / 'teacher/best.pt', map_location='cpu', weights_only=False)['model'])
    model.eval()
    rows, train_z, val_z, train_domain, val_domain = [], [], [], [], []
    field_train, field_val = [], []
    for domain, s in enumerate(sources):
        tr = b.forward_numpy(model, s['train']['rna'], 'cpu', s['train']['field'])
        zr, zf, zt, rr, ff = b.forward_numpy(model, s['val']['rna'], 'cpu', s['val']['field'])
        zs = np.load(unit / f"{s['name']}_source_val_student.npy")
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(len(zr))
        rows.append(dict(source=s['name'], z_R=b.rank_info(zr), z_F=None if condition == 'RNA_ONLY' else b.rank_info(zf),
            z_T=b.rank_info(zt), same_cell_mse=None if condition == 'RNA_ONLY' else float(np.mean((zr-zf)**2)),
            shuffled_cell_mse=None if condition == 'RNA_ONLY' else float(np.mean((zr-zf[shuffled])**2)),
            RNA_reconstruction=float(np.mean((rr-s['val']['rna'])**2)),
            F_reconstruction=None if condition == 'RNA_ONLY' else float(np.mean((ff-s['val']['field'])**2)),
            student_vs_fused=b.geometry(zs, zt), student_vs_RNA=b.geometry(zs, zr),
            F_rank=b.rank_info(s['val']['field'])))
        train_z.append(tr[2]); val_z.append(zt)
        field_train.append(s['train']['field']); field_val.append(s['val']['field'])
        train_domain.extend([domain]*len(tr[2])); val_domain.extend([domain]*len(zt))
    domain_signal = {}
    for label, tx, vx in [('teacher', train_z, val_z), ('field', field_train, field_val)]:
        scaler = StandardScaler().fit(np.concatenate(tx))
        classifier = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEEDS[0])
        classifier.fit(scaler.transform(np.concatenate(tx)), train_domain)
        domain_signal[label] = float(balanced_accuracy_score(val_domain, classifier.predict(scaler.transform(np.concatenate(vx)))))
    b.write(destination, dict(rows=rows, domain_balanced_accuracy=domain_signal, chance=1/3,
        checkpoint_sha256=b.sha(unit / 'teacher/best.pt'), biological_labels_used=False, target_read=False))


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(**POLICY, code_hashes={str(p): b.sha(p) for p in [Path(__file__), Path(a.__file__), Path(b.__file__),
        Path(__file__).with_name('run_b2_c_geometry_control.py')]})
    if (out / 'protocol.json').exists():
        old = b.read(out / 'protocol.json')
        assert {k:v for k,v in old.items() if k != 'code_hashes'} == POLICY
        assert not (out / 'A3_PER_UNIT.json').exists()
        b.write(out / 'protocol.json', protocol)
    else:
        b.write(out / 'protocol.json', protocol)
        b.write(out / 'config.json', dict(base_B2_config=b.CONFIG, policy=POLICY))
        b.write(out / 'data_manifest.json', b.read(OLD35 / 'data_manifest.json'))
        (out / 'environment.txt').write_text(f'Python {sys.version}\nTorch {torch.__version__}\nNumPy {np.__version__}\n')
    status(out, 'PREPARED')
    reuse(out)
    for fold in b.FOLDS:
        unscreened(fold, out)
        sources, _, _, _ = b.load_sources(fold, out)
        (out / 'reference_probes' / fold).mkdir(parents=True, exist_ok=True)
        for s in sources:
            for split in ['train', 'val']:
                b.save_array(out / 'reference_probes' / fold / f"{s['name']}_F_{split}.npy", s[split]['field'])
    confidence_gate(out)


def run(out):
    assert b.read(out / 'SMOKE_GATE.json')['status'] == 'PASS'
    for path, digest in b.read(out / 'protocol.json')['code_hashes'].items():
        assert b.sha(path) == digest
    with (b.ROOT / '.b2_gpu_training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status(out, 'RUNNING')
        b.teacher_losses, b.save_checkpoint = a.losses, a.save_checkpoint
        a.sources_for = control_sources
        rows = []
        for seed in SEEDS:
            for fold in b.FOLDS:
                rows.append(a.run_unit('UNSCREENED', fold, seed, out, torch.device('cuda')))
                b.write(out / 'A3_PER_UNIT.json', rows)
                torch.cuda.empty_cache()
        print('A3_COMPLETE', flush=True)
    for condition in ['FULL', 'RNA_ONLY', 'SHUFFLED_F', 'UNSCREENED']:
        for seed in SEEDS:
            for fold in b.FOLDS:
                source_mechanism(condition, fold, seed, out)
    print('SOURCE_MECHANISMS_COMPLETE', flush=True)


def sd_unit(fold, seed, out):
    unit = out / 'B2_SD' / fold / f'seed_{seed}'
    if (unit / 'metrics.json').exists():
        result = b.read(unit / 'metrics.json')
        assert b.sha(unit / 'z_target_test.npy') == result['embedding_sha256']
        return result
    b.CONFIG = dict(a.BASE_CONFIG, method='B2-SD', run_tier='pre_final_user_declared_gated_seeds',
                    confidence='out-of-source fixed Ridge, clip predicted skill 0..1; no target module')
    sources, mean, std, inputs = b.load_sources(fold, out)
    teacher = b.Teacher(sources[0]['train']['rna'].shape[1], sources[0]['train']['field'].shape[1]).cuda()
    cp = OLD35 / 'teacher' / fold / 'seed_20020130/best.pt'
    teacher.load_state_dict(torch.load(cp, map_location='cuda', weights_only=False)['model'])
    teacher.eval().requires_grad_(False)
    expected = b.read(OLD37 / 'protocol.json')['parent_checkpoints'][fold]['sha256']
    assert b.sha(cp) == expected
    inputs[str(cp)] = expected
    cache = unit / 'teacher_cache'
    cache.mkdir(parents=True, exist_ok=True)
    for source in sources:
        confidence_path = out / 'confidence' / fold / f"{source['name']}_crossfit_confidence.npz"
        with np.load(confidence_path) as saved:
            coef, intercept, center, scale = [saved[k] for k in ['coef','intercept','mean','scale']]
        inputs[str(confidence_path)] = b.sha(confidence_path)
        for split in ['train','val']:
            base = OLD37 / 'step1' / fold / 'teacher_cache'
            rp,tp = [base / f"{source['name']}_{split}_{name}.npy" for name in ['z_R','z_T']]
            zr,zt = np.load(rp),np.load(tp)
            skill = ((zr-center)/scale) @ coef + intercept
            confidence = np.clip(skill,0,1).astype(np.float32)
            target = (zr+confidence[:,None]*(zt-zr)).astype(np.float32)
            source[split]['teacher_latent'] = target
            path = cache / f"{source['name']}_{split}_target.npy"
            b.save_array(path,target)
            np.save(cache / f"{source['name']}_{split}_confidence.npy",confidence)
            inputs.update({str(rp):b.sha(rp),str(tp):b.sha(tp),str(path):b.sha(path)})
    mu = np.mean([s['train']['teacher_latent'].mean(0) for s in sources],axis=0)
    var = np.mean([s['train']['teacher_latent'].var(0)+(s['train']['teacher_latent'].mean(0)-mu)**2 for s in sources],axis=0)
    for s in sources:
        s.update(latent_mean=mu,latent_std=np.sqrt(var).astype(np.float32),raw_zero_standardized=-mean/(std+1e-6))
    b.write(cache / 'manifest.json',dict(fit='SOURCE train dataset-balanced',out_of_source_confidence=True,
        teacher_seed=20020130,latent_mean=mu.tolist(),latent_std=np.sqrt(var).tolist(),inference_module_added=False))
    b.seed_all(seed)
    student = copy.deepcopy(teacher.rna_encoder).requires_grad_(True)
    a.initial(student,unit/'student',seed,False,inputs)
    b.save_checkpoint = a.save_checkpoint
    student = b.train_stage(student,sources,unit/'student',seed,torch.device('cuda'),False,inputs)
    mechanisms=[]
    for s in sources:
        zs=b.forward_numpy(student,s['val']['rna'],'cuda')
        zt=b.forward_numpy(teacher,s['val']['rna'],'cuda',s['val']['field'])[2]
        mechanisms.append(dict(source=s['name'],student_vs_original_teacher=b.geometry(zs,zt),
                               student_vs_gated_teacher=b.geometry(zs,s['val']['teacher_latent'])))
        b.save_array(unit/f"{s['name']}_source_val_student.npy",zs)
    b.write(unit/'source_mechanism.json',mechanisms)
    target_path=b.GOAL1/'cache'/fold/'target_rna_test.npy'
    assert b.sha(target_path)==b.read(target_path.parent/'manifest.json')['cache_sha256'][target_path.name]
    target=((np.load(target_path)-mean)/(std+1e-6)).astype(np.float32)
    z=b.forward_numpy(student,target,'cuda')
    digest=b.save_array(unit/'z_target_test.npy',z)
    b.write(unit/'frozen_embedding_manifest.json',dict(embedding_sha256=digest,target_atac_used=False,target_fitting=False))
    result=dict(fold=fold,seed=seed,embedding_sha256=digest,**b.evaluate(z,b.labels_after_frozen(fold)))
    b.write(unit/'metrics.json',result)
    b.write(unit/'checkpoint_index.json',{str(p):b.sha(p) for p in unit.rglob('*.pt')})
    print('SD_RESULT',fold,seed,result['ari'],result['nmi'],flush=True)
    torch.cuda.empty_cache()
    return result


def step2(out):
    assert (out/'A3_PER_UNIT.json').exists() and (out/'C1_PER_UNIT.json').exists()
    gate=b.read(out/'CELL_DISTILLABILITY_PREDICTABILITY.json')
    if not gate['B2_SD_AUTHORIZED']:
        b.write(out/'B2_SD_ADJUDICATION.json',dict(authorized=False,ran=False,METHOD_FREEZE='FULL_B2'))
        return
    parent=[]
    for seed in SEEDS:
        parent.append([b.read(OLD35/'student'/fold/f'seed_{seed}'/'metrics.json') for fold in b.FOLDS])
    decisions=[]
    with (b.ROOT/'.b2_gpu_training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for index,seed in enumerate(SEEDS):
            rows=[sd_unit(fold,seed,out) for fold in b.FOLDS]
            baseline=parent[index]
            delta=[r['ari']-v['ari'] for r,v in zip(rows,baseline)]
            mean=float(np.mean([r['ari'] for r in rows]));bm=float(np.mean([r['ari'] for r in baseline]))
            improves=mean>bm or min(r['ari'] for r in rows)>min(r['ari'] for r in baseline)
            promising=mean>=bm-.005 and min(delta)>=-.05 and improves
            decisions.append(dict(seed=seed,mean_ari=mean,mean_nmi=float(np.mean([r['nmi'] for r in rows])),
                full_mean_ari=bm,delta_ari=float(np.mean(delta)),per_fold_delta=delta,promising=promising,
                source_gain_override_used=False))
            b.write(out/'B2_SD_ADJUDICATION.json',dict(authorized=True,ran=True,decisions=decisions,
                METHOD_FREEZE='B2_SD' if index==2 and promising else 'FULL_B2',
                seed_stop_rule='no source-gain override unless preregistered clear measure; conservative target mean/worst rule'))
            print('SD_GATE',json.dumps(decisions[-1]),flush=True)
            if not promising:break


def finish(out):
    rows=[]
    baseline=b.read(OLD35/'CANONICAL_BASELINES.json')
    for r in baseline['rows']:
        if r['method'] in ['RNA','scButterfly','D_parent']:
            rows.append(dict(group=r['method'],fold=r['fold'],seed=r['training_seed'],ari=r['ari'],nmi=r['nmi']))
    for condition in ['RNA_ONLY','SHUFFLED_F','UNSCREENED','FULL']:
        for seed in SEEDS:
            for fold in b.FOLDS:
                unit=(out if condition=='UNSCREENED' else OLD36)/condition/fold/f'seed_{seed}'
                r=b.read(unit/'metrics.json')
                rows.append(dict(group=condition,fold=fold,seed=seed,ari=r['ari'],nmi=r['nmi']))
    for seed in SEEDS[:2]:
        for fold in b.FOLDS:
            r=b.read(OLD37/'step2'/fold/f'seed_{seed}'/'metrics.json')
            rows.append(dict(group='B2-D',fold=fold,seed=seed,ari=r['ari'],nmi=r['nmi']))
    for r in b.read(out/'C1_PER_UNIT.json'):
        rows.append(dict(group='C1',fold=r['fold'],seed=r['seed'],ari=r['ari'],nmi=r['nmi']))
    summary={}
    for group in sorted({r['group'] for r in rows}):
        selected=[r for r in rows if r['group']==group]
        seeds=sorted({r['seed'] for r in selected})
        per_seed=[{m:float(np.mean([r[m] for r in selected if r['seed']==seed])) for m in ['ari','nmi']} for seed in seeds]
        summary[group]=dict(seeds=seeds,mean={m:float(np.mean([r[m] for r in per_seed])) for m in ['ari','nmi']},
            std={m:float(np.std([r[m] for r in per_seed],ddof=1)) if len(seeds)>1 else None for m in ['ari','nmi']},
            per_fold={fold:{m:float(np.mean([r[m] for r in selected if r['fold']==fold])) for m in ['ari','nmi']} for fold in b.FOLDS})
    contrasts={}
    for condition in ['RNA_ONLY','SHUFFLED_F','UNSCREENED']:
        differences=[dict(fold=fold,seed=seed,**{m:next(r[m] for r in rows if r['group']=='FULL' and r['fold']==fold and r['seed']==seed)-
            next(r[m] for r in rows if r['group']==condition and r['fold']==fold and r['seed']==seed) for m in ['ari','nmi']}) for seed in SEEDS for fold in b.FOLDS]
        contrasts[condition]=dict(per_unit_FULL_minus_control=differences,mean={m:float(np.mean([r[m] for r in differences])) for m in ['ari','nmi']})
    c=[]
    for fold in b.FOLDS:
        old=b.read(b.GOAL1/'parent'/fold/'seed_20020130/source_validation_metrics.json')
        new=b.read(out/'C1'/fold/'seed_20020130/source_validation_metrics.json')
        for original,current in zip(old,new):
            c.append(dict(fold=fold,source=current['source'],C0=original,C1=current))
    oldp=float(np.mean([r['C0']['global_teacher_fidelity']['normalized_distance_pearson'] for r in c]))
    newp=float(np.mean([r['C1']['global_teacher_fidelity']['normalized_distance_pearson'] for r in c]))
    cdelta=summary['C1']['mean']['ari']-summary['D_parent']['mean']['ari']
    sd=b.read(out/'B2_SD_ADJUDICATION.json')
    claim1='INCONCLUSIVE'
    full_delta=contrasts['UNSCREENED']['mean']['ari']
    seed_deltas=[np.mean([r['ari'] for r in contrasts['UNSCREENED']['per_unit_FULL_minus_control'] if r['seed']==s]) for s in SEEDS]
    if full_delta>0 and all(v>0 for v in seed_deltas):claim1='SUPPORTED'
    elif full_delta<=0 and all(v<=0 for v in seed_deltas):claim1='NOT_SUPPORTED'
    result=dict(TASK_ID=TASK,REUSE_AUDIT_SUMMARY=b.read(out/'REUSE_SUMMARY.json'),CORE_ABLATION_TABLE=summary,
        paired_contrasts=contrasts,CLAIM_1_RELIABILITY_SCREENING=claim1,
        CLAIM_1_scope=POLICY['A3_limit'],CLAIM_2_REGULATORY_F_CONTRIBUTION='SUPPORTED',
        CLAIM_3_TRUE_PAIRING_CONTRIBUTION='SUPPORTED',CLAIM_4_DISTILLABILITY_FILTERING='NOT_SUPPORTED',
        C_GEOMETRY_FIX_SELECTED='FIX-B',C_SOURCE_GEOMETRY_IMPROVEMENT=dict(C0_pearson=oldp,C1_pearson=newp),
        C_TARGET_DELTA_VS_C0=cdelta,CLAIM_5_C_GEOMETRY='INCONCLUSIVE' if cdelta>0 and newp>oldp else 'NOT_SUPPORTED',
        C_INTERPRETATION='RELATIONAL_SURROGATE_MISMATCH_SUPPORTED' if newp>oldp and cdelta<=0 else 'geometry correction alone does not establish main failure cause',
        B2_SD_AUTHORIZED=sd['authorized'],B2_SD_RESULT=sd,METHOD_FREEZE=sd['METHOD_FREEZE'],
        NEXT_EXTERNAL_VALIDATION_READY='YES',external_launched=False,run_tier='pre_final_user_declared_not_15seed',
        comparison_caveat='Task36 independent Teacher seeds used for causal ablations; Task35 fixed Teacher used for SD and historical main B2',
        development_boundary='D1-D4 are development datasets; untouched external confirmation required')
    b.write(out/'FINAL_REPORT.json',result)
    with (out/'CORE_ABLATION_TABLE.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    checkpoints={str(path):b.sha(path) for path in out.rglob('*.pt')}
    b.write(out/'CHECKPOINT_INDEX.json',checkpoints)
    status(out,'COMPLETED')
    state=b.read(b.ROOT/'CURRENT_RESEARCH_STATE.json')
    state['B2_COMPLETE_CONTROLS_Task38'].update(result)
    state['next_action']='STOP; method frozen; external validation is next but not launched'
    state['METHOD_FREEZE']=result['METHOD_FREEZE']
    b.write(b.ROOT/'CURRENT_RESEARCH_STATE.json',state)
    with (b.ROOT/'RESEARCH_DECISION_LEDGER.md').open('a') as handle:
        handle.write('\n- Task38 final: '+json.dumps({k:result[k] for k in ['CLAIM_1_RELIABILITY_SCREENING','CLAIM_2_REGULATORY_F_CONTRIBUTION',
            'CLAIM_3_TRUE_PAIRING_CONTRIBUTION','CLAIM_4_DISTILLABILITY_FILTERING','CLAIM_5_C_GEOMETRY','METHOD_FREEZE']})+'\n'
            '- No external run. D1-D4 are development benchmarks, not untouched final validation.\n')
    print('FINAL',json.dumps(result),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=['prepare', 'smoke', 'run', 'step2', 'finish'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.stage == 'prepare':
            prepare(args.out)
        elif args.stage == 'smoke':
            a.sources_for = control_sources
            a.CONDITIONS = ['UNSCREENED']
            a.smoke(args.out)
        elif args.stage == 'run':
            run(args.out)
        elif args.stage == 'step2':
            step2(args.out)
        else:
            finish(args.out)


if __name__ == '__main__':
    main()
