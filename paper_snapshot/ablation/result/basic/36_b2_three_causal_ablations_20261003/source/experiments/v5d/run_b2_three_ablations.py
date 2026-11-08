"""Paired five-seed SOURCE-only B2 causal controls; reuse immutable B2 helpers."""
import argparse
import copy
import csv
import fcntl
import io
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from threadpoolctl import threadpool_limits

import run_b2_strict_lodo as b

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v5d.dmode_teacher import fit_dmode_field
from v5d.regulatory_graph import build_candidate_graph, read_tss

CONDITIONS = ['RNA_ONLY', 'SHUFFLED_F', 'NO_CROSS', 'FULL']
SEEDS = list(range(20020130, 20020135))
BASE_CONFIG = copy.deepcopy(b.CONFIG)
ORIGINAL_LOSS = b.teacher_losses
ORIGINAL_SAVE = b.save_checkpoint


def record_status(out, status):
    registry = b.ROOT / 'result_contain/task_registry.csv'
    with registry.open('r+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        reader = csv.DictReader(handle)
        columns, rows = reader.fieldnames, list(reader)
        matches = [row for row in rows if row['task_id'] == '36']
        if matches:
            assert len(matches) == 1 and matches[0]['result_path'] == str(out)
            row = matches[0]
        else:
            used = {int(row['task_id']) for row in rows}
            for directory in [b.ROOT / 'slum', b.ROOT / 'result_contain']:
                for path in directory.iterdir():
                    match = re.match(r'(?:task)?(\d+)(?:_|\b)', path.name, re.I)
                    if match and path.resolve() != out.resolve():
                        used.add(int(match.group(1)))
            assert max(used) == 35 and 36 not in used
            row = dict.fromkeys(columns, '')
            row.update(task_id='36', experiment_name=out.name, model_version='B2-three-causal-controls',
                       result_path=str(out), created_at=b.now(), launcher='existing SSH host',
                       scheduler='sequential GPU', launch_method='durable detached runner',
                       datasets='strict SOURCE LODO D1-D4', seeds=','.join(map(str, SEEDS)))
            rows.append(row)
        row.update(status=status, updated_at=b.now(), pid=str(os.getpid()),
                   run_script_path=str(Path(__file__)), stdout_path=str(out / 'stdout.log'),
                   stderr_path=str(out / 'stdout.log'))
        if status == 'COMPLETED':
            row['completed_at'] = b.now()
        handle.seek(0)
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        handle.truncate()
    manifest = dict(status=status, task_id=36, result_path=str(out), pid=os.getpid(),
                    run_tier='pre_final', target_atac_used=False, updated_at=b.now(),
                    protocol_sha256=b.sha(out / 'protocol.json'), runner_sha256=b.sha(__file__))
    b.write(out / 'manifest.json', manifest)
    state = b.read(b.ROOT / 'CURRENT_RESEARCH_STATE.json')
    state['B2_THREE_CAUSAL_ABLATIONS'] = manifest
    state['current_max_task_id'] = max(36, state.get('current_max_task_id', 0))
    state['next_action'] = ('STOP after three-control adjudication; no new architecture' if status == 'COMPLETED'
                            else 'Complete predetermined B2 controls; do not tune on target')
    b.write(b.ROOT / 'CURRENT_RESEARCH_STATE.json', state)
    with (b.ROOT / 'RESEARCH_DECISION_LEDGER.md').open('a') as handle:
        handle.write(f'\n\n## Task36 B2 three causal controls {b.now()}\n\n'
                     f'- Status: {status}; five independent pipeline seeds30-34, four folds, pre_final.\n'
                     '- RNA_ONLY, within-source/split SHUFFLED_F, C_cross/V removal with per-gene nnz match, and fresh FULL reference.\n'
                     '- SOURCE selection only; scientific alignment failures are diagnostic outcomes, not censorship.\n'
                     '- Initial/best/latest/periodic/new-best checkpoints, optimizer/RNG, field states and permutations retained.\n')


class RNATeacher(torch.nn.Module):
    def __init__(self, full):
        super().__init__()
        self.rna_encoder = full.rna_encoder
        self.rna_decoder = full.rna_decoder

    def forward(self, rna, field):
        z = self.rna_encoder(rna)
        zero = torch.zeros_like(z)
        return z, zero, z, self.rna_decoder(z), torch.zeros_like(field)


def losses(model, rna, field):
    if isinstance(model, RNATeacher):
        reconstruction = torch.nn.functional.mse_loss(model(rna, field)[3], rna)
        return reconstruction, reconstruction.new_zeros(()), reconstruction.new_zeros(())
    return ORIGINAL_LOSS(model, rna, field)


def save_checkpoint(path, checkpoint):
    ORIGINAL_SAVE(path, checkpoint)
    if path.name == 'best.pt':
        event = path.parent / f"new_best_epoch_{checkpoint['epoch']:03d}.pt"
    elif checkpoint['epoch'] % 10 == 0:
        event = path.parent / f"epoch_{checkpoint['epoch']:03d}.pt"
    else:
        return
    if not event.exists():
        ORIGINAL_SAVE(event, checkpoint)


def configure(condition):
    b.CONFIG = dict(BASE_CONFIG, method='B2-three-ablation', condition=condition,
                    run_tier='pre_final', independent_teacher_seeds=SEEDS,
                    alignment_gate_policy='diagnostic only; no outcome-based censoring',
                    teacher_objectives=['RNA reconstruction'] if condition == 'RNA_ONLY'
                    else BASE_CONFIG['teacher_objectives'])


def no_cross_field(fold, out):
    destination = out / 'fields' / fold
    if (destination / 'manifest.json').exists():
        for path, digest in b.read(destination / 'manifest.json')['hashes'].items():
            assert b.sha(path) == digest, path
        return
    cm = b.read(b.GOAL1 / 'cache' / fold / 'manifest.json')
    root = Path(cm['source_harmonized_root']).parent
    genes = np.load(root / 'gene_ids.npy')
    peaks = np.load(root / 'canonical_peak_names.npy')
    gtf = b.ROOT.parent.parent / 'glue_156478/graph_files/gencode.v32.primary_assembly.annotation.gtf.gz'
    graph, _ = build_candidate_graph(genes, peaks, read_tss(gtf), 250000, 50)
    mask, index = graph['candidate_mask'], graph['candidate_index']
    eg = np.repeat(np.arange(len(genes)), index.shape[1])[mask.ravel()]
    ep = index[mask]
    full = sparse.load_npz(root / 'B_D.npz').tocsc()
    score = np.load(root / 'R_D.npy')
    sign = np.load(root / 'sign_D.npy')
    selected = np.load(root / 'selected_support.npy').astype(bool)
    raw = np.where(selected, sign * score, 0).astype(np.float32)
    for gene in range(len(genes)):
        positions = np.flatnonzero(eg == gene)
        denominator = np.abs(raw[positions]).sum()
        if denominator > 0:
            raw[positions] /= denominator
    nz = np.abs(raw) > 1e-8
    recovered = sparse.coo_matrix((raw[nz], (ep[nz], eg[nz])), shape=full.shape).tocsc()
    difference = recovered - full
    assert difference.nnz == 0 or np.abs(difference.data).max() < 1e-7
    with np.load(root / 'dataset_weights.npz') as weights:
        omega = weights['omega']
    within = np.stack([np.load(root / f'{name}_S10.npy') for name in cm['sources']])
    mass = (omega[:, None] * np.nan_to_num(within, nan=0)).sum(0).astype(np.float32)
    assert np.isfinite(mass).all() and len(mass) == len(eg)
    values = np.zeros_like(mass)
    for gene in range(len(genes)):
        positions = np.flatnonzero((eg == gene) & (mass > 0) & (sign != 0))
        count = full.indptr[gene + 1] - full.indptr[gene]
        assert len(positions) >= count, (fold, gene)
        chosen = positions[np.argsort(-mass[positions], kind='stable')[:count]]
        if count:
            values[chosen] = sign[chosen] * mass[chosen]
            values[chosen] /= np.abs(values[chosen]).sum()
    nz = np.abs(values) > 1e-8
    control = sparse.coo_matrix((values[nz], (ep[nz], eg[nz])), shape=full.shape).tocsc()
    assert np.array_equal(np.diff(control.indptr), np.diff(full.indptr))
    destination.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(destination / 'B_NO_CROSS.npz', control)
    np.save(destination / 'edge_gene.npy', eg)
    np.save(destination / 'edge_peak.npy', ep)
    input_paths = [gtf, root / 'B_D.npz', root / 'gene_ids.npy', root / 'canonical_peak_names.npy',
                   root / 'R_D.npy', root / 'sign_D.npy', root / 'selected_support.npy', root / 'dataset_weights.npz']
    for name in cm['sources']:
        directory = Path(cm['source_harmonized_root']) / name
        with np.load(directory / 'metadata.npz', allow_pickle=True) as metadata:
            train, val = metadata['train_index'], metadata['val_index']
        assert not np.intersect1d(train, val).size
        atac_path = directory / 'atac_binary.npz'
        atac = sparse.load_npz(atac_path)[np.concatenate([train, val])]
        field, state, diagnostics = fit_dmode_field(atac, control, np.arange(len(train)))
        np.save(destination / f'{name}_F_train.npy', field[:len(train)])
        np.save(destination / f'{name}_F_val.npy', field[len(train):])
        np.savez(destination / f'{name}_field_state.npz', **state)
        b.write(destination / f'{name}_field_diagnostics.json', diagnostics)
        input_paths += [atac_path, directory / 'metadata.npz', root / f'{name}_S10.npy']
    hashes = {str(p): b.sha(p) for p in destination.iterdir() if p.is_file()}
    hashes.update({str(p): b.sha(p) for p in input_paths})
    b.write(destination / 'manifest.json', dict(hashes=hashes, target_read=False,
            full_reconstruction_pass=True, exact_per_gene_nnz_match=True,
            full_nnz=full.nnz, control_nnz=control.nnz,
            score='M_D only; remove C_cross and V; preserve within-source S, omega and consensus signs',
            limitation='consensus signs and quality weights still aggregate sources; not a removal of all reliability'))


def sources_for(condition, fold, seed, out):
    sources, mean, std, inputs = b.load_sources(fold, out)
    if condition == 'NO_CROSS':
        directory = out / 'fields' / fold
        inputs.update(b.read(directory / 'manifest.json')['hashes'])
        for source in sources:
            for split in ['train', 'val']:
                source[split]['field'] = np.load(directory / f"{source['name']}_F_{split}.npy")
    if condition == 'SHUFFLED_F':
        directory = out / condition / fold / f'seed_{seed}' / 'permutations'
        directory.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        for source in sources:
            for split in ['train', 'val']:
                n = len(source[split]['field'])
                order = rng.permutation(n)
                permutation = np.empty(n, dtype=np.int64)
                permutation[order] = np.roll(order, 1)
                assert n > 1 and np.all(permutation != np.arange(n))
                assert np.array_equal(np.sort(permutation), np.arange(n))
                path = directory / f"{source['name']}_{split}.npy"
                if path.exists():
                    assert np.array_equal(np.load(path), permutation)
                else:
                    np.save(path, permutation)
                source[split]['field'] = source[split]['field'][permutation]
                inputs[str(path)] = b.sha(path)
    inputs[str(Path(__file__))] = b.sha(__file__)
    return sources, mean, std, inputs


def initial(model, unit, seed, teacher, inputs):
    path = unit / 'initial.pt'
    if path.exists():
        return
    unit.mkdir(parents=True, exist_ok=True)
    stage = 'teacher' if teacher else 'student'
    optimizer = torch.optim.AdamW(model.parameters(), lr=b.CONFIG[f'{stage}_lr'], weight_decay=1e-4)
    ORIGINAL_SAVE(path, dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=0,
                  rng=b.rng_state(np.random.default_rng(seed)), input_hashes=inputs, config=copy.deepcopy(b.CONFIG),
                  scope='initial weights/RNG; train_stage computes SOURCE initial scales before epoch1'))


def run_unit(condition, fold, seed, out, device):
    configure(condition)
    unit = out / condition / fold / f'seed_{seed}'
    if (unit / 'metrics.json').exists():
        result = b.read(unit / 'metrics.json')
        for path, digest in b.read(unit / 'checkpoint_index.json').items():
            assert b.sha(path) == digest
        assert b.sha(unit / 'z_target_test.npy') == result['embedding_sha256']
        return result
    sources, mean, std, inputs = sources_for(condition, fold, seed, out)
    b.seed_all(seed)
    width = sources[0]['train']['rna'].shape[1]
    teacher = b.Teacher(width, sources[0]['train']['field'].shape[1])
    if condition == 'RNA_ONLY':
        teacher = RNATeacher(teacher)
    teacher = teacher.to(device)
    initial(teacher, unit / 'teacher', seed, True, inputs)
    teacher = b.train_stage(teacher, sources, unit / 'teacher', seed, device, True, inputs)
    teacher.requires_grad_(False)
    if condition != 'RNA_ONLY':
        b.write(unit / 'teacher_alignment_audit.json', b.teacher_audit(teacher, sources, device))
    cache = unit / 'teacher_cache'
    cache.mkdir(parents=True, exist_ok=True)
    for source in sources:
        for split in ['train', 'val']:
            latent = b.forward_numpy(teacher, source[split]['rna'], device, source[split]['field'])[2]
            source[split]['teacher_latent'] = latent
            path = cache / f"{source['name']}_{split}_zT.npy"
            b.save_array(path, latent)
            inputs[str(path)] = b.sha(path)
    mu = np.mean([s['train']['teacher_latent'].mean(0) for s in sources], axis=0)
    var = np.mean([s['train']['teacher_latent'].var(0) + (s['train']['teacher_latent'].mean(0) - mu)**2 for s in sources], axis=0)
    sigma = np.sqrt(var).astype(np.float32)
    for source in sources:
        source.update(latent_mean=mu, latent_std=sigma, raw_zero_standardized=-mean/(std + 1e-6))
    b.write(cache / 'manifest.json', dict(latent_mean=mu.tolist(), latent_std=sigma.tolist(),
            fit='SOURCE train dataset-balanced', teacher_checkpoint_sha256=b.sha(unit / 'teacher/best.pt')))
    inputs[str(unit / 'teacher/best.pt')] = b.sha(unit / 'teacher/best.pt')
    b.seed_all(seed)
    student = copy.deepcopy(teacher.rna_encoder).requires_grad_(True)
    initial(student, unit / 'student', seed, False, inputs)
    student = b.train_stage(student, sources, unit / 'student', seed, device, False, inputs)
    mechanisms = []
    for source in sources:
        zr, zf, zt, _, _ = b.forward_numpy(teacher, source['val']['rna'], device, source['val']['field'])
        zs = b.forward_numpy(student, source['val']['rna'], device)
        reference = out / 'reference_probes' / fold
        original_field = np.load(reference / f"{source['name']}_F_val.npy")
        original_train = np.load(reference / f"{source['name']}_F_train.npy")
        train_z = b.forward_numpy(teacher, source['train']['rna'], device, source['train']['field'])[2]
        train_s = b.forward_numpy(student, source['train']['rna'], device)
        mechanisms.append(dict(source=source['name'], teacher_rank=b.rank_info(zt), student_rank=b.rank_info(zs),
                student_vs_teacher=b.geometry(zs, zt), student_vs_teacher_rna=b.geometry(zs, zr),
                paired_mse=float(np.mean((zr-zf)**2)) if condition != 'RNA_ONLY' else None,
                common_F_teacher_probe=b.probe(train_z, original_train, zt, original_field),
                common_F_student_probe=b.probe(train_s, original_train, zs, original_field)))
        b.save_array(unit / f"{source['name']}_source_val_student.npy", zs)
        b.save_array(unit / f"{source['name']}_source_val_teacher_rna.npy", zr)
        b.save_array(unit / f"{source['name']}_source_val_teacher_fused.npy", zt)
    b.write(unit / 'source_mechanism.json', mechanisms)
    target_path = b.GOAL1 / 'cache' / fold / 'target_rna_test.npy'
    cm = b.read(target_path.parent / 'manifest.json')
    assert b.sha(target_path) == cm['cache_sha256']['target_rna_test.npy']
    target = ((np.load(target_path) - mean) / (std + 1e-6)).astype(np.float32)
    embedding = b.forward_numpy(student, target, device)
    digest = b.save_array(unit / 'z_target_test.npy', embedding)
    b.write(unit / 'frozen_embedding_manifest.json', dict(embedding_sha256=digest,
            target_rna_sha256=b.sha(target_path), target_atac_used=False, target_fitting=False, frozen_at=b.now()))
    result = dict(condition=condition, fold=fold, seed=seed, embedding_sha256=digest,
                  **b.evaluate(embedding, b.labels_after_frozen(fold)))
    b.write(unit / 'checkpoint_index.json', {str(p): b.sha(p) for p in unit.rglob('*.pt')})
    b.write(unit / 'metrics.json', result)
    print('RESULT', condition, fold, seed, result['ari'], result['nmi'], flush=True)
    return result


def prepare(out):
    old = b.ROOT / 'result_contain/35_b2_shared_regulatory_teacher_dense_kd_20261003'
    manifest = b.read(old / 'data_manifest.json')
    for path, digest in manifest['source_files_sha256'].items():
        assert b.sha(path) == digest, path
    out.mkdir(parents=True, exist_ok=True)
    b.write(out / 'data_manifest.json', manifest)
    for fold in b.FOLDS:
        no_cross_field(fold, out)
        sources, _, _, _ = b.load_sources(fold, out)
        directory = out / 'reference_probes' / fold
        directory.mkdir(parents=True, exist_ok=True)
        for source in sources:
            for split in ['train', 'val']:
                b.save_array(directory / f"{source['name']}_F_{split}.npy", source[split]['field'])
    b.write(out / 'protocol.json', dict(conditions=CONDITIONS, seeds=SEEDS, run_tier='pre_final',
            folds=b.FOLDS, independent_teacher_and_student_per_seed=True,
            base_config=BASE_CONFIG, target_atac_used=False, target_labels='only after embedding frozen',
            screening='remove explicit C_cross and V penalties; retain within S, omega, consensus sign',
            shuffled='fixed deterministic derangement within each SOURCE and each split',
            RNA_only='remove F encoder/decoder and pair loss; matched initial RNA weights, not total parameters',
            checkpoint_policy='initial/best/latest/every10/new-best; optimizer and RNG retained',
            selection='SOURCE validation only, no target gating', historical_full_used='descriptive only',
            code_hashes={str(p): b.sha(p) for p in [Path(__file__), Path(b.__file__),
                Path(sys.modules['v5d.regulatory_graph'].__file__), Path(sys.modules['v5d.dmode_teacher'].__file__)]}))


def smoke(out):
    device = torch.device('cuda')
    for condition in CONDITIONS:
        configure(condition)
        sources, _, _, _ = sources_for(condition, b.FOLDS[0], SEEDS[0], out)
        b.seed_all(SEEDS[0])
        width = sources[0]['train']['rna'].shape[1]
        full = b.Teacher(width, width)
        model = RNATeacher(full) if condition == 'RNA_ONLY' else full
        model = model.to(device)
        batch = next(b.batches(sources, np.random.default_rng(SEEDS[0])))
        rna, field = b.batch_tensor(batch, 'rna', device), b.batch_tensor(batch, 'field', device)
        loss = sum(losses(model, rna, field))
        loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        b.seed_all(SEEDS[0])
        restored = b.Teacher(width, width)
        restored = RNATeacher(restored) if condition == 'RNA_ONLY' else restored
        restored = restored.to(device)
        assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
        student = copy.deepcopy(model.rna_encoder).requires_grad_(True)
        model.eval()
        with torch.no_grad():
            target = model(rna, field)[2]
        kd = torch.nn.functional.mse_loss(student(rna), target)
        kd.backward()
        assert torch.isfinite(kd) and all(p.grad is None or torch.isfinite(p.grad).all() for p in student.parameters())
        optimizer = torch.optim.AdamW(student.parameters(), lr=BASE_CONFIG['student_lr'], weight_decay=1e-4)
        rng = np.random.default_rng(SEEDS[0])
        checkpoint = dict(model=student.state_dict(), optimizer=optimizer.state_dict(), rng=b.rng_state(rng))
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        expected = torch.rand(5, device=device)
        expected_batch = rng.integers(100, size=5)
        buffer.seek(0)
        saved = torch.load(buffer, map_location=device, weights_only=False)
        student.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        b.restore_rng(saved['rng'], rng)
        assert torch.equal(expected, torch.rand(5, device=device))
        assert np.array_equal(expected_batch, rng.integers(100, size=5))
        print('SMOKE_PASS', condition, flush=True)
    b.write(out / 'SMOKE_GATE.json', dict(status='PASS', conditions=CONDITIONS,
             no_optimizer_step=True, source_only=True, deterministic_initialization=True,
             teacher_student_backward_finite=True, checkpoint_rng_roundtrip=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--stage', required=True, choices=['prepare', 'smoke', 'run'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    b.teacher_losses, b.save_checkpoint = losses, save_checkpoint
    with threadpool_limits(limits=2):
        if args.stage == 'prepare':
            prepare(args.out)
        elif args.stage == 'smoke':
            smoke(args.out)
        else:
            assert b.read(args.out / 'SMOKE_GATE.json')['status'] == 'PASS'
            with (b.ROOT / '.b2_gpu_training.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                for path, digest in b.read(args.out / 'protocol.json')['code_hashes'].items():
                    assert b.sha(path) == digest, path
                record_status(args.out, 'RUNNING')
                rows = []
                for condition in CONDITIONS:
                    for seed in SEEDS:
                        for fold in b.FOLDS:
                            b.write(args.out / 'progress.json', dict(status='RUNNING', condition=condition, seed=seed,
                                    fold=fold, completed_units=len(rows), total_units=80, updated_at=b.now()))
                            rows.append(run_unit(condition, fold, seed, args.out, torch.device('cuda')))
                            b.write(args.out / 'per_unit_metrics.json', rows)
                            torch.cuda.empty_cache()
                summaries = {}
                for condition in CONDITIONS:
                    means = [{m: float(np.mean([r[m] for r in rows if r['condition'] == condition and r['seed'] == seed]))
                              for m in ['ari', 'nmi']} for seed in SEEDS]
                    summaries[condition] = {m: dict(mean=float(np.mean([v[m] for v in means])),
                            std=float(np.std([v[m] for v in means], ddof=1))) for m in ['ari', 'nmi']}
                contrasts = {}
                for condition in CONDITIONS[:-1]:
                    differences = []
                    for seed in SEEDS:
                        delta = {}
                        for metric in ['ari', 'nmi']:
                            full = [r[metric] for r in rows if r['condition'] == 'FULL' and r['seed'] == seed]
                            control = [r[metric] for r in rows if r['condition'] == condition and r['seed'] == seed]
                            delta[metric] = float(np.mean(full) - np.mean(control))
                        differences.append(dict(seed=seed, **delta))
                    contrasts[condition] = dict(per_seed_FULL_minus_control=differences,
                        summary={m: dict(mean=float(np.mean([d[m] for d in differences])),
                                    std=float(np.std([d[m] for d in differences], ddof=1))) for m in ['ari', 'nmi']})
                b.write(args.out / 'summary.json', dict(status='COMPLETE', run_tier='pre_final',
                        summaries=summaries, paired_contrasts=contrasts))
                b.write(args.out / 'progress.json', dict(status='COMPLETE', completed_units=80, total_units=80))
                record_status(args.out, 'COMPLETED')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if '--out' in sys.argv and '--stage' in sys.argv and sys.argv[sys.argv.index('--stage') + 1] == 'run':
            record_status(Path(sys.argv[sys.argv.index('--out') + 1]), 'FAILED')
        raise
