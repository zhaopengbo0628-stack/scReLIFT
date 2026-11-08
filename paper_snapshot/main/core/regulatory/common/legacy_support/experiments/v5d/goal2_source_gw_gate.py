from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import run_strict_lodo_two_step as strict

ROOT = Path('/home/u202522802045/share/zpb')
GOAL1 = ROOT / 'result_contain/strict_lodo_two_step_20261002'
CONFIG = {
    'formulation': 'unregularized square-loss Gromov-Wasserstein, uniform marginals, uniform initial coupling, conditional gradient',
    'rna_relation': 'within-modality Euclidean distances of L2-normalized frozen SOURCE RNA backbone latent',
    'regulatory_relation': 'within-modality Euclidean distances of L2-normalized existing SOURCE F_val',
    'distance_scale': 'each source validation relation divided by its off-diagonal mean',
    'sample_size_max': 512, 'max_iter': 100, 'tol_rel': 1e-9, 'tol_abs': 1e-9,
    'perturbation': 'one symmetric deterministic Gaussian relation perturbation, 0.001 times off-diagonal standard deviation',
    'seed': 20020130, 'stability_min': 0.6, 'identity_top1_random_fold_min': 5.0,
    'identity_top10_random_fold_min': 2.0, 'teacher_relation_spearman_min': 0.5,
    'teacher_top15_retention_min': 0.15, 'teacher_vs_random_gain_min': 0.1,
    'ready_units_min': 9, 'ready_sources_per_fold_min': 2,
    'topk_ties': 'fractional expected recovery under uniform tie-breaking; not arbitrary zero-mass ranking',
    'source_only': True, 'labels_used': False, 'target_data_accessed': False,
    'no_formulation_sweep': True, 'no_learned_alignment': True,
}


def relation(values):
    values = np.asarray(values, dtype=np.float64)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    distance = cdist(values, values)
    np.fill_diagonal(distance, 0)
    scale = float(distance.sum() / (len(distance) * (len(distance) - 1)))
    return distance / max(scale, 1e-12), scale


def corr(left, right):
    mask = ~np.eye(len(left), dtype=bool)
    a, b = left[mask], right[mask]
    return float(spearmanr(a, b).statistic) if a.std() > 1e-12 and b.std() > 1e-12 else 0.0


def topk_same_cell(plan, k):
    diagonal = np.diag(plan)[:, None]
    greater = np.sum(plan > diagonal + 1e-12, axis=1)
    equal = np.sum(np.abs(plan - diagonal) <= 1e-12, axis=1)
    return float(np.mean(np.clip((k - greater) / np.maximum(equal, 1), 0, 1)))


def top15_overlap(left, right):
    a, b = left.copy(), right.copy()
    np.fill_diagonal(a, -np.inf)
    np.fill_diagonal(b, -np.inf)
    k = min(15, len(left) - 1)
    ia = np.argpartition(-a, k - 1, axis=1)[:, :k]
    ib = np.argpartition(-b, k - 1, axis=1)[:, :k]
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(ia, ib)]))


def solve(ot, a, b):
    n = len(a)
    p = np.full(n, 1.0 / n)
    plan, log = ot.gromov.gromov_wasserstein(a, b, p, p, loss_fun='square_loss', symmetric=True,
                                           G0=np.outer(p, p), max_iter=100, tol_rel=1e-9,
                                           tol_abs=1e-9, log=True)
    assert np.isfinite(plan).all() and plan.min() >= -1e-12
    assert np.max(np.abs(plan.sum(0) - p)) < 1e-7
    assert np.max(np.abs(plan.sum(1) - p)) < 1e-7
    return plan, float(log['gw_dist'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(ROOT / 'goal2_dependencies'), str(ROOT / 'V5/Teacher-S10x20_Student-64D-KL/source')]
    os.environ['POT_BACKEND_DISABLE_PYTORCH'] = '1'
    os.environ['POT_BACKEND_DISABLE_JAX'] = '1'
    os.environ['POT_BACKEND_DISABLE_CUPY'] = '1'
    os.environ['POT_BACKEND_DISABLE_TENSORFLOW'] = '1'
    import ot
    from v5s.denoise import DenoisingAutoencoder
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    strict.write(args.out / 'PREDECLARED_GW_CONFIG.json', {**CONFIG, 'POT_version': ot.__version__, 'runner_sha256': strict.sha(__file__)})
    source_inventory = json.loads((GOAL1 / 'COMPLETION_MANIFEST.json').read_text())['source_files_sha256']
    records = []
    for fold, (_, source_names) in strict.parent.FOLDS.items():
        checkpoint = GOAL1 / 'rna' / fold / 'seed_20020130/shared_rna_best.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        dae = DenoisingAutoencoder(saved['feature_count'], 512, 64)
        dae.load_state_dict(saved['model_state_dict'])
        dae.eval()
        source_root = ROOT / 'result_contain' / strict.SOURCE_ROOT / fold
        for source_key in source_names:
            name = strict.parent.DATASET_NAMES[source_key]
            source = strict.RNAOnlySource(source_root / 'harmonized', name)
            rng = np.random.default_rng(20020130 + int(fold[-1]) * 100 + int(source_key[-1]))
            sample = np.sort(rng.choice(len(source.val), min(512, len(source.val)), replace=False))
            rna = strict.normalized_rows(source, source.val[sample], saved['rna_mean'], saved['rna_std'])
            with torch.no_grad():
                latent = dae.encoder(torch.from_numpy(rna)).numpy()
            fp = source_root / f'{name}_F_val.npy'
            teacher_path = ROOT / 'result_contain/23_dmode_snf_correction_localization_audit_20260927/matched_geometry' / fold / name / 'G_SNF_verified.npy'
            paths = [fp, teacher_path, source.root / 'rna.npy', source.root / 'metadata.npz']
            for p in paths:
                assert strict.sha(p) == source_inventory[str(p)], p
            f_all = np.load(fp, mmap_mode='r')
            teacher_all = np.load(teacher_path, mmap_mode='r')
            assert len(f_all) == len(source.val) and teacher_all.shape == (len(source.val), len(source.val))
            ca, scale_r = relation(latent)
            cb, scale_f = relation(f_all[sample])
            assert np.isfinite(ca).all() and np.isfinite(cb).all() and scale_r > 1e-12 and scale_f > 1e-12
            plan, cost = solve(ot, ca, cb)
            noise = rng.standard_normal(cb.shape)
            noise = (noise + noise.T) / 2
            mask = ~np.eye(len(cb), dtype=bool)
            perturbed = np.maximum(cb + 0.001 * cb[mask].std() * noise, 0)
            np.fill_diagonal(perturbed, 0)
            alt, alt_cost = solve(ot, ca, perturbed)
            n = len(plan)
            row_plan = plan * n
            teacher = np.asarray(teacher_all[np.ix_(sample, sample)], dtype=np.float64)
            transported_teacher = row_plan @ teacher @ row_plan.T
            permutation = rng.permutation(n)
            random_teacher = teacher[np.ix_(permutation, permutation)]
            stability = 1 - 0.5 * float(np.abs(plan - alt).sum())
            teacher_corr = corr(teacher, transported_teacher)
            random_corr = corr(teacher, random_teacher)
            retention = top15_overlap(teacher, transported_teacher)
            recovery1, recovery10 = topk_same_cell(plan, 1), topk_same_cell(plan, 10)
            identity = recovery1 >= 5 / n and recovery10 >= 20 / n
            structure = teacher_corr >= 0.5 and retention >= 0.15 and teacher_corr - random_corr >= 0.1
            ready = stability >= 0.6 and (identity or structure)
            positive = row_plan[row_plan > 0]
            entropy = float(-np.sum(positive * np.log(positive)) / n)
            directory = args.out / fold / name
            directory.mkdir(parents=True, exist_ok=True)
            for key, value in [('source_validation_positions', sample), ('C_R', ca), ('C_F', cb), ('coupling', plan), ('perturbed_coupling', alt)]:
                np.save(directory / f'{key}.npy', value)
            record = {'fold': fold, 'source': name, 'sample_cells': n, 'full_source_validation_cells': len(source.val), 'same_cell_top1': recovery1, 'same_cell_top10': recovery10, 'random_top1': 1 / n, 'random_top10': 10 / n, 'diagonal_transport_mass': float(np.trace(plan)), 'random_diagonal_mass': 1 / n, 'mean_row_entropy': entropy, 'normalized_row_entropy': entropy / np.log(n), 'mean_row_max_mass': float(np.mean(row_plan.max(1))), 'coupling_mass_stability': stability, 'argmax_stability': float(np.mean(plan.argmax(1) == alt.argmax(1))), 'rna_vs_transported_F_relation_spearman': corr(ca, row_plan @ cb @ row_plan.T), 'teacher_relation_spearman': teacher_corr, 'random_permutation_teacher_spearman': random_corr, 'teacher_top15_retention': retention, 'random_top15_expectation': 15 / (n - 1), 'gw_objective': cost, 'perturbed_gw_objective': alt_cost, 'source_identity_signal': identity, 'source_structural_signal': structure, 'ready': ready, 'input_sha256': {str(p): strict.sha(p) for p in paths}, 'source_backbone_sha256': strict.sha(checkpoint), 'rna_relation_scale': scale_r, 'F_relation_scale': scale_f, 'labels_used': False, 'target_data_accessed': False}
            records.append(record)
            strict.write(directory / 'metrics.json', record)
            strict.write(args.out / 'RECORDS.json', records)
            print('GW_SOURCE', fold, name, 'top1', recovery1, 'stability', stability, 'teacher_spearman', teacher_corr, 'ready', ready, flush=True)
    ready_count = sum(r['ready'] for r in records)
    ready_fold = all(sum(r['ready'] for r in records if r['fold'] == f) >= 2 for f in strict.parent.FOLDS)
    ready = ready_count >= 9 and ready_fold
    unstable = float(np.mean([r['coupling_mass_stability'] for r in records])) < 0.6
    signal = 'STRONG' if ready and sum(r['source_identity_signal'] for r in records) >= 9 else 'MODERATE' if ready else 'UNSTABLE' if unstable else 'WEAK'
    old_sanity = json.loads((GOAL1 / 'TEACHER_PAIRBANK_SANITY.json').read_text())
    result = {'GW_MATCHING_SIGNAL': signal, 'OT_BRANCH_READY': 'YES' if ready else 'NO', 'OT_SOURCE_GATE': 'PASS' if ready else 'OT_REJECTED_AT_SOURCE_GATE', 'ready_source_units': ready_count, 'total_source_units': 12, 'records': records, 'predeclared_config': CONFIG, 'pairbank_reference': {'low_frequency_subspace_similarity_mean': float(np.mean([r['cached_overlap']['leading_low_frequency_subspace_similarity_full_vs_pairbank'] for r in old_sanity['checks']])), 'top15_retention_mean': float(np.mean([r['top15_preservation_rechecked'] for r in old_sanity['checks']])), 'comparison_caveat': 'GW transport matching and sparse pair-bank spectral fidelity are different quantities, not directly interchangeable.'}, 'weak_sinkhorn_reference': 'Historical weak signature probe referenced by user; exact numeric artifact not yet located. No naive cross-modality cost is used here.', 'student_training_authorized': ready, 'target_data_accessed': False, 'labels_used': False}
    strict.write(args.out / 'GW_SOURCE_GATE_ADJUDICATION.json', result)
    print('GW_GATE', signal, result['OT_SOURCE_GATE'], 'ready_units', ready_count, flush=True)


if __name__ == '__main__':
    main()
