from __future__ import annotations

import argparse
import csv
import fcntl
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import subspace_angles
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, adjusted_mutual_info_score, adjusted_rand_score,
                             balanced_accuracy_score, f1_score, normalized_mutual_info_score)
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import run_goal2_strict_d_branches as branches
import run_goal2_strict_scbutterfly as scb

strict, ROOT, GOAL1, PREP = branches.strict, branches.ROOT, branches.GOAL1, scb.PREP


def table(path, rows):
    strict.write(path.with_suffix('.json'), rows)
    with path.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    columns = list(rows[0])
    text = '| ' + ' | '.join(columns) + ' |\n| ' + ' | '.join(['---'] * len(columns)) + ' |\n'
    text += ''.join('| ' + ' | '.join(str(row[k]) for k in columns) + ' |\n' for row in rows)
    path.with_suffix('.md').write_text(text)


def target_labels(fold):
    cm = branches.read(GOAL1 / 'cache' / fold / 'manifest.json')
    raw = Path(cm['raw_target_source'])
    processed = raw.parent if raw.suffix == '.npz' else raw.parent.parent / 'processed'
    test = np.load(GOAL1 / 'cache' / fold / 'target_test_index.npy')
    with (processed / 'label.csv').open(newline='') as f:
        return np.asarray([r['cell_type'] for r in csv.DictReader(f)])[test]


def annotation(x, labels, seed):
    _, labels = np.unique(labels, return_inverse=True)
    k = len(np.unique(labels))
    clusters = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(StandardScaler().fit_transform(x))
    counts = np.zeros((k, k), dtype=np.int64)
    np.add.at(counts, (labels, clusters), 1)
    a, b = linear_sum_assignment(-counts)
    mapping = dict(zip(b, a))
    matched = np.asarray([mapping[c] for c in clusters])
    return {'ARI': float(adjusted_rand_score(labels, clusters)), 'NMI': float(normalized_mutual_info_score(labels, clusters)),
            'AMI': float(adjusted_mutual_info_score(labels, clusters)), 'Cluster ACC': float(accuracy_score(labels, matched)),
            'Accuracy': float(accuracy_score(labels, matched)), 'Balanced Accuracy': float(balanced_accuracy_score(labels, matched)),
            'Macro-F1': float(f1_score(labels, matched, average='macro')), 'Weighted-F1': float(f1_score(labels, matched, average='weighted')),
            'annotation_scope': 'posthoc target Hungarian-mapped clusters, NOT source-supervised label transfer'}


def topology(x, labels, reference, seed):
    from sklearn.preprocessing import StandardScaler
    x = StandardScaler().fit_transform(np.asarray(x, dtype=np.float32))
    n = len(x)
    k = min(15, n - 1)
    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x, return_distance=False)
    neighbors = np.asarray([ids[ids != i][:k] for i, ids in enumerate(neighbors)])
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, n, size=(2, 4096))
    keep = i != j
    i, j = i[keep], j[keep]
    d = np.linalg.norm(x[i] - x[j], axis=1)
    r = np.linalg.norm(reference[i] - reference[j], axis=1)
    same = labels[i] == labels[j]
    same_d = float(d[same].mean()) if same.any() else None
    diff_d = float(d[~same].mean()) if (~same).any() else None
    nondegenerate = np.std(d) > 1e-12 and np.std(r) > 1e-12
    return {'kNN_purity_at_15': float(np.mean(labels[neighbors] == labels[:, None])),
            'geometry_Pearson_vs_frozen_RNA': float(pearsonr(d, r).statistic) if nondegenerate else None,
            'geometry_Spearman_vs_frozen_RNA': float(spearmanr(d, r).statistic) if nondegenerate else None,
            'same_type_distance': same_d, 'different_type_distance': diff_d,
            'separation_ratio': diff_d / same_d if same_d and diff_d is not None else None,
            'geometry_scope': '4096 deterministic pairs; StandardScaler on frozen target output; RNA reference seed30; not ATAC ground truth',
            'ATAC_ground_truth_topology': 'NA', 'target_atac_used': False}


def capacity_extras(directory):
    sys.path.insert(0, str(ROOT / 'V5/Teacher-S10x20_Student-64D-KL/source'))
    from v5d.evaluate_teacher import evaluate_geometry
    for fold in strict.parent.FOLDS:
        unit = directory / fold / 'seed_20020130'
        if (unit / 'supplemental_diagnostics.json').exists():
            continue
        data = branches.load_fold(fold)
        sources, backbone, mean, std = data[:4]
        rho = branches.read(PREP / 'CAPACITY_SCALE_DERIVATION.json')['rho']
        model = branches.RNAScaffoldVectorNormResidualEncoder(backbone, 64, 128, rho)
        model.load_state_dict(torch.load(unit / 'best_by_topology.pt', map_location='cpu', weights_only=False)['model_state_dict'])
        model.eval()
        spectral = []
        for source in sources:
            with torch.no_grad():
                z = np.concatenate([model(torch.from_numpy(strict.normalized_rows(source, source.val[s:s+256], mean, std))).numpy()
                                    for s in range(0, len(source.val), 256)])
            graph_path = ROOT / 'result_contain/23_dmode_snf_correction_localization_audit_20260927/matched_geometry' / fold / source.name / 'G_SNF_verified.npy'
            _, coordinates = strict.parent._spectral_coordinates(np.load(graph_path))
            angles = subspace_angles(z - z.mean(0), coordinates - coordinates.mean(0))
            spectral.append({'source': source.name, 'centered_Student_vs_FullTeacher_coordinate_subspace_cosine': float(np.mean(np.cos(angles))),
                             'scope': 'linear coordinate subspace similarity, not an eigenspectrum of Student kNN graph'})
        frozen = np.load(unit / 'z_target_test.npy')
        labels = target_labels(fold)
        reference = np.load(GOAL1 / 'rna' / fold / 'seed_20020130/z_target_test.npy')
        metrics = evaluate_geometry(frozen, reference, labels, 20020130)
        strict.write(unit / 'annotation_metrics.json', {**metrics, 'scope': 'posthoc Hungarian-mapped target clusters; not supervised label transfer'})
        diag = branches.read(unit / 'diagnostics.json')
        gradients = diag['gradient_records']
        summary = {'source_spectral_fidelity': spectral, 'parameter_updates': False,
                   'gradient_cosine': strict.distribution([g['cosine'] for g in gradients]),
                   'fraction_cosine_negative': float(np.mean([g['cosine'] < 0 for g in gradients])),
                   'fraction_cosine_below_minus_0_2': float(np.mean([g['cosine'] < -0.2 for g in gradients])),
                   'normalized_global_over_local_gradient_norm_ratio': strict.distribution([g['normalized_global_over_local_norm'] for g in gradients]),
                   'target_topology': topology(frozen, labels, reference, 20020130)}
        strict.write(unit / 'supplemental_diagnostics.json', summary)
        print('CAPACITY_EXTRAS', fold, flush=True)


def finalize():
    assignments = branches.read(PREP / 'TASK_ASSIGNMENTS.json')
    capacity, butterfly = [Path(assignments[k]['result_path']) for k in ['capacity', 'scbutterfly']]
    assert len(branches.read(butterfly / 'RESULTS.json')) == 12
    source_inventory = branches.read(GOAL1 / 'COMPLETION_MANIFEST.json')['source_files_sha256']
    for path, expected in source_inventory.items():
        assert strict.sha(path) == expected, path
    upstream = branches.read(PREP / 'SCBUTTERFLY_SOURCE_PROVENANCE.json')['upstream_source_sha256']
    for name, expected in upstream.items():
        assert strict.sha(ROOT / 'goal2_dependencies/scButterfly' / name) == expected, name
    for fold in strict.parent.FOLDS:
        strict.verify_cache(PREP / 'scbutterfly_cache' / fold)
    capacity_extras(capacity)
    records, efficiency, graph_rows, annotation_rows = [], [], [], []
    families = [('STRICT RNA', GOAL1 / 'rna', strict.SEEDS, 'target_metrics.json'),
                ('STRICT D PARENT', GOAL1 / 'parent', [20020130], 'target_metrics.json'),
                ('D + CAPACITY', capacity, [20020130], 'metrics.json'),
                ('scButterfly-B', butterfly, strict.SEEDS, 'metrics.json')]
    for method, root, seeds, metric_name in families:
        for fold in strict.parent.FOLDS:
            reference = np.load(GOAL1 / 'rna' / fold / 'seed_20020130/z_target_test.npy')
            for seed in seeds:
                unit = root / fold / f'seed_{seed}'
                manifest = branches.read(unit / 'manifest.json')
                metric = branches.read(unit / metric_name)
                records.append({'method': method, 'protocol': 'strict source-only fitting/frozen target RNA inference; inherited fixed cohort',
                                'fold': fold, 'seed': seed, 'ARI': metric['ari'], 'NMI': metric['nmi']})
                labels = target_labels(fold)
                if method == 'scButterfly-B':
                    scores = dict(zip(['ARI', 'NMI', 'AMI', 'Cluster ACC', 'Accuracy', 'Balanced Accuracy', 'Macro-F1', 'Weighted-F1'],
                                      [metric[k] for k in ['ari', 'nmi', 'ami', 'cluster_acc', 'accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1']]))
                    scores['annotation_scope'] = metric['annotation_score_scope']
                else:
                    scores = annotation(np.load(unit / 'z_target_test.npy'), labels, seed)
                    assert np.isclose(scores['ARI'], metric['ari']) and np.isclose(scores['NMI'], metric['nmi'])
                annotation_rows.append({'method': method, 'fold': fold, 'seed': seed, **scores})
                if method == 'D + CAPACITY':
                    strict.write(unit / 'annotation_metrics.json', scores)
                resources = manifest['resources']
                efficiency.append({'method': method, 'fold': fold, 'seed': seed,
                    'total_wall_seconds': resources.get('wall_seconds', resources.get('total_wall_seconds')),
                    'trainable_parameters': resources['trainable_parameters'],
                    'peak_GPU_allocated_bytes': resources.get('peak_allocated_bytes', resources.get('peak_gpu_allocated_bytes')),
                    'inference_seconds': resources['inference_seconds'],
                    'cells_per_second': resources.get('inference_cells_per_second', resources.get('cells_per_second')),
                    'GPU': resources.get('gpu', resources.get('GPU'))})
                if method == 'scButterfly-B':
                    prediction = np.load(unit / 'predicted_atac.npy', mmap_mode='r')
                    labels = target_labels(fold)
                    strict.write(unit / 'topology_metrics.json', topology(prediction, labels, reference, seed))
                    assert (unit / 'latest.pt').exists() and (unit / 'best_all.pt').exists()
                elif method == 'D + CAPACITY':
                    assert (unit / 'latest.pt').exists() and (unit / 'best_by_topology.pt').exists()
                if method in ['D + CAPACITY', 'scButterfly-B']:
                    manifest['status'] = 'COMPLETE'
                    manifest['metrics_sha256'] = strict.sha(unit / metric_name)
                    manifest['checkpoint_files_sha256'] = {str(p.relative_to(unit)): strict.sha(p) for p in unit.rglob('*.pt')}
                    strict.write(unit / 'manifest.json', manifest)
        if method in ['STRICT D PARENT', 'D + CAPACITY']:
            for fold in strict.parent.FOLDS:
                srcroot = ROOT / 'result_contain' / strict.SOURCE_ROOT / fold
                original = branches.read(srcroot / 'fold_manifest.json')
                from scipy import sparse
                b = sparse.load_npz(srcroot / 'B_D.npz')
                graph_rows.append({'method': method, 'fold': fold, 'candidate_genes': b.shape[1],
                                   'candidate_peaks': b.shape[0], 'candidate_edges': original['selection']['candidate_edge_count'],
                                   'retained_genes': b.shape[1], 'retained_edges': b.nnz, 'source_artifacts_rebuilt': False,
                                   'source_weights_sha256': strict.sha(srcroot / 'dataset_weights.npz'),
                                   'regulatory_graph_record': 'hash-preserved original source B_D/F/Teacher, no new edge selection',
                                   'selector_time_seconds_current_run': 0, 'Teacher_geometry_time_seconds_current_run': 0})
    summary = []
    for method in [x[0] for x in families]:
        group = [r for r in records if r['method'] == method]
        seeds = sorted(set(r['seed'] for r in group))
        seed_means = {m: [np.mean([r[m] for r in group if r['seed'] == s]) for s in seeds] for m in ['ARI', 'NMI']}
        row = {'Method': method, 'Protocol': 'source-only LODO, fixed cohort', 'Seeds': len(seeds)}
        for fold in strict.parent.FOLDS:
            values = [r for r in group if r['fold'] == fold]
            row[fold.replace('target_', '') + ' ARI/NMI'] = f"{np.mean([r['ARI'] for r in values]):.6f}/{np.mean([r['NMI'] for r in values]):.6f}"
        row['Mean ARI/NMI'] = f"{np.mean(seed_means['ARI']):.6f}/{np.mean(seed_means['NMI']):.6f}"
        row['Std ARI/NMI'] = f"{np.std(seed_means['ARI'], ddof=1):.6f}/{np.std(seed_means['NMI'], ddof=1):.6f}" if len(seeds) > 1 else 'NA (one seed)'
        summary.append(row)
    rejected = {k: 'NA' for k in summary[0]}
    rejected.update(Method='D + OT', Protocol='REJECTED_AT_SOURCE_GATE', Seeds=0)
    summary.insert(3, rejected)
    table(PREP / 'FINAL_COMPARISON', summary)
    table(PREP / 'PER_SEED', records)
    table(PREP / 'EFFICIENCY', efficiency)
    table(PREP / 'REGULATORY_GRAPH', graph_rows)
    table(PREP / 'ANNOTATION_METRICS', annotation_rows)
    decision = {'status': 'COMPUTATION_COMPLETE', 'timestamp': datetime.now(timezone.utc).isoformat(),
                'CAPACITY_BOTTLENECK_REPRODUCED': 'YES', 'CAPACITY_MECHANISM_CONFIRMED': 'NO',
                'capacity_claim_scope': 'tested source-calibrated parameterization did not improve global fit; not disproof of every capacity intervention',
                'OT_SOURCE_GATE': 'OT_REJECTED_AT_SOURCE_GATE', 'OT_TRANSFER_MECHANISM_CONFIRMED': 'REJECTED_AT_GATE',
                'SCBUTTERFLY_STRICT_ZERO_SHOT_VALID': 'YES', 'strict_scope': scb.CONFIG['cohort_caveat'],
                'SCBUTTERFLY_DEFAULT_DEVIATIONS': scb.CONFIG['default_deviations'],
                'source_provenance_hashes_verified': len(source_inventory), 'upstream_source_hashes_verified': len(upstream),
                'BEST_CURRENT_STRICT_D_MODEL': 'STRICT D PARENT',
                'NEXT_RECOMMENDED_SCIENTIFIC_STEP': 'zero-training source loss/topology and output-gradient diagnostic; do not combine capacity+OT or launch further training automatically',
                'std_scope': 'sample std across seed-specific four-fold means; one-seed std unavailable',
                'summary': summary, 'run_tier': 'pre_final', 'combined_model_created': False}
    strict.write(PREP / 'FINAL_GOAL2_ADJUDICATION.json', decision)
    registry = ROOT / 'result_contain/task_registry.csv'
    with (ROOT / 'result_contain/task_registry.csv.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with registry.open(newline='') as f:
            reader = csv.DictReader(f)
            columns, rows = reader.fieldnames, list(reader)
        for row in rows:
            if row['task_id'] == str(assignments['capacity']['task_id']):
                row.update(status='COMPLETED_REJECT', updated_at=decision['timestamp'])
            if row['task_id'] == str(assignments['scbutterfly']['task_id']):
                row.update(status='COMPLETED', seeds='20020130,20020131,20020132', model_version='scButterfly-B-0.0.9', updated_at=decision['timestamp'])
        temp = registry.with_suffix('.csv.goal2.tmp')
        with temp.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(registry)
    state = branches.read(ROOT / 'CURRENT_RESEARCH_STATE.json')
    state['goal2'].update(status='COMPUTATION_COMPLETE_ARCHIVE_PENDING', adjudication=decision)
    state['goal2']['step2']['status'] = 'COMPUTATION_COMPLETE'
    state['next_action'] = 'STOP after Goal2. No combined capacity+OT model, no additional scientific Task or training.'
    strict.write(ROOT / 'CURRENT_RESEARCH_STATE.json', state)
    with (ROOT / 'RESEARCH_DECISION_LEDGER.md').open('a') as f:
        f.write('\n\n## Goal2 strict benchmark completed\n\n' + (PREP / 'FINAL_COMPARISON.md').read_text())
        f.write('\n- All rows are user-declared pre-final seeds, not final 15-seed claims. Std is over seed-specific four-fold means; one-seed std is NA.\n'
                '- Capacity mechanism not confirmed by the tested source-calibrated intervention; global fit did not improve, no additional D seeds. OT rejected at source gate; no OT Student. Strict D parent remains selected by the preregistered source-and-target gate.\n'
                '- scButterfly-B uses source-only merged peaks/preprocessing, no target ATAC or target statistics before frozen prediction; target labels only posthoc. Historical fixed cohort and split caveat retained. Default schedule/architecture retained; independent best SOURCE-validation weights preserved against upstream final overwrite.\n'
                '- STOP. Recommended next scientific step is a zero-training source objective/topology-gradient diagnostic, not automatic combined capacity+OT or further training.\n')
    print('GOAL2_COMPUTATION_COMPLETE', json.dumps(decision), flush=True)


def archive_audit():
    assignments = branches.read(PREP / 'TASK_ASSIGNMENTS.json')
    decision = branches.read(PREP / 'FINAL_GOAL2_ADJUDICATION.json')
    assert decision['status'] == 'COMPUTATION_COMPLETE'
    audited = []
    for family, seeds, best, output in [('capacity', [20020130], 'best_by_topology.pt', 'z_target_test.npy'),
                                      ('scbutterfly', strict.SEEDS, 'best_all.pt', 'predicted_atac.npy')]:
        directory = Path(assignments[family]['result_path'])
        for fold in strict.parent.FOLDS:
            for seed in seeds:
                unit = directory / fold / f'seed_{seed}'
                manifest = branches.read(unit / 'manifest.json')
                assert manifest['status'] == 'COMPLETE'
                for name in ['config.json', 'data_manifest.json', 'environment.txt', 'metrics.json', 'history.json', best, 'latest.pt', output]:
                    assert (unit / name).is_file(), unit / name
                config_hash = strict.sha(unit / 'config.json')
                for name in [best, 'latest.pt']:
                    state = torch.load(unit / name, map_location='cpu', weights_only=False)
                    assert state['config_hash'] == config_hash and state['epoch'] > 0
                    assert state['optimizer_state_dict'] and state['rng_state']
                    assert {'python', 'numpy', 'torch_cpu', 'torch_cuda'} <= set(state['rng_state'])
                    assert 'scheduler_state_dict' in state
                    del state
                expected = manifest.get('prediction_sha256', manifest.get('embedding_sha256'))
                assert strict.sha(unit / output) == expected
                array = np.load(unit / output, mmap_mode='r')
                for start in range(0, len(array), 256):
                    assert np.isfinite(array[start:start + 256]).all()
                if family == 'scbutterfly':
                    metrics = branches.read(unit / 'metrics.json')
                    assert metrics['target_atac_used'] is False and metrics['labels_loaded_after_frozen_prediction']
                    assert (unit / 'topology_metrics.json').is_file()
                else:
                    assert (unit / 'supplemental_diagnostics.json').is_file()
                    assert len(branches.read(unit / 'annotation_metrics.json')) >= 8
                audited.append({'family': family, 'fold': fold, 'seed': seed, 'status': 'PASS'})
    assert len(audited) == 16
    assert len(branches.read(PREP / 'ANNOTATION_METRICS.json')) == 32
    roots = [PREP] + [Path(assignments[k]['result_path']) for k in ['capacity', 'scbutterfly']]
    for directory in roots:
        source = directory / 'source'
        source.mkdir(exist_ok=True)
        for name in ['run_strict_lodo_two_step.py', 'run_goal2_strict_d_branches.py',
                     'goal2_source_gw_gate.py', 'run_goal2_strict_scbutterfly.py', 'finalize_goal2_strict.py', 'GOAL2_STRICT_PROTOCOL.md']:
            shutil.copy2(HERE / name, source / name)
        if directory != PREP:
            shutil.copy2(PREP / 'README.md', directory / 'README.md')
        inventory = {str(p.relative_to(directory)): strict.sha(p) for p in sorted(directory.rglob('*'))
                     if p.is_file() and p.name != 'COMPLETION_MANIFEST.json'}
        strict.write(directory / 'COMPLETION_MANIFEST.json', {
            'status': 'SERVER_AUDIT_COMPLETE_LOCAL_MIRROR_PENDING', 'run_tier': 'pre_final',
            'timestamp': datetime.now(timezone.utc).isoformat(), 'files_sha256': inventory,
            'audited_units': audited, 'source_code_provenance': 'exact archived runner bytes; upstream commit in source provenance',
            'scheduler_note': 'scheduler_state_dict=None means no scheduler used, not missing state',
            'strict_scope': decision['strict_scope'], 'target_atac_used_for_new_inference': False})
        print('ARCHIVE_AUDIT_PASS', directory.name, len(inventory), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['capacity-extras', 'finalize', 'archive-audit'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == 'capacity-extras':
        capacity_extras(Path(branches.read(PREP / 'TASK_ASSIGNMENTS.json')['capacity']['result_path']))
    elif args.stage == 'finalize':
        finalize()
    else:
        archive_audit()


if __name__ == '__main__':
    main()
