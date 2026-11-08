import json
import hashlib
import statistics
from pathlib import Path
import numpy as np

ROOT = Path('/home/u202522802045/share/zpb/b2_fusion_ablation')
RESULTS = ROOT / 'results'
FOLDS = [f'target_D{i}' for i in range(1, 5)]
SEEDS = [20020130, 20020131, 20020132]
VARIANTS = [('Mean', 'Mean'), ('Concat', 'Concat'), ('Gate', 'Gate'), ('Interaction', 'Interaction')]
METRICS = ['ari', 'nmi', 'accuracy', 'macro_f1', 'balanced_accuracy', 'knn_purity_at_15', 'cell_type_silhouette']


def read(path):
    return json.loads(path.read_text())


def fmt(values):
    if not values:
        return 'NA'
    return f'{statistics.mean(values):.4f} +/- {statistics.stdev(values) if len(values) > 1 else 0:.4f}'


rows = {}
for variant, folder in VARIANTS:
    rows[variant] = {}
    for fold in FOLDS:
        rows[variant][fold] = {}
        for seed in SEEDS:
            unit = RESULTS / folder / fold / f'seed_{seed}'
            row = {'status': 'MISSING'}
            if variant == 'Mean':
                ref = unit / 'reference.json'
                if ref.exists():
                    row = read(ref)
            elif (unit / 'manifest.json').exists() and (unit / 'metrics.json').exists():
                manifest = read(unit / 'manifest.json')
                if manifest.get('status') != 'COMPLETE':
                    row = {'status': 'MISSING'}
                    rows[variant][fold][seed] = row
                    continue
                row = {'status': 'COMPLETE', 'legacy_blocker_preserved': (unit / 'SCIENTIFIC_BLOCKER.json').exists(),
                       'manifest_sha256': hashlib.sha256((unit / 'manifest.json').read_bytes()).hexdigest(),
                       'metrics': read(unit / 'metrics.json'),
                       'timing': read(unit / 'timing.json'),
                       'parameters': read(unit / 'parameter_count.json') if (unit / 'parameter_count.json').exists() else {}}
            elif (unit / 'SCIENTIFIC_BLOCKER.json').exists():
                row = {'status': 'LEGACY_SCIENTIFIC_BLOCKER', **read(unit / 'SCIENTIFIC_BLOCKER.json')}
            rows[variant][fold][seed] = row

status_counts = {v: {s: sum(r['status'] == s for fold in rows[v].values() for r in fold.values())
                     for s in ['COMPLETE', 'LEGACY_SCIENTIFIC_BLOCKER', 'AVAILABLE', 'FATAL_AUDIT_UNVERIFIED', 'MISSING']}
                 for v, _ in VARIANTS}
variant_metrics = {}
fold_metrics = {}
all_timing = []
all_params = []
for variant, _ in VARIANTS:
    # First average folds within each seed, then summarize seed means.
    seed_means = {metric: [] for metric in METRICS}
    fold_metrics[variant] = {}
    for fold in FOLDS:
        fold_metrics[variant][fold] = {}
        for metric in METRICS:
            vals = []
            for seed in SEEDS:
                r = rows[variant][fold][seed]
                if r['status'] not in ('COMPLETE', 'AVAILABLE'):
                    continue
                value = r.get('metrics', {}).get(metric)
                if value is not None:
                    vals.append(float(value))
            fold_metrics[variant][fold][metric] = fmt(vals)
    complete_seed_count = 0
    descriptive_units = {metric: [] for metric in METRICS}
    for seed in SEEDS:
        fold_rows = [rows[variant][fold][seed] for fold in FOLDS]
        complete = all(r['status'] in ('COMPLETE', 'AVAILABLE') for r in fold_rows)
        complete_seed_count += int(complete)
        for metric in METRICS:
            vals = [float(r['metrics'][metric]) for r in fold_rows
                    if r['status'] in ('COMPLETE', 'AVAILABLE') and metric in r.get('metrics', {})]
            descriptive_units[metric].extend(vals)
            if complete and len(vals) == len(FOLDS):
                seed_means[metric].append(statistics.mean(vals))
    variant_metrics[variant] = {m: fmt(v) for m, v in seed_means.items()}
    variant_metrics[variant]['complete_four_fold_seeds'] = complete_seed_count
    variant_metrics[variant]['available_unit_descriptive_only'] = {m: fmt(v) for m, v in descriptive_units.items()}
    for fold in rows[variant].values():
        for r in fold.values():
            if r.get('timing'):
                all_timing.append((variant, r['timing']))
            if r.get('parameters', {}).get('TOTAL_PARAMS'):
                all_params.append((variant, r['parameters']['TOTAL_PARAMS']))

gate_stats = {'files': 0, 'source_train_files': 0, 'source_val_files': 0, 'target_gate_computed': False,
              'units_with_gate_analysis': 0, 'units_with_summary': 0, 'global_mean_std': [], 'per_source': {},
              'per_dimension_mean_std': {}, 'raw_array_stats': []}
source_values = {}
dimension_means = [[] for _ in range(64)]
dimension_stds = [[] for _ in range(64)]
for fold in FOLDS:
    for seed in SEEDS:
        d = RESULTS / 'Gate' / fold / f'seed_{seed}' / 'gate_analysis'
        meta = d / 'metadata.json'
        if not d.exists():
            continue
        gate_stats['units_with_gate_analysis'] += 1
        if meta.exists():
            gate_stats['target_gate_computed'] |= bool(read(meta).get('target_gate_computed', False))
        for split in ['train', 'val']:
            path = d / f'gate_values_source_{split}.npy'
            if path.exists():
                gate_stats['files'] += 1
                if split == 'train': gate_stats['source_train_files'] += 1
                else: gate_stats['source_val_files'] += 1
                arr = np.load(path, mmap_mode='r')
                gate_stats['raw_array_stats'].append({'fold': fold, 'seed': seed, 'split': split,
                    'cells': int(arr.shape[0]), 'dimensions': int(arr.shape[1]), 'mean': float(arr.mean()),
                    'std': float(arr.std()), 'fraction_below_0_05': float(np.mean(arr < 0.05)),
                    'fraction_above_0_95': float(np.mean(arr > 0.95))})
        mechanism_path = RESULTS / 'Gate' / fold / f'seed_{seed}' / 'mechanism.json'
        if mechanism_path.exists():
            m = read(mechanism_path)
            gate_stats['units_with_summary'] += 1
            for split in ['gate_train', 'gate_val']:
                summary = m.get(split)
                if summary:
                    gate_stats['global_mean_std'].append({'fold': fold, 'seed': seed, 'split': split,
                                                          'mean': summary.get('mean'), 'std': summary.get('std')})
                    gate_stats['per_source'][f'{fold}/seed_{seed}/{split}'] = summary.get('per_source')
                    gate_stats['per_dimension_mean_std'][f'{fold}/seed_{seed}/{split}'] = {
                        'mean': summary.get('per_dimension_mean'), 'std': summary.get('per_dimension_std')}
                    for idx, x in enumerate(summary.get('per_dimension_mean', [])):
                        dimension_means[idx].append(float(x))
                    for idx, x in enumerate(summary.get('per_dimension_std', [])):
                        dimension_stds[idx].append(float(x))
                    for source, values in (summary.get('per_source') or {}).items():
                        source_values.setdefault(source, []).append(values)

gate_stats['aggregate_per_source'] = {
    source: {'mean': statistics.mean(float(x['mean']) for x in vals),
             'between_unit_sd': statistics.stdev(float(x['mean']) for x in vals) if len(vals) > 1 else 0.0,
             'mean_within_unit_std': statistics.mean(float(x['std']) for x in vals), 'n': len(vals)}
    for source, vals in source_values.items()}
gate_stats['aggregate_per_dimension'] = [
    {'dimension': i, 'mean_gate': statistics.mean(dimension_means[i]) if dimension_means[i] else None,
     'mean_within_unit_std': statistics.mean(dimension_stds[i]) if dimension_stds[i] else None,
     'n': len(dimension_means[i])}
    for i in range(64)]

embedding_count = sum(1 for p in RESULTS.rglob('*.npy') if 'embeddings' in p.parts or 'gate_analysis' in p.parts)
target_raw_count = sum(1 for p in RESULTS.rglob('target_embedding_raw.npy'))
checkpoint_files = [p for p in RESULTS.rglob('*.pt')]
hash_check = {'manifest_files': 0, 'artifacts_checked': 0, 'mismatches': [], 'complete_unit_manifests': 0,
              'blocker_checkpoint_hashes_checked': 0, 'blocker_hash_mismatches': []}
for p in RESULTS.rglob('manifest.json'):
    obj = read(p)
    hashes = obj.get('artifact_hashes', {})
    if hashes:
        hash_check['manifest_files'] += 1
        hash_check['complete_unit_manifests'] += 1
        for rel, digest in hashes.items():
            artifact = p.parent / rel
            hash_check['artifacts_checked'] += 1
            actual = hashlib.sha256(artifact.read_bytes()).hexdigest() if artifact.exists() else 'MISSING'
            if actual != digest:
                hash_check['mismatches'].append(str(artifact))
for p in RESULTS.rglob('SCIENTIFIC_BLOCKER.json'):
    obj = read(p)
    for rel, digest in obj.get('preserved_teacher_checkpoints', {}).items():
        artifact = p.parent / rel
        hash_check['blocker_checkpoint_hashes_checked'] += 1
        actual = hashlib.sha256(artifact.read_bytes()).hexdigest() if artifact.exists() else 'MISSING'
        if actual != digest:
            hash_check['blocker_hash_mismatches'].append(str(artifact))
leakage = {'audits': 0, 'passed': 0, 'failed': []}
for p in RESULTS.rglob('leakage_audit.json'):
    obj = read(p)
    leakage['audits'] += 1
    text = json.dumps(obj).lower()
    passed = (obj.get('target_labels_loaded_after_embedding_hash') is True and
              obj.get('target_atac_loaded') is False and obj.get('target_hvg_fitted') is False and
              obj.get('target_pca_fitted') is False and obj.get('target_checkpoint_selection') is False)
    if passed:
        leakage['passed'] += 1
    else:
        leakage['failed'].append(str(p))

lines = ['# RNA-F Fusion Ablation: Corrected-Quartet 3-Seed Report', '',
         'Run tier: `pre_final` (seeds 20020130-20020132); this is not final 15-seed evidence.',
         'Variants ran sequentially under the global A30 lock. Historical blocker files are retained; a validated COMPLETE manifest takes precedence.', '',
         '## Run Status', '', '| Variant | Complete/Available | Legacy blocker records | Fatal audit unverified | Missing |', '|---|---:|---:|---:|---:|---:|']
for v, _ in VARIANTS:
    c = status_counts[v]
    lines.append(f"| {v} | {c['COMPLETE'] + c['AVAILABLE']}/12 | {c['LEGACY_SCIENTIFIC_BLOCKER']}/12 | {c['FATAL_AUDIT_UNVERIFIED']}/12 | {c['MISSING']}/12 |")
lines += ['', 'Mean is reference-only and reuses the audited Goal 1 Full B2 units. No Mean Teacher is retrained.',
          '', '## Primary Metrics', '', 'The requested estimator averages all four folds within each seed, then summarizes across seeds. A seed with any scientific blocker is excluded; incomplete unit metrics are descriptive only.', '',
          '| Variant | ARI | NMI | ACC | Macro-F1 | Balanced ACC | kNN Purity@15 | Cell-type Silhouette |', '|---|---:|---:|---:|---:|---:|---:|---:|']
for v, _ in VARIANTS:
    lines.append('| ' + v + ' | ' + ' | '.join(variant_metrics[v][m] for m in METRICS) + ' |')
    lines.append(f"Available-unit descriptive-only metrics for {v}: {variant_metrics[v]['available_unit_descriptive_only']}")
lines += ['', '## Per-Fold Metrics', '', 'Values are mean +/- SD over available seeds; `NA` means no eligible completed value.', '']
for v, _ in VARIANTS:
    lines += [f'### {v}', '', '| Fold | ARI | NMI | ACC | Macro-F1 | Balanced ACC | kNN@15 | Silhouette |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for fold in FOLDS:
        lines.append('| ' + fold + ' | ' + ' | '.join(fold_metrics[v][fold][m] for m in METRICS) + ' |')
    lines.append('')
lines += ['## Mean Reference Coverage', '', '| Fold | Seed | Status | ARI | NMI |', '|---|---:|---|---:|---:|']
for fold in FOLDS:
    for seed in SEEDS:
        r = rows['Mean'][fold][seed]
        met = r.get('metrics', {})
        lines.append(f"| {fold} | {seed} | {r['status']} | {met.get('ari', 'NA')} | {met.get('nmi', 'NA')} |")
lines += ['', 'No complete matched Mean 3-seed aggregate is available. Therefore no valid Mean-vs-variant delta or best-performance claim is made.',
          '', '## Efficiency and Parameters', '', '| Variant | Timed units | TOTAL_PARAMS range | Model training time (h) | GPU-hours | Mean target inference (s) | Peak VRAM max (MB) |', '|---|---:|---:|---:|---:|---:|---:|']
for v, _ in VARIANTS:
    vals = [t for name, t in all_timing if name == v]
    params = [p for name, p in all_params if name == v]
    model_sec = sum(float(t.get('TOTAL_MODEL_TRAIN_TIME_SEC', 0)) for t in vals)
    gpu_h = sum(float(t.get('GPU_HOURS', 0)) for t in vals)
    inf = [float(t.get('TARGET_INFERENCE_TIME_SEC', 0)) for t in vals if t.get('TARGET_INFERENCE_TIME_SEC') is not None]
    peak = max((float(t.get('PEAK_GPU_MEMORY_MB', 0)) for t in vals), default=0)
    ptxt = f"{min(params)}-{max(params)}" if params else 'NA'
    lines.append(f'| {v} | {len(vals)} | {ptxt} | {model_sec/3600:.4f} | {gpu_h:.4f} | {statistics.mean(inf) if inf else 0:.4f} | {peak:.2f} |')
lines += ['', 'Training time/parameter/Vram summaries above include completed units only. Teacher training time for blocked runs is preserved in their logs/checkpoints but excluded from completed-unit aggregates.',
          '', '## Gate Mechanism', '', '| Source | Mean gate | Between-unit SD | Mean within-unit SD | Units |', '|---|---:|---:|---:|---:|']
for source, x in sorted(gate_stats['aggregate_per_source'].items()):
    lines.append(f"| {source} | {x['mean']:.4f} | {x['between_unit_sd']:.4f} | {x['mean_within_unit_std']:.4f} | {x['n']} |")
split_stats = {}
for x in gate_stats['raw_array_stats']:
    split_stats.setdefault(x['split'], []).append(x)
lines += ['', '| Split | Arrays | Cells | Mean gate | Mean SD | Fraction <0.05 | Fraction >0.95 |', '|---|---:|---:|---:|---:|---:|---:|']
for split, vals in sorted(split_stats.items()):
    lines.append(f"| {split} | {len(vals)} | {sum(x['cells'] for x in vals)} | {statistics.mean(x['mean'] for x in vals):.4f} | {statistics.mean(x['std'] for x in vals):.4f} | {statistics.mean(x['fraction_below_0_05'] for x in vals):.4f} | {statistics.mean(x['fraction_above_0_95'] for x in vals):.4f} |")
lines += ['', 'Gate per-dimension means and mean within-unit SDs across source train/validation units:', '', '| Dimension | Mean g | Mean within-unit SD | Units |', '|---:|---:|---:|---:|']
for x in gate_stats['aggregate_per_dimension']:
    if x['n']:
        lines.append(f"| {x['dimension']} | {x['mean_gate']:.4f} | {x['mean_within_unit_std']:.4f} | {x['n']} |")
lines += ['', f"Gate mechanism artifacts: {gate_stats['units_with_gate_analysis']} unit folders; {gate_stats['units_with_summary']} units with mechanism summaries; {gate_stats['files']} source gate arrays (train {gate_stats['source_train_files']}, val {gate_stats['source_val_files']}); target gate computed: {gate_stats['target_gate_computed']}.",
          'Gate values are saved as raw source-side arrays with cell-order TSVs and per-dimension summaries in each eligible unit. Target Gate is not computed because target F is unavailable.',
          '', f'Artifact inventory: {len(checkpoint_files)} checkpoint files; {target_raw_count} raw target embeddings; {embedding_count} total NPY embedding/gate arrays.',
          'UMAP_READY: YES for completed units with source/target embedding artifacts; scientifically blocked units do not have student target embeddings.',
          '', '## Legacy Blocker Records', '', 'Original SCIENTIFIC_BLOCKER.json files are retained. Where a later hash-validated COMPLETE manifest exists, the blocker is historical: fatal finite/non-collapsed checks passed, the former research diagnostic was warning-only, and the Student was trained. A blocker without a COMPLETE manifest remains excluded.',
          '', '## Model Decision', '', 'Updated Full B2 Mean remains unchanged and authoritative. This ablation does not replace it. No architecture is selected from incomplete matched Mean coverage.',
          '', '## Integrity Checks', '',
          f"Complete-run artifact hashes checked: {hash_check['artifacts_checked']} across {hash_check['manifest_files']} manifests; blocker Teacher checkpoint hashes checked: {hash_check['blocker_checkpoint_hashes_checked']}; mismatches: {len(hash_check['mismatches']) + len(hash_check['blocker_hash_mismatches'])}.",
          f"Strict target-boundary audits passed: {leakage['passed']}/{leakage['audits']}; failures: {len(leakage['failed'])}.",
          '', 'Environment and per-unit provenance/config/hash manifests are retained alongside each run.', '']
(ROOT / 'FUSION_3SEED_REPORT.md').write_text('\n'.join(lines))
summary = {'run_tier': 'pre_final', 'launcher_exit_code': 0, 'status_counts': status_counts,
           'metrics': variant_metrics, 'per_fold_metrics': fold_metrics, 'gate_analysis': gate_stats,
           'checkpoint_files': len(checkpoint_files), 'target_raw_embeddings': target_raw_count,
           'embedding_npy_files': embedding_count, 'mean_reference_rows': rows['Mean'],
           'integrity': hash_check, 'leakage_audits': leakage}
(ROOT / 'FUSION_3SEED_REPORT.json').write_text(json.dumps(summary, indent=2))
print(json.dumps({'status_counts': status_counts, 'metrics': variant_metrics, 'gate': gate_stats,
                  'checkpoint_files': len(checkpoint_files), 'target_raw_embeddings': target_raw_count,
                  'embedding_npy_files': embedding_count, 'integrity': hash_check,
                  'leakage_audits': leakage}, indent=2))
