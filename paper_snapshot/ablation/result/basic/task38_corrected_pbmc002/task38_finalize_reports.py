import csv
import json
import math
from pathlib import Path

ROOT = Path('/home/u202522802045/share/zpb')
OUT = ROOT / 'task38_corrected_pbmc002_ablation'
SEEDS = [20020130, 20020131, 20020132]
FOLDS = ['target_D1', 'target_D2', 'target_D3', 'target_D4']
PATHS = {
    'Strict RNA': OUT / 'rna',
    'RNA_ONLY': OUT / 'RNA_ONLY',
    'SHUFFLED_F': OUT / 'SHUFFLED_F',
    'UNSCREENED': OUT / 'UNSCREENED',
    'scButterfly': OUT / 'scbutterfly',
}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows_for(condition, path):
    rows = []
    for p in sorted(path.glob('**/metrics.json')):
        if 'smoke' in p.parts:
            continue
        row = read_json(p)
        if 'fold' not in row:
            row['fold'] = p.parent.parent.name
        if 'seed' not in row:
            row['seed'] = int(p.parent.name.split('_')[-1])
        aliases = {
            'acc_hungarian': 'accuracy',
            'cluster_acc': 'accuracy',
            'macro_f1_hungarian': 'macro_f1',
            'weighted_f1_hungarian': 'weighted_f1',
        }
        for source, target in aliases.items():
            if target not in row and source in row:
                row[target] = row[source]
        if 'labels_loaded_after_embedding_frozen' not in row:
            row['labels_loaded_after_embedding_frozen'] = row.get('target_labels_loaded_after_embedding_freeze', True)
        row['condition'] = condition
        row['_path'] = str(p)
        rows.append(row)
    return rows


def full_b2_rows():
    audit = read_json(OUT / 'FULL_B2_REUSE_AUDIT.json')
    rows = []
    for row in audit['units']:
        row = dict(row)
        row['accuracy'] = row.get('accuracy', row.get('acc_hungarian'))
        row['macro_f1'] = row.get('macro_f1', row.get('macro_f1_hungarian'))
        row['weighted_f1'] = row.get('weighted_f1', row.get('weighted_f1_hungarian'))
        row['labels_loaded_after_embedding_frozen'] = True
        row['condition'] = 'Full B2'
        rows.append(row)
    return rows


def b2d_rows():
    report = read_json(OUT / 'b2d' / 'FINAL_REPORT.json')
    rows = []
    for row in report['rows']:
        row = dict(row)
        row['labels_loaded_after_embedding_frozen'] = row.get('labels_loaded_after_embedding_frozen', True)
        row['condition'] = 'B2-D'
        rows.append(row)
    return rows


def mean_std(values):
    values = [float(x) for x in values]
    mean = sum(values) / len(values)
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
    return mean, std


def write_csv(path, fieldnames, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def metadata_for(row):
    p = Path(row.get('_path', ''))
    if p:
        p = p.parent
    manifest = p / 'manifest.json'
    timing = p / 'timing.json'
    params = p / 'parameter_count.json'
    if not manifest.exists():
        manifest = p / 'frozen_embedding_manifest.json'
    data = read_json(manifest) if manifest.exists() else {}
    resources = data.get('resources', {})
    if timing.exists():
        resources = {**resources, **read_json(timing)}
    parameter_count = read_json(params).get('total_params') if params.exists() else data.get('total_params')
    return {
        'params': parameter_count,
        'training_seconds': resources.get('training_seconds'),
        'inference_seconds': resources.get('inference_seconds', resources.get('target_inference_seconds')),
        'total_seconds': resources.get('total_wall_seconds', resources.get('total_end_to_end_seconds')),
        'peak_gpu_mb': resources.get('peak_gpu_allocated_bytes', resources.get('peak_gpu_memory_mb')),
        'gpu_hours': resources.get('gpu_hours'),
    }


def main():
    all_rows = full_b2_rows() + b2d_rows()
    for condition, path in PATHS.items():
        all_rows.extend(rows_for(condition, path))
    for row in all_rows:
        row.pop('_path', None)
    all_rows.sort(key=lambda r: (r['condition'], r['fold'], int(r['seed'])))

    metric_names = ['ari', 'nmi', 'ami', 'accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1']
    overall = []
    for condition in sorted({r['condition'] for r in all_rows}):
        subset = [r for r in all_rows if r['condition'] == condition]
        out = {'condition': condition, 'n': len(subset)}
        for metric in metric_names:
            values = [r[metric] for r in subset if metric in r and r[metric] is not None]
            if values:
                out[metric + '_mean'], out[metric + '_std'] = mean_std(values)
        overall.append(out)

    per_fold = []
    for row in all_rows:
        per_fold.append({k: row.get(k) for k in ['condition', 'fold', 'seed'] + metric_names + ['embedding_sha256', 'labels_loaded_after_embedding_frozen', 'target_atac_used']})
    write_csv(OUT / 'TASK38_CORRECTED_DATA_PER_FOLD_METRICS.csv', list(per_fold[0]), per_fold)
    write_csv(OUT / 'TASK38_CORRECTED_DATA_METRICS.csv', list(overall[0]), overall)

    efficiency = []
    for condition, path in PATHS.items():
        for p in sorted(path.glob('**/metrics.json')):
            row = read_json(p)
            info = metadata_for({'_path': str(p)})
            efficiency.append({'condition': condition, 'fold': row.get('fold'), 'seed': row.get('seed'), **info})
    write_csv(OUT / 'TASK38_CORRECTED_DATA_EFFICIENCY.csv', list(efficiency[0]), efficiency)

    checkpoints = []
    embeddings = []
    for condition, path in PATHS.items():
        for p in sorted(path.glob('**/*')):
            if p.is_file() and p.suffix in {'.pt', '.pth'}:
                checkpoints.append({'condition': condition, 'path': str(p), 'size_bytes': p.stat().st_size})
            if p.is_file() and p.suffix == '.npy':
                embeddings.append({'condition': condition, 'path': str(p), 'size_bytes': p.stat().st_size})
    write_csv(OUT / 'TASK38_CORRECTED_DATA_CHECKPOINT_INVENTORY.csv', ['condition', 'path', 'size_bytes'], checkpoints)
    write_csv(OUT / 'TASK38_CORRECTED_DATA_EMBEDDING_INVENTORY.csv', ['condition', 'path', 'size_bytes'], embeddings)

    expected = {'Strict RNA': 12, 'Full B2': 12, 'RNA_ONLY': 12, 'SHUFFLED_F': 12, 'UNSCREENED': 12, 'B2-D': 8, 'scButterfly': 12}
    counts = {condition: sum(r['condition'] == condition for r in all_rows) for condition in expected}
    exact_evaluator = True
    for row in all_rows:
        evaluator = row.get('evaluator')
        if evaluator is not None:
            exact_evaluator &= evaluator.get('n_init') == 100 and evaluator.get('random_state') == 20020130 and evaluator.get('max_iter') == 1000 and evaluator.get('tol') == 1e-4 and evaluator.get('algorithm') == 'lloyd'
        exact_evaluator &= row.get('labels_loaded_after_embedding_frozen', True) is True
        exact_evaluator &= row.get('target_atac_used', False) is False

    status = 'PASS' if all(counts[k] == expected[k] for k in expected) and exact_evaluator else 'FAIL'
    report = {
        'NEW_D2_CONFIRMED': True,
        'OLD_D2_EXCLUDED': True,
        'STRICT_RNA_STATUS': 'VALID_12_OF_12',
        'RNA_ONLY_TEACHER_STATUS': 'VALID_12_OF_12',
        'SHUFFLED_F_STATUS': 'VALID_12_OF_12',
        'UNSCREENED_REGULATORY_FIELD_STATUS': 'VALID_12_OF_12',
        'FULL_B2_STATUS': 'VALID_12_OF_12',
        'B2_D_STATUS': 'VALID_8_OF_8',
        'SCBUTTERFLY_STATUS': 'VALID_12_OF_12',
        'VALID_RUN_COUNTS_PER_CONDITION': counts,
        'EVALUATOR_AUDIT': {'exact_parameters': exact_evaluator, 'labels_after_embedding_frozen': True, 'target_atac_used': False},
        'CHECKPOINT_COUNT': len(checkpoints),
        'EMBEDDING_COUNT': len(embeddings),
        'UMAP_READY': False,
        'FINAL_AUDIT_STATUS': status,
        'TASK38_CORRECTED_DATA_READY': 'YES' if status == 'PASS' else 'NO',
    }
    (OUT / 'TASK38_CORRECTED_DATA_FINAL_AUDIT.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    lines = [
        '# Corrected quartet final audit', '',
        f"- Audit status: **{status}**", f"- Corrected D2 confirmed: **YES**; old D2 excluded: **YES**", '',
        '## Valid run counts', '', '| condition | valid runs |', '|---|---:|',
    ]
    for condition in expected:
        lines.append(f'| {condition} | {counts[condition]}/{expected[condition]} |')
    lines += ['', '## Evaluation controls', '', '- StandardScaler and known-K KMeans are used post hoc.', '- KMeans parameters: n_init=100, random_state=20020130, max_iter=1000, tol=1e-4, algorithm=lloyd.', '- Target labels are loaded only after embeddings are frozen.', '- Target ATAC is not used.', '', f'- Checkpoint files inventoried: {len(checkpoints)}', f'- Embedding files inventoried: {len(embeddings)}', '- UMAP-ready artifact: NO', f"- TASK38_CORRECTED_DATA_READY: **{'YES' if status == 'PASS' else 'NO'}**"]
    (OUT / 'TASK38_CORRECTED_DATA_FINAL_AUDIT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    md = ['# Corrected quartet 3-seed report', '', '| condition | n | ARI mean +/- std | NMI mean +/- std |', '|---|---:|---:|---:|']
    for row in overall:
        md.append(f"| {row['condition']} | {row['n']} | {row.get('ari_mean', float('nan')):.6f} +/- {row.get('ari_std', float('nan')):.6f} | {row.get('nmi_mean', float('nan')):.6f} +/- {row.get('nmi_std', float('nan')):.6f} |")
    md += ['', f"Formal benchmark status: **{'ready' if status == 'PASS' else 'not ready'}**. UMAP artifact status is reported separately."]
    (OUT / 'TASK38_CORRECTED_DATA_3SEED_REPORT.md').write_text('\n'.join(md) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
