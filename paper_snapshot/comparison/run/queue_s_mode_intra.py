"""Run the declared comparison tasks serially, preserving trained checkpoints."""
import argparse
import csv
import fcntl
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pbmc_intra_common import ROOT, prepare, read, write, sha, evaluate

SCORES = {'ARI': 'ari', 'NMI': 'nmi', 'ACC': 'cluster_acc', 'Macro-F1': 'macro_f1',
          'Balanced ACC': 'balanced_accuracy', 'Purity@15': 'knn_purity_at_15', 'Silhouette': 'silhouette'}
COLUMNS = list(SCORES) + ['Method', 'Total time', 'Trainable params', 'Peak VRAM',
                         'Inference time', 'Cells/sec', 'GPU', 'Dataset', 'Seed', 'Status']


def tables(folder, name, rows):
    folder.mkdir(parents=True, exist_ok=True)
    write(folder/f'{name}.json', rows)
    with (folder/f'{name}.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    text = '3-seed pre_final; seconds, Peak VRAM in GiB (allocated). ACC/F1 are posthoc Hungarian cluster scores.\n\n'
    text += '| ' + ' | '.join(COLUMNS) + ' |\n|' + '---|'*len(COLUMNS) + '\n'
    text += ''.join('| ' + ' | '.join(str(row[c]) for c in COLUMNS) + ' |\n' for row in rows)
    (folder/f'{name}.md').write_text(text)


def collect(task):
    out = Path(task['result_path'])
    rows, details = [], []
    for dataset in task['datasets']:
        for seed in task['seeds']:
            unit = out/'checkpoints'/dataset/f'seed_{seed}'
            row = dict.fromkeys(COLUMNS, 'NA')
            row.update(Method=task['method_name'], Dataset=dataset, Seed=seed, Status='PENDING')
            if (unit/'manifest.json').exists():
                manifest = read(unit/'manifest.json')
                row['Status'] = manifest['status']
                checkpoint = unit/manifest.get('checkpoint_file', 'source_palette.rds' if task['method']=='palette' else 'best.pt')
                assert sha(checkpoint) == manifest['checkpoint_sha256']
                row.update({'Total time': manifest['total_wall_seconds'],
                            'Trainable params': manifest['trainable_parameters'] if manifest['trainable_parameters'] is not None else 'NA',
                            'Peak VRAM': manifest['peak_gpu_allocated_bytes']/(1024**3) if manifest['peak_gpu_allocated_bytes'] is not None else 'NA',
                            'Inference time': manifest['inference_seconds'], 'Cells/sec': manifest['cells_per_second'],
                            'GPU': manifest.get('gpu_model') or 'CPU (GPU unused)'})
                if (unit/'metrics.json').exists():
                    metrics = read(unit/'metrics.json')
                    assert sha(unit/'rna_test_embedding.npy') == metrics['prediction_sha256']
                    row.update({column: metrics[field] if metrics[field] is not None else 'NA' for column, field in SCORES.items()})
                    details.append(dict(dataset=dataset, seed=seed, checkpoint_path=str(checkpoint),
                                        checkpoint_sha256=manifest['checkpoint_sha256'], resources=manifest, metrics=metrics))
            elif unit.exists():
                row['Status'] = 'INCOMPLETE_UNIT'
            rows.append(row)
    tables(out/'metrics', 'per_seed', rows)
    aggregate = []
    for dataset in task['datasets']:
        selected = [r for r in rows if r['Dataset']==dataset and r['Status']=='COMPLETE']
        row = dict.fromkeys(COLUMNS, 'NA')
        row.update(Method=task['method_name'], Dataset=dataset, Seed=f'{len(selected)}/{len(task["seeds"])}',
                   Status='COMPLETE' if len(selected)==len(task['seeds']) else 'PARTIAL')
        for column in list(SCORES)+['Total time','Trainable params','Peak VRAM','Inference time','Cells/sec']:
            values = [r[column] for r in selected if isinstance(r[column], (float,int))]
            if values:
                row[column] = f'{np.mean(values):.6f} ± {np.std(values,ddof=1):.6f}' if len(values)>1 else f'{values[0]:.6f} (n=1)'
        row['GPU'] = ', '.join(sorted({r['GPU'] for r in selected})) or 'NA'
        aggregate.append(row)
    tables(out/'metrics', 'mean_std', aggregate)
    write(out/'metrics/audit_details.json', details)
    return rows


def status(task, value):
    out = Path(task['result_path'])
    manifest = read(out/'manifest.json')
    now = datetime.now(timezone.utc).isoformat()
    manifest.update(status=value, updated_at=now, queue_pid=os.getpid())
    write(out/'manifest.json', manifest)
    registry = ROOT/'result_contain/task_registry.csv'
    with (ROOT/'result_contain/task_registry.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with registry.open(newline='') as handle:
            reader = csv.DictReader(handle)
            fields, records = reader.fieldnames, list(reader)
        row = next(r for r in records if int(r['task_id'])==task['task_id'])
        row.update(status=value, updated_at=now, pid=str(os.getpid()))
        if value=='COMPLETED':
            row['completed_at'] = now
        temp = registry.with_suffix('.queue.tmp')
        with temp.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        temp.replace(registry)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    with (ROOT/'.comparison_sintra_queue.lock').open('a') as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for task in plan['tasks']:
            out = Path(task['result_path'])
            status(task, 'RUNNING')
            failures = []
            for dataset in task['datasets']:
                for seed in task['seeds']:
                    unit = out/'checkpoints'/dataset/f'seed_{seed}'
                    try:
                        for path, digest in task['source_sha256'].items():
                            assert sha(path)==digest, f'Source changed since submission: {path}'
                        if (unit/'manifest.json').exists() and read(unit/'manifest.json')['status']=='COMPLETE':
                            collect(task)
                            continue
                        if (unit/'manifest.json').exists() and read(unit/'manifest.json')['status']=='TRAINED':
                            evaluate(unit, prepare(dataset), dataset)
                            manifest = read(unit/'manifest.json')
                            manifest['status'] = 'COMPLETE'
                            write(unit/'manifest.json', manifest)
                        else:
                            command = [task['python'], '-B', task['entrypoint'], '--dataset', dataset,
                                       '--seed', str(seed), '--output-root', str(out/'checkpoints')]
                            if task['method']!='scbutterfly_b':
                                command += ['--stage','unit','--device',task['device'],'--run-seeds'] + [str(s) for s in task['seeds']]
                            print('RUN', task['task_id'], command, flush=True)
                            with (out/'logs'/f'{dataset}_seed_{seed}.log').open('a') as log:
                                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                    except Exception as error:
                        failures.append(dict(dataset=dataset, seed=seed, error=repr(error)))
                        write(out/'logs/queue_failures.json', failures)
                        print('UNIT_FAILED', task['task_id'], dataset, seed, repr(error), flush=True)
                    try:
                        collect(task)
                    except Exception as error:
                        failures.append(dict(dataset=dataset, seed=seed, error='summary: '+repr(error)))
                        write(out/'logs/queue_failures.json', failures)
            rows = collect(task)
            done = sum(row['Status']=='COMPLETE' for row in rows)
            status(task, 'COMPLETED' if done==len(rows) and not failures else 'PARTIAL_FAILED')
            (out/'logs/exitcode').write_text('0\n' if done==len(rows) and not failures else '1\n')
        write(args.plan.parent/'queue_status.json', dict(status='FINISHED', tasks=[
            dict(task_id=t['task_id'], status=read(Path(t['result_path'])/'manifest.json')['status']) for t in plan['tasks']]))


if __name__ == '__main__':
    main()
