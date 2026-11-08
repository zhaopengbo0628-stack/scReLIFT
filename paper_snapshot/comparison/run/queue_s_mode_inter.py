"""Serial PBMC inter queue: reuse completed Source models, then fit missing seeds."""
import argparse
import csv
import fcntl
import os
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pbmc_intra_common import ROOT, read, write, sha
from s_mode_inter import KEYS, SCORES, PROTOCOL, NAMES


def table(path,rows):
    if not rows: return
    write(path.with_suffix('.json'),rows)
    with path.with_suffix('.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    columns = list(rows[0])
    text = 'pre_final; equal-weight 12 directed pairs within each seed. NA means incomplete or unavailable.\n\n'
    text += '| '+' | '.join(columns)+' |\n|'+'---|'*len(columns)+'\n'
    for row in rows:
        text += '| '+' | '.join('NA' if row[c] is None else f'{row[c]:.6f}' if isinstance(row[c],float) else str(row[c]) for c in columns)+' |\n'
    path.with_suffix('.md').write_text(text)


def collect(task):
    out = Path(task['result_path'])
    records = [read(p) for p in sorted((out/'evaluations').glob('D*/seed_*/D*/manifest.json'))]
    for r in records:
        assert r['status']=='COMPLETE' and r['protocol_sha256']==sha(PROTOCOL)
    rows = [dict(Source=r['source'],Target=r['target'],Seed=r['seed'],**r['display']) for r in records]
    table(out/'metrics/per_unit',rows)
    complete = []
    columns = read(PROTOCOL)['display_columns']
    for seed in task['seeds']:
        selected = [r for r in records if r['seed']==seed]
        pairs = {(r['source'],r['target']) for r in selected}
        if pairs != {(s,t) for s in KEYS for t in KEYS if s!=t}: continue
        row = dict(Seed=seed,**dict.fromkeys(columns))
        row.update(Method=task['method_name'],GPU=', '.join(sorted({r['display']['GPU'] for r in selected})))
        for column in columns[1:-1]:
            values = [r['display'][column] for r in selected]
            if all(isinstance(v,(int,float)) for v in values): row[column] = float(np.mean(values))
        complete.append(row)
    table(out/'metrics/per_seed_12pair_mean',complete)
    mean = dict(Method=task['method_name'],Evaluations=len(records),Expected=60,Complete_seeds=len(complete),**dict.fromkeys(columns[1:]))
    sd = dict(mean)
    mean['GPU'] = sd['GPU'] = ', '.join(sorted({r['display']['GPU'] for r in records})) or None
    if len(complete)==5:
        for column in columns[1:-1]:
            values = [r[column] for r in complete]
            if all(isinstance(v,(int,float)) for v in values):
                mean[column] = float(np.mean(values)); sd[column] = float(np.std(values,ddof=1))
    table(out/'metrics/inter_mean',[mean]); table(out/'metrics/inter_sample_sd',[sd])
    coverage = ['Method','Evaluations','Expected','Complete_seeds']
    score_columns = coverage+list(SCORES)
    efficiency_columns = coverage+columns[8:]
    table(out/'metrics/scoring_mean',[{c:mean[c] for c in score_columns}])
    table(out/'metrics/efficiency_mean',[{c:mean[c] for c in efficiency_columns}])
    annotation = ['ari','nmi','ami','cluster_acc','accuracy','balanced_accuracy','macro_f1','weighted_f1']
    table(out/'metrics/annotation_per_unit',[dict(Source=r['source'],Target=r['target'],Seed=r['seed'],
        **{k:r['metrics'][k] for k in annotation}) for r in records])
    return len(records)


def status(task,value):
    out = Path(task['result_path']); now = datetime.now(timezone.utc).isoformat()
    manifest = read(out/'manifest.json')
    manifest.update(status=value,updated_at=now,completed_evaluations=collect(task),queue_pid=os.getpid())
    write(out/'manifest.json',manifest)
    with (ROOT/'result_contain/task_registry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        registry = ROOT/'result_contain/task_registry.csv'
        with registry.open() as handle:
            reader=csv.DictReader(handle); fields=reader.fieldnames; rows=list(reader)
        row = next(r for r in rows if int(r['task_id'])==task['task_id'])
        row.update(status=value,updated_at=now,pid=str(os.getpid()))
        if value=='COMPLETED': row['completed_at']=now
        temp=registry.with_suffix('.inter.tmp')
        with temp.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        os.replace(temp,registry)


def execute(task,source,seed,train=False,smoke=False):
    out = Path(task['result_path']); unit = out/'checkpoints'/source/f'seed_{seed}'
    for path,digest in task['source_sha256'].items(): assert sha(path)==digest,path
    if train:
        command = [task['python'],'-B',str(ROOT/f'comparison/run/{task["method"]}/s_mode_intra.py'),
            '--stage','unit','--dataset',source,'--seed',str(seed),'--device',task['device'],
            '--output-root',str(out/'checkpoints'),'--run-seeds',*[str(s) for s in task['seeds']]]
    else:
        command = [task['python'],'-B',str(ROOT/'comparison/run/s_mode_inter.py'),
            '--method',task['method'],'--source',source,'--seed',str(seed),'--unit',str(unit),'--output',str(out)]
        if smoke: command += ['--smoke']
    kind = 'train' if train else 'smoke' if smoke else 'inter'
    write(out/'logs'/f'{kind}_{source}_{seed}_command.json',dict(argv=command))
    print('RUN',task['task_id'],kind,source,seed,flush=True)
    with (out/'logs'/f'{kind}_{source}_{seed}.log').open('a') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


def main(plan):
    failures = []
    blocked = set()
    with (ROOT/'.comparison_sinter_queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            # Source intra reload checks for all five baselines before full inter dispatch.
            for task in plan['tasks']:
                if task['method']=='b2': continue
                try: execute(task,'D1',20020130,smoke=True)
                except Exception:
                    blocked.add(task['task_id'])
                    failures.append(dict(task_id=task['task_id'],stage='reload_gate',traceback=traceback.format_exc()))
            for task in plan['tasks']:
                if task['task_id'] in blocked: continue
                status(task,'RUNNING_EXISTING_CHECKPOINTS')
                for source in KEYS:
                    for seed in task['seeds']:
                        if task['method']!='b2' and seed>20020132: continue
                        unit = Path(task['result_path'])/'checkpoints'/source/f'seed_{seed}'
                        if all((Path(task['result_path'])/'evaluations'/source/f'seed_{seed}'/target/'manifest.json').exists() for target in KEYS if target!=source): continue
                        try: execute(task,source,seed)
                        except Exception:
                            failures.append(dict(task_id=task['task_id'],source=source,seed=seed,stage='inter',traceback=traceback.format_exc()))
                        status(task,'RUNNING_EXISTING_CHECKPOINTS')
                status(task,'COMPLETED' if collect(task)==60 else 'WAITING_MISSING_SEEDS')
            for task in plan['tasks']:
                if task['method']=='b2' or task['task_id'] in blocked: continue
                status(task,'FITTING_MISSING_SEEDS')
                for source in KEYS:
                    for seed in task['seeds'][3:]:
                        unit = Path(task['result_path'])/'checkpoints'/source/f'seed_{seed}'
                        try:
                            if not (unit/'manifest.json').exists(): execute(task,source,seed,train=True)
                            assert read(unit/'manifest.json')['status']=='COMPLETE'
                            execute(task,source,seed)
                        except Exception:
                            failures.append(dict(task_id=task['task_id'],source=source,seed=seed,stage='missing_seed',traceback=traceback.format_exc()))
                        status(task,'FITTING_MISSING_SEEDS')
                status(task,'COMPLETED' if collect(task)==60 else 'PARTIAL_FAILED')
        except Exception:
            failures.append(dict(stage='queue_gate',traceback=traceback.format_exc()))
        finally:
            for task in plan['tasks']:
                out=Path(task['result_path']); count=collect(task)
                value='COMPLETED' if count==60 else 'PARTIAL_FAILED'
                status(task,value)
                write(out/'logs/queue_failures.json',[r for r in failures if r.get('task_id',task['task_id'])==task['task_id']])
                (out/'logs/exitcode').write_text('0\n' if count==60 else '1\n')
            write(Path(plan['queue_path'])/'queue_status.json',dict(status='FINISHED',failures=failures))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--plan',type=Path,required=True)
    main(read(parser.parse_args().plan))
