"""Prepare on CPU, queue after task49, then run the five-seed B2-S ablation."""
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

ROOT=Path('/home/u202522802045/share/zpb')
config_path=Path(sys.argv[1]);config=json.loads(config_path.read_text())
out=Path(config['result_root']);prefix=ROOT/'slum'/out.name
manifest=json.loads((out/'manifest.json').read_text())
manifest['launch_time']=datetime.now(timezone.utc).isoformat()
METRICS=['ari','nmi','ami','cluster_acc','accuracy','balanced_accuracy','macro_f1','weighted_f1','knn_purity_at_15']


def status(value):
    now=datetime.now(timezone.utc).isoformat();manifest.update(status=value,pid=os.getpid(),updated_at=now)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    registry=ROOT/'result_contain/task_registry.csv'
    with (ROOT/'result_contain/task_registry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with registry.open(newline='') as stream:
            reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
        row=next(r for r in rows if r['task_id']=='50')
        row.update(status=value,pid=str(os.getpid()),updated_at=now,launch_time=manifest['launch_time'])
        if value=='COMPLETED':row['completed_at']=now
        tmp=registry.with_suffix('.tmp')
        with tmp.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        tmp.replace(registry)


def run(stage,source,seed=20020130):
    command=[sys.executable,'-B',config['runner'],'--config',str(config_path),
             '--stage',stage,'--source',source,'--seed',str(seed)]
    print('RUN',command,flush=True);subprocess.run(command,check=True)


def table(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if rows:
        with path.with_suffix('.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        lines=['| '+' | '.join(rows[0])+' |','|'+'---|'*len(rows[0])]
        lines += ['| '+' | '.join('NA' if v is None else str(v) for v in r.values())+' |' for r in rows]
        path.with_suffix('.md').write_text('\n'.join(lines)+'\n')


def summarize():
    records=[];efficiency=[];graphs=[]
    for source in ['D1','D2','D3','D4']:
        cache=json.loads((Path(config['cache_root'])/source/'manifest.json').read_text())
        graphs.append(dict(source=source,regulatory_graph_used=False,candidate_edges=0,retained_edges=0,
                           selector_seconds=0,field_seconds=0,atac_peaks=cache['atac_peaks'],
                           preprocessing_seconds=cache['preprocessing_seconds']))
    for path in sorted((out/'models').glob('*/seed_*/evaluation/*/manifest.json')):
        m=json.loads(path.read_text());assert m['status']=='COMPLETE'
        records.append(dict(source=m['source'],target=m['target'],seed=m['seed'],
                            family='intra' if m['source']==m['target'] else 'inter',
                            **{k:m['metrics'][k] for k in METRICS},geometry_pearson=None,geometry_spearman=None,
                            annotation_scope='posthoc Hungarian cluster scores, not supervised label transfer'))
        training=json.loads((path.parents[2]/'training_manifest.json').read_text())
        t=training['stages']['teacher']['resources'];s=training['stages']['student']['resources']
        cache=json.loads((Path(config['cache_root'])/m['source']/'manifest.json').read_text())
        efficiency.append(dict(source=m['source'],target=m['target'],seed=m['seed'],
            teacher_trainable_parameters=t['parameters'],student_trainable_parameters=s['parameters'],
            teacher_training_seconds=t['training_seconds'],student_training_seconds=s['training_seconds'],
            preprocessing_seconds=cache['preprocessing_seconds'],
            total_wall_seconds=t['training_seconds']+s['training_seconds']+cache['preprocessing_seconds']+m['inference_seconds'],
            peak_allocated_bytes=max(t['peak_allocated_bytes'],s['peak_allocated_bytes']),
            peak_reserved_bytes=max(t['peak_reserved_bytes'],s['peak_reserved_bytes']),
            inference_seconds=m['inference_seconds'],cells_per_second=m['cells_per_second'],GPU=training['environment']['gpu']))
    for family in ['intra','inter']:
        selected=[r for r in records if r['family']==family];aggregate=[]
        for source in ['D1','D2','D3','D4']:
            for target in ['D1','D2','D3','D4']:
                rows=[r for r in selected if r['source']==source and r['target']==target]
                if not rows:continue
                r=dict(source=source,target=target,n_seeds=len(rows),run_tier='pre_final')
                for metric in METRICS:
                    values=[x[metric] for x in rows];r[metric+'_mean']=float(np.mean(values))
                    r[metric+'_sample_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
                aggregate.append(r)
        table(out/'metrics'/family/'per_seed',selected);table(out/'metrics'/family/'mean_std',aggregate)
    table(out/'metrics/efficiency',efficiency);table(out/'metrics/regulatory_graphs',graphs)
    complete=len(records)==80 and len(list((out/'models').glob('*/seed_*/manifest.json')))==20
    if complete:
        expected={(s,t,seed) for s in ['D1','D2','D3','D4'] for t in ['D1','D2','D3','D4'] for seed in config['seeds']}
        assert {(r['source'],r['target'],r['seed']) for r in records}==expected
        assert all(np.isfinite(r[k]) for r in records for k in METRICS)
        for unit in (out/'models').glob('*/seed_*'):
            for stage in ['teacher','student']:
                m=json.loads((unit/'checkpoints'/stage/'manifest.json').read_text())
                history=json.loads((unit/'checkpoints'/stage/'history.json').read_text())
                assert m['best_source_val_score']==min(r['val'][-1] for r in history)
    (out/'metrics/status.json').write_text(json.dumps(dict(status='COMPLETE' if complete else 'PARTIAL',
        completed_evaluations=len(records),expected_evaluations=80,run_tier='pre_final',seeds=config['seeds']),indent=2)+'\n')
    return complete


Path(str(prefix)+'.pid').write_text(str(os.getpid())+'\n')
try:
    status('CPU_PREFLIGHT_RUNNING')
    for source in ['D1','D2','D3','D4']:run('preflight',source)
    status('QUEUED_AFTER_TASK49')
    while not Path(config['wait_exitcode']).exists():
        try:os.kill(config['wait_pid'],0)
        except ProcessLookupError:break
        time.sleep(30)
    status('RUNNING')
    for source in ['D1','D2','D3','D4']:
        for seed in config['seeds']:
            run('run',source,seed);summarize()
    assert summarize()
    status('COMPLETED');Path(str(prefix)+'.exitcode').write_text('0\n')
except BaseException:
    status('FAILED');Path(str(prefix)+'.exitcode').write_text('1\n')
    raise
