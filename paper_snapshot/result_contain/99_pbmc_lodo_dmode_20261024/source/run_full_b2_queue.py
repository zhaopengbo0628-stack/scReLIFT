#!/usr/bin/env python
import json, subprocess, time, fcntl, os
from pathlib import Path
ROOT=Path('/home/u202522802045/share/zpb')
TASK=ROOT/'result_contain/99_pbmc_lodo_dmode_20261024'
RUN=TASK/'source/run_updated_b2_validated.py'
OUT=TASK/'full_b2'
PYTHON='/home/u202522802045/envs/graph_attention/bin/python'
FOLDS=['target_D1','target_D2','target_D3','target_D4']
SEEDS=[20020130,20020131,20020132]
PENDING=['Multigrate','ECLARE','scMomer','Palette-U','MultiVI','MIDAS','scButterfly-C']
def write(v):
    (TASK/'status.json').write_text(json.dumps(v,indent=2)+'\n')
def main():
    (TASK/'logs').mkdir(exist_ok=True)
    completed=failed=0
    base={'status':'RUNNING_FULL_B2_ONLY','updated':None,'completed_units':0,'planned_full_b2_units':12,'planned_units':96,
          'method_progress':{'Full B2':{'completed':0,'expected':12}},'pending_methods':PENDING,
          'smoke':'PASS','target_atac_used':False,'target_labels_before_embedding':False}
    lock=(ROOT/'.benchmark_a30.lock').open('a')
    os.dup2(lock.fileno(),9)
    for fold in FOLDS:
        for seed in SEEDS:
            unit=OUT/fold/f'seed_{seed}'
            manifest=unit/'manifest.json'
            if manifest.exists():
                try:
                    if json.loads(manifest.read_text()).get('status')=='COMPLETE':
                        completed+=1
                        base['completed_units']=completed;base['method_progress']['Full B2']['completed']=completed
                        base['updated']=time.strftime('%Y-%m-%dT%H:%M:%S');write(base);continue
                except Exception: pass
            log=TASK/'logs'/f'full_b2_{fold}_{seed}.log'
            cmd=[PYTHON,'-u',str(RUN),'--fold',fold,'--seed',str(seed),'--output-dir',str(unit)]
            base['current']={'fold':fold,'seed':seed};base['updated']=time.strftime('%Y-%m-%dT%H:%M:%S');write(base)
            with log.open('ab') as fh:
                fh.write(('CMD '+' '.join(cmd)+'\n').encode())
                fcntl.flock(9,fcntl.LOCK_EX)
                rc=subprocess.run(cmd,stdout=fh,stderr=subprocess.STDOUT,pass_fds=(9,)).returncode
                fcntl.flock(9,fcntl.LOCK_UN)
            if rc==0: completed+=1
            else:
                failed+=1
                base.setdefault('failed_runs',[]).append({'fold':fold,'seed':seed,'returncode':rc})
            base['completed_units']=completed;base['failed_units']=failed;base['method_progress']['Full B2']['completed']=completed
            base['updated']=time.strftime('%Y-%m-%dT%H:%M:%S');write(base)
    base['status']='PARTIAL_FULL_B2_COMPLETE_PENDING_COMPARATORS';base['current']=None;base['updated']=time.strftime('%Y-%m-%dT%H:%M:%S');write(base)
    lock.close()
if __name__=='__main__': main()

