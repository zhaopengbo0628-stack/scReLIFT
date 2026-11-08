"""Persistent, per-fold CPU pipeline: discovery, full-gene DEG, all-seed GO/Reactome."""
import os,sys,json,time,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PY='/home/u202522802045/envs/graph_attention/bin/python'
target=sys.argv[1];run=ROOT/'analyses'/target
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMBA_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')


def call(script):
    print('STAGE_START',target,script,flush=True)
    subprocess.run([PY,'-u',str(run/'source'/script)],cwd=run,env=env,check=True)
    print('STAGE_COMPLETE',target,script,flush=True)


if target!='D1':
    while not (ROOT/'folds'/target/'audit/ALL_UMAPS_COMPLETE.json').exists():time.sleep(30)
    call('inventory.py');call('deg_all_candidates.py')
call('fgsea_all_seeds_expressed.py')
call('fgsea_all_seeds.py')
p=run/'audit/ALL_SEED_ANALYSIS_COMPLETE.json'
p.write_text(json.dumps(dict(status='COMPLETE',fold=target,all_seed_DEG=True,all_seed_GSEA=True,primary_and_expression_sensitivity=True),indent=2))
print('ALL_SEED_ANALYSIS_COMPLETE',target,flush=True)
