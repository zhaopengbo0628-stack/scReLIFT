import json, subprocess, time
from pathlib import Path

BASE=Path('/home/u202522802045/share/zpb/result_contain/99_pbmc_lodo_dmode_20261024')
SCRIPT=BASE/'source/task99_lodo_adapters.py'
OUT=BASE/'units'
SEEDS=(20020130,20020131,20020132)
FOLDS=('target_D1','target_D2','target_D3','target_D4')
PY={'multigrate':'/home/u202522802045/envs/multigrate_002/bin/python','palette':'/home/u202522802045/envs/multigrate_002/bin/python','eclare':'/home/u202522802045/envs/scmomer_eclare_sintra/bin/python','multivi':'/home/u202522802045/envs/multivi_scvi/bin/python'}
METHODS=('palette','multigrate','eclare','multivi')
state=BASE/'formal_queue_status.json'
def write(obj): state.write_text(json.dumps(obj,indent=2)+'\n')
status={'status':'RUNNING','methods':list(METHODS),'folds':list(FOLDS),'seeds':list(SEEDS),'completed':[],'failed':[],'started':time.strftime('%Y-%m-%dT%H:%M:%S%z')}; write(status)
for method in METHODS:
  for fold in FOLDS:
    for seed in SEEDS:
      unit=OUT/method/fold/f'seed_{seed}'
      key=f'{method}/{fold}/{seed}'
      if (unit/'metrics.json').exists() and (unit/'adapter_smoke.json').exists(): status['completed'].append(key); write(status); continue
      unit.mkdir(parents=True,exist_ok=True)
      log=unit/'run.log'
      cmd=[PY[method],str(SCRIPT),'--method',method,'--fold',fold,'--seed',str(seed),'--out',str(unit),'--device','cuda','--peak-cap','2048','--formal']
      try:
        with log.open('w') as fh:
          subprocess.run(cmd,stdout=fh,stderr=subprocess.STDOUT,check=True)
        status['completed'].append(key)
      except Exception as exc:
        (unit/'failure.json').write_text(json.dumps({'status':'FAILED','error':repr(exc),'command':cmd},indent=2)+'\n'); status['failed'].append({'unit':key,'error':repr(exc)})
      write(status)
status['status']='COMPLETE' if not status['failed'] else 'PARTIAL'; status['finished']=time.strftime('%Y-%m-%dT%H:%M:%S%z'); write(status)
