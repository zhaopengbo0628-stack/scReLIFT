#!/usr/bin/env python
import json, subprocess, time
from pathlib import Path
ROOT = Path('/home/u202522802045/share/zpb')
RUN = ROOT/'comparison/run/bmmc_multivi_intra.py'
OUT = ROOT/'comparison/result/BMMC_intra_multivi_v2'
SITES = ['BMMC_Site1','BMMC_Site2','BMMC_Site3','BMMC_Site4']
SEEDS = [20020130,20020131,20020132]
PYTHON = '/home/u202522802045/envs/multivi_scvi/bin/python'
def status(v):
    (OUT/'QUEUE_STATUS.json').write_text(json.dumps({'updated':time.strftime('%Y-%m-%dT%H:%M:%S'),**v},indent=2)+'\n')
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    status({'state':'FULL','method':'MultiVI','completed':0,'failed':0,'expected':12,'smoke':'PASS'})
    completed=failed=0
    for site in SITES:
        for seed in SEEDS:
            unit = OUT/'pre_final'/site/f'seed_{seed}'
            if (unit/'manifest.json').exists():
                try:
                    if json.loads((unit/'manifest.json').read_text()).get('status') == 'COMPLETE':
                        completed += 1
                        status({'state':'FULL','method':'MultiVI','site':site,'seed':seed,'result':'SKIP','completed':completed,'failed':failed,'expected':12,'smoke':'PASS'})
                        continue
                except Exception:
                    pass
            cmd=[PYTHON,'-u',str(RUN),'--stage','unit','--dataset',site,'--seed',str(seed),'--device','cuda','--output-root',str(OUT)]
            log=OUT/'logs'/f'{site}_seed_{seed}.log'; log.parent.mkdir(parents=True,exist_ok=True)
            status({'state':'FULL','method':'MultiVI','site':site,'seed':seed,'result':'RUNNING','completed':completed,'failed':failed,'expected':12,'smoke':'PASS'})
            with log.open('ab') as fh:
                fh.write(('CMD '+' '.join(cmd)+'\n').encode())
                rc=subprocess.run(cmd,stdout=fh,stderr=subprocess.STDOUT).returncode
            if rc == 0:
                completed += 1
                result='RC=0'
            else:
                failed += 1
                result=f'RC={rc}'
            status({'state':'FULL','method':'MultiVI','site':site,'seed':seed,'result':result,'completed':completed,'failed':failed,'expected':12,'smoke':'PASS'})
    status({'state':'COMPLETE','method':'MultiVI','completed':completed,'failed':failed,'expected':12,'smoke':'PASS'})
if __name__ == '__main__':
    main()
