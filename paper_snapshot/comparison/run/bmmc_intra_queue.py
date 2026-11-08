"""Sequential BMMC intra queue: missing smokes first, then 60 full units."""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT=Path('/home/u202522802045/share/zpb')
RUN=ROOT/'comparison/run'
OUT=ROOT/'comparison/result/BMMC_intra'
SITES=['BMMC_Site1','BMMC_Site2','BMMC_Site3','BMMC_Site4']
SEEDS=[0,1,2]
METHODS={
 'multigrate': dict(python='/home/u202522802045/envs/multivi_scvi/bin/python', entry='bmmc_s_mode_intra.py', extra=['--method','multigrate'], device='cuda'),
 'cobolt': dict(python='/home/u202522802045/envs/graph_attention/bin/python', entry='bmmc_s_mode_intra.py', extra=['--method','cobolt'], device='cuda'),
 'palette': dict(python='/home/u202522802045/envs/graph_attention/bin/python', entry='bmmc_s_mode_intra.py', extra=['--method','palette'], device='cpu'),
 'eclare': dict(python='/home/u202522802045/envs/scmomer_eclare_sintra/bin/python', entry='bmmc_eclare_intra.py', extra=[], device='cuda'),
 'scmomer': dict(python='/home/u202522802045/envs/scmomer_eclare_sintra/bin/python', entry='bmmc_scmomer_intra.py', extra=[], device='cuda'),
}


def read(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return {}

def write(p,v):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,indent=2)+'\n')

def status(v):
    write(OUT/'QUEUE_STATUS.json',dict(updated=time.strftime('%Y-%m-%dT%H:%M:%S'),**v))

def run(method,stage,site,seed):
    c=METHODS[method]; unit=OUT/method/('smoke' if stage=='smoke' else 'pre_final')/site/f'seed_{seed}'
    if stage=='smoke' and read(unit/'manifest.json').get('status')=='SMOKE_PASS': return 'SKIP'
    if stage!='smoke' and read(unit/'manifest.json').get('status')=='COMPLETE': return 'SKIP'
    if unit.exists(): return 'EXISTS_UNVERIFIED'
    unit.parent.mkdir(parents=True,exist_ok=True)
    log=unit.parent/f'seed_{seed}.log'
    cmd=[c['python'],str(RUN/c['entry']),*c['extra'],'--stage',stage,'--dataset',site,'--seed',str(seed),'--device',c['device']]
    print('RUN',method,stage,site,seed,flush=True)
    with log.open('ab') as f:
        f.write(('CMD '+' '.join(cmd)+'\n').encode())
        rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
    return f'RC={rc}'

def main():
    status(dict(state='SMOKE',completed=0,failed=0,current=None))
    smoke_fail=0
    for method in METHODS:
        for site in SITES:
            result=run(method,'smoke',site,0)
            if result not in ('SKIP',) and result!='RC=0': smoke_fail+=1
            status(dict(state='SMOKE',method=method,site=site,result=result,smoke_fail=smoke_fail))
    status(dict(state='FULL',completed=0,failed=0,smoke_fail=smoke_fail))
    complete=failed=0
    for method in METHODS:
        for site in SITES:
            for seed in SEEDS:
                result=run(method,'unit',site,seed)
                if result in ('SKIP','RC=0'): complete+=1
                else: failed+=1
                status(dict(state='FULL',method=method,site=site,seed=seed,result=result,
                    complete=complete,failed=failed,smoke_fail=smoke_fail))
    status(dict(state='COMPLETE',complete=complete,failed=failed,smoke_fail=smoke_fail,
        expected=len(METHODS)*len(SITES)*len(SEEDS)))

if __name__=='__main__':
    main()
