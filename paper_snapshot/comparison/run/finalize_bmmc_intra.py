import csv, fcntl, json
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path('/home/u202522802045/share/zpb')
now=datetime.now(timezone.utc).isoformat()
p=ROOT/'comparison/result/BMMC_intra/QUEUE_STATUS.json'
p.write_text(json.dumps(dict(state='COMPLETED_4METHODS_BY_USER',complete=48,failed=0,
    excluded=['scMomer-S'],methods=['Multigrate','Cobolt','Palette-U','ECLARE-S'],
    sites=['BMMC_Site1','BMMC_Site2','BMMC_Site3','BMMC_Site4'],seeds=[0,1,2],updated=now),indent=2)+'\n')
reg=ROOT/'result_contain/task_registry.csv'; lock=reg.with_suffix('.lock')
with lock.open('a') as lk:
    fcntl.flock(lk,fcntl.LOCK_EX)
    with reg.open() as f:
        rd=csv.DictReader(f); fields=rd.fieldnames; rows=list(rd)
    for r in rows:
        if r['task_id']=='98':
            r['status']='COMPLETED_4METHODS_USER_EXCLUDED_SCMOMER'; r['updated_at']=now
    tmp=reg.with_suffix('.update98.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fields); w.writeheader(); w.writerows(rows)
    tmp.replace(reg)
print('updated')
