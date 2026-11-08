import csv, fcntl
from pathlib import Path
ROOT=Path('/home/u202522802045/share/zpb')
reg=ROOT/'result_contain/task_registry.csv'; lock=reg.with_suffix('.lock')
with lock.open('a') as lk:
    fcntl.flock(lk,fcntl.LOCK_EX)
    with reg.open() as f:
        rd=csv.DictReader(f); fields=rd.fieldnames; rows=list(rd)
    for r in rows:
        if r['task_id']=='86' and r['experiment_name']=='86_mousebrain_b2_rnaonly_intra_1seed_20261023':
            r['status']='RUNNING'
            r['updated_at']=r['created_at']
    tmp=reg.with_suffix('.restore86.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fields); w.writeheader(); w.writerows(rows)
    tmp.replace(reg)
print('restored 86')
