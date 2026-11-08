import csv, fcntl
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path('/home/u202522802045/share/zpb')
reg=ROOT/'result_contain/task_registry.csv'; lock=reg.with_suffix('.lock'); now=datetime.now(timezone.utc).isoformat()
with lock.open('a') as lk:
    fcntl.flock(lk,fcntl.LOCK_EX)
    with reg.open() as f:
        rd=csv.DictReader(f); fields=rd.fieldnames; rows=list(rd)
    ex=[r for r in rows if r['task_id']=='86']
    if ex:
        ex[0]['status']='COMPLETED_LODO_BD_AUDIT'; ex[0]['updated_at']=now
    else:
        r=dict.fromkeys(fields,'')
        r.update(task_id='86',status='COMPLETED_LODO_BD_AUDIT',experiment_name='86_brain_lodo_bd_audit',
            model_version='Mouse-brain four-fold LODO B_D preprocessing audit',result_path=str(ROOT/'result_contain/86_brain_lodo_bd_audit'),
            run_script_path=str(ROOT/'data/run/d_mode/build_four_lodo.py'),launcher='direct_python_nohup',scheduler='none',
            launch_method='CPU preprocessing',created_at=now,updated_at=now,
            datasets='D3=GSE210747; D4=GSE229169; Gao=NeMO dat-5y9mf0h; FC=GSE294772',seeds='20020130')
        rows.append(r)
    tmp=reg.with_suffix('.register86.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fields); w.writeheader(); w.writerows(rows)
    tmp.replace(reg)
print('registered 86')
