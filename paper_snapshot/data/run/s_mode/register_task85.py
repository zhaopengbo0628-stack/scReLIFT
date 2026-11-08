import csv, fcntl
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path('/home/u202522802045/share/zpb')
reg=ROOT/'result_contain/task_registry.csv'; lock=reg.with_suffix('.lock')
now=datetime.now(timezone.utc).isoformat()
with lock.open('a') as lk:
    fcntl.flock(lk,fcntl.LOCK_EX)
    with reg.open() as f:
        reader=csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
    ex=[r for r in rows if r['task_id']=='85']
    if ex:
        r=ex[0]; r['status']='COMPLETED_FOUR_DATASET_INTRA_READY'; r['updated_at']=now
    else:
        r=dict.fromkeys(fields,'')
        r.update(task_id='85',status='COMPLETED_FOUR_DATASET_INTRA_READY',experiment_name='85_brain_four_dataset_preprocessing',
            model_version='Mouse-brain four-dataset formal preprocessing',result_path=str(ROOT/'result_contain/85_brain_four_dataset_preprocessing'),
            run_script_path=str(ROOT/'data/run/s_mode/four_dataset_preflight.py'),launcher='direct_python_nohup',scheduler='none',
            launch_method='CPU preprocessing',created_at=now,updated_at=now,
            datasets='D3=GSE210747; D4=GSE229169; Gao=NeMO dat-5y9mf0h; FC=GSE294772',seeds='20020130')
        rows.append(r)
    tmp=reg.with_suffix('.register85.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fields); w.writeheader(); w.writerows(rows)
    tmp.replace(reg)
print('registered 85')
