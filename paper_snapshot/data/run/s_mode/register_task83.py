import csv, fcntl, os
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path('/home/u202522802045/share/zpb')
reg=ROOT/'result_contain/task_registry.csv'
lock=reg.with_suffix('.lock')
now=datetime.now(timezone.utc).isoformat()
with lock.open('a') as lk:
    fcntl.flock(lk,fcntl.LOCK_EX)
    with reg.open() as f:
        reader=csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
    existing=[r for r in rows if r['task_id']=='83']
    if existing:
        r=existing[0]; r['status']='BLOCKED_HVG_SPARSITY'; r['updated_at']=now
    else:
        r=dict.fromkeys(fields,'')
        r.update(task_id='83',status='BLOCKED_HVG_SPARSITY',experiment_name='83_gse229169_fullb2_preprocessing',
            model_version='Full B2 S-mode intra preprocessing',result_path=str(ROOT/'result_contain/83_gse229169_fullb2_preprocessing'),
            run_script_path=str(ROOT/'data/run/s_mode/prepare_gse229169_intra.py'),
            stdout_path=str(ROOT/'data/audit/Mouse-brain/gse229169_preprocessing_20261023/prepare_cache.log'),
            launcher='direct_python_nohup',scheduler='none',launch_method='CPU preprocessing',
            created_at=now,updated_at=now,datasets='GSE229169 P56 male MOp 47404 cells 8 libraries',seeds='20020130')
        rows.append(r)
    tmp=reg.with_suffix('.register83.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fields); w.writeheader(); w.writerows(rows)
    tmp.replace(reg)
print('registered 83')
