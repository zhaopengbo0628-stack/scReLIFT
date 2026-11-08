import csv, fcntl
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path('/home/u202522802045/share/zpb')
reg=ROOT/'result_contain/task_registry.csv'; lock=reg.with_suffix('.lock'); now=datetime.now(timezone.utc).isoformat()
with lock.open('a') as lk:
    fcntl.flock(lk,fcntl.LOCK_EX)
    with reg.open() as f:
        rd=csv.DictReader(f); fields=rd.fieldnames; rows=list(rd)
    assert not [r for r in rows if r['task_id']=='89']
    r=dict.fromkeys(fields,'')
    r.update(task_id='89',status='COMPLETED_ISSAAC_READY_ZEPPILLI_AUTHOR_LABELS_BLOCKED',experiment_name='89_brain_two_new_sources',
        model_version='Two new mouse sources: ISSAAC + Zeppilli SSp',result_path=str(ROOT/'result_contain/89_brain_two_new_sources'),
        run_script_path=str(ROOT/'data/run/s_mode/build_issaac.py'),launcher='direct_python',scheduler='none',
        launch_method='CPU download/preprocessing',created_at=now,updated_at=now,
        datasets='E-MTAB-11264; GSE239477 SSp N1/N4/N6/N7',seeds='20020130')
    rows.append(r)
    tmp=reg.with_suffix('.register89.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fields); w.writeheader(); w.writerows(rows)
    tmp.replace(reg)
print('registered 89')
