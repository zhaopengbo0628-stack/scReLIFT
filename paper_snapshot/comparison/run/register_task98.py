import csv, fcntl
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path('/home/u202522802045/share/zpb')
reg=ROOT/'result_contain/task_registry.csv'; lock=reg.with_suffix('.lock'); now=datetime.now(timezone.utc).isoformat()
with lock.open('a') as lk:
    fcntl.flock(lk,fcntl.LOCK_EX)
    with reg.open() as f:
        rd=csv.DictReader(f); fields=rd.fieldnames; rows=list(rd)
    assert not any(r['task_id']=='98' for r in rows)
    r=dict.fromkeys(fields,'')
    r.update(task_id='98',status='RUNNING_BMMC_INTRA_5METHODS_3SEED',experiment_name='BMMC_intra_5methods_3seed',
        model_version='BMMC Site intra: Multigrate, Cobolt, Palette-U, ECLARE-S, scMomer-S',
        result_path=str(ROOT/'comparison/result/BMMC_intra'),run_script_path=str(ROOT/'comparison/run/bmmc_intra_queue.py'),
        launcher='detached_python_queue',scheduler='sequential GPU/CPU',launch_method='setsid',
        created_at=now,updated_at=now,datasets='BMMC_Site1;BMMC_Site2;BMMC_Site3;BMMC_Site4',seeds='0,1,2')
    rows.append(r)
    tmp=reg.with_suffix('.register98.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fields); w.writeheader(); w.writerows(rows)
    tmp.replace(reg)
print('registered 98')
