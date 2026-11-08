import csv, fcntl
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path('/home/u202522802045/share/zpb')
reg=ROOT/'result_contain/task_registry.csv'; lock=reg.with_suffix('.lock'); now=datetime.now(timezone.utc).isoformat()
with lock.open('a') as lk:
    fcntl.flock(lk,fcntl.LOCK_EX)
    with reg.open() as f:
        rd=csv.DictReader(f); fields=rd.fieldnames; rows=list(rd)
    assert not any(r['task_id']=='97' for r in rows)
    r=dict.fromkeys(fields,'')
    r.update(task_id='97',status='COMPLETED_BMMC_B2FULL_RNAONLY_3SEED_INTRA',experiment_name='97_bmmc_b2full_rnaonly_3seed_intra_20261024',
        model_version='Full B2-S intra + RNA-only AE intra',result_path=str(ROOT/'result_contain/97_bmmc_b2full_rnaonly_3seed_intra_20261024'),
        run_script_path=str(ROOT/'data/run/s_mode/run_bmmc_intra.py'),launcher='direct_python',scheduler='none',
        launch_method='GPU A30; sequential Site x seed',created_at=now,updated_at=now,
        datasets='GSE194122 BMMC Sites 1-4',seeds='20020130,20020131,20020132')
    rows.append(r)
    tmp=reg.with_suffix('.register97.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fields); w.writeheader(); w.writerows(rows)
    tmp.replace(reg)
print('registered 97')
