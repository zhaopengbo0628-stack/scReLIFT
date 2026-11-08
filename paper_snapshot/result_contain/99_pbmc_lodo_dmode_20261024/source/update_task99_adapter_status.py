import csv, json
from pathlib import Path

base=Path('/home/u202522802045/share/zpb/result_contain/99_pbmc_lodo_dmode_20261024')
ready={'Multigrate','ECLARE','Palette-U','MultiVI'}
audit=base/'comparison_lodo_adapter_audit.json'
a=json.loads(audit.read_text())
for row in a['methods']:
    if row['method'] in ready:
        row.update(status='ADAPTER_READY_SMOKE_PASS',reason='source-only LODO adapter smoke passed on target_D1 seed_20020130; formal 4-fold x 3-seed queue not started',smoke_fold='target_D1',smoke_seed='20020130',target_atac_used=False,target_labels_before_embedding=False)
    elif row['method']=='MIDAS':
        row.update(status='BLOCKED_IMPLEMENTATION_ABSENT',reason='no MIDAS implementation or runnable environment found under /home/u202522802045/share/zpb; no result fabricated',target_atac_used=False,target_labels_before_embedding=False)
audit.write_text(json.dumps(a,indent=2,ensure_ascii=False)+'\n')
status=base/'status.json'; s=json.loads(status.read_text())
s.update(status='ADAPTERS_SMOKE_PASS_MIDAS_BLOCKED',updated='2026-10-25T06:30:00',current=None)
s['adapter_progress']={m:{'status':'ADAPTER_READY_SMOKE_PASS','smoke':'target_D1/20020130'} for m in ready}
s['adapter_progress']['MIDAS']={'status':'BLOCKED_IMPLEMENTATION_ABSENT','smoke':'not_run'}
s['pending_methods']=['Multigrate','ECLARE','Palette-U','MultiVI','MIDAS']
status.write_text(json.dumps(s,indent=2,ensure_ascii=False)+'\n')
rows=[]
with (base/'failed_runs.csv').open() as f:
    for row in csv.DictReader(f):
        if row['method'] in ready:
            row['status']='ADAPTER_READY_SMOKE_PASS'; row['reason']='strict source-only LODO adapter smoke passed on target_D1 seed 20020130; formal queue pending'
        else:
            row['status']='BLOCKED_IMPLEMENTATION_ABSENT'; row['reason']='MIDAS implementation/runnable environment absent on server; no result fabricated'
        rows.append(row)
with (base/'failed_runs.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['method','status','reason','target_atac_used','target_labels_before_embedding','seeds']); w.writeheader(); w.writerows(rows)
