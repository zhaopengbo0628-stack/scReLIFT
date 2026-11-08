import json
from pathlib import Path
base=Path('/home/u202522802045/share/zpb/result_contain/99_pbmc_lodo_dmode_20261024')
p=base/'status.json'
s=json.loads(p.read_text())
s.update(status='FORMAL_QUEUE_RUNNING',updated='2026-10-25T22:30:00',current='MultiVI target_D1 seed_20020131',formal_completed_units=37,formal_planned_comparator_units=48,formal_failed_units=0)
p.write_text(json.dumps(s,indent=2,ensure_ascii=False)+'\n')
