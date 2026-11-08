import hashlib, json
from pathlib import Path
OUT=Path('/home/u202522802045/share/zpb/result_contain/89_brain_two_new_sources')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
files=[p for p in OUT.iterdir() if p.is_file() and p.name!='processing_manifest.json']
man=dict(task_id=89,task_name='89_brain_two_new_sources',status='COMPLETED_ISSAAC_READY_ZEPPILLI_AUTHOR_LABELS_BLOCKED',
    created_at='2026-10-24T00:15:00+08:00',server='zyh-new (10.1.40.252:21872, u202522802045)',
    project_root='/home/u202522802045/share/zpb',result_dir=str(OUT),training_started=False,
    datasets=dict(
      ISSAAC=dict(status='READY',accession='E-MTAB-11264',cells=10361,classes=23,libraries=2,peaks=169180,
        cache='/home/u202522802045/share/zpb/data/cache/s_mode/Mouse-brain/ISSAAC_E-MTAB-11264_v1_20261023/ISSAAC'),
      ZeppilliSSp=dict(status='BLOCKED_AUTHOR_LABELS',accession='GSE239477',cells=1872,classes=0,libraries=4,peaks=180604,
        cache=None)),
    outputs={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files},files=sorted(p.name for p in files))
(OUT/'processing_manifest.json').write_text(json.dumps(man,indent=2)+'\n',encoding='utf-8')
print('wrote',len(files),'outputs')
