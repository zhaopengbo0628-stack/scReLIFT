import hashlib, json
from pathlib import Path
ROOT=Path('/home/u202522802045/share/zpb')
BASE=ROOT/'result_contain/99_pbmc_lodo_dmode_20261024/full_b2/validated_source_artifacts'
OUT=ROOT/'result_contain/pbmc_dmode_regulatory_biology_20261026'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
rows=[]
for d in ['target_D1','target_D2','target_D3','target_D4']:
 m=json.loads((BASE/d/'manifest.json').read_text()); hashes=m['hashes']
 for name in ['B_D.npz','gene_ids.npy']:
  p=BASE/d/name; exp=hashes.get(str(p));
  rows.append(dict(fold=d,file=name,exists=p.exists(),sha256=sha(p) if p.exists() else '',manifest_sha256=exp or '',match=(exp==sha(p) if p.exists() and exp else False)))
 p=BASE/d/'harmonized/canonical_peak_names.npy'; exp=hashes.get(str(p))
 rows.append(dict(fold=d,file='canonical_peak_names.npy',exists=p.exists(),sha256=sha(p) if p.exists() else '',manifest_sha256=exp or '',match=(exp==sha(p) if p.exists() and exp else False)))
print('\n'.join(f'{r["fold"]} {r["file"]} match={r["match"]}' for r in rows))
(OUT/'reference_manifest/input_hash_audit.csv').write_text('fold,file,exists,sha256,manifest_sha256,match\n'+'\n'.join(','.join(str(r[k]) for k in ['fold','file','exists','sha256','manifest_sha256','match']) for r in rows)+'\n')
