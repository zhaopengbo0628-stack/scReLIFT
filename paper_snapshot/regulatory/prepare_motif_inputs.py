import re, zipfile
from pathlib import Path
import numpy as np
from scipy import sparse

ROOT=Path('/eaas/default/groups/public_cluster/home/share/zpb')
BASE=ROOT/'result_contain/99_pbmc_lodo_dmode_20261024/full_b2/validated_source_artifacts'
OUT=Path('/home/u2021082/zpb/main/core/regulatory/pbmc_biology_20261027/motif')
FOLDS=['target_D1','target_D2','target_D3','target_D4']
def peak(x):
 m=re.match(r'^(chr[^:]+):(\d+)-(\d+)$',str(x)); return m.group(1),int(m.group(2)),int(m.group(3))
OUT.mkdir(parents=True,exist_ok=True)
for f in FOLDS:
 d=BASE/f; names=np.load(d/'harmonized/canonical_peak_names.npy',allow_pickle=True).astype(str)
 b=np.load(d/'B_D.npz'); mat=sparse.csr_matrix((b['data'],b['indices'],b['indptr']),shape=tuple(b['shape']))
 sel=np.unique(mat.tocoo().row)
 ep=np.load(d/'edge_peak.npy'); cand=np.unique(ep)
 for kind,idx in [('selected',sel),('candidate',cand)]:
  with open(OUT/f'{f}_{kind}.bed','w') as h:
   for i in idx:
    c,s,e=peak(names[int(i)]); h.write(f'{c}\t{s}\t{e}\t{f}_{kind}_{int(i)}\n')
 print(f, 'selected',len(sel),'candidate',len(cand))
