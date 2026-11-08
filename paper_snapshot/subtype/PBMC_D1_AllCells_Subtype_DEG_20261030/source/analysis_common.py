import hashlib,json,re,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse,stats

RUN=Path(__file__).resolve().parents[1]
BASE=RUN.parent/'PBMC_D1_AllCells_Dmode_20261030'
SHARED=Path('/home/u202522802045/share/zpb')
SEEDS=[20020130,20020131,20020132]
METHODS=['FullB2','PCA_sourcefit','PCA_targetfit_fullnorm']
sys.path.insert(0,str(RUN/'source'))
import frozen_task99_analysis as frozen
bh=frozen.bh
def slug(s):return re.sub('[^A-Za-z0-9]+','_',s).strip('_')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def context():
    cells=np.asarray((BASE/'artifacts/D1_barcodes.tsv').read_text().splitlines())
    labels=pd.read_csv(BASE/'audit/D1_allcell_labels.csv',index_col=0).loc[cells,'cell_type'].astype(str).to_numpy()
    assert len(cells)==9347 and len(set(cells))==9347
    return cells,labels
def partition(row,cl,labels,cells):
    sel=(labels==row['reference_cell_type'])&np.isin(cl,[row['cluster_a'],row['cluster_b']])
    return pd.Series(np.where(cl[sel]==row['cluster_a'],'A','B'),index=cells[sel])
def align(a,b):
    scores=[]
    for swap in [False,True]:
        c=b.map({'A':'B','B':'A'}) if swap else b
        score=0
        for g in ['A','B']:
            aa=set(a[a==g].index);bb=set(c[c==g].index)
            score+=len(aa&bb)/max(1,len(aa|bb))
        scores.append(score)
    return (b.map({'A':'B','B':'A'}) if scores[1]>scores[0] else b),max(scores)/2
