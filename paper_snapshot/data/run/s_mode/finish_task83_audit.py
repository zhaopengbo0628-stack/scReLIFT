import json, numpy as np, pandas as pd
from pathlib import Path
ROOT=Path('/home/u202522802045/share/zpb')
BENCH=ROOT/'data/benchmark/Mouse-brain/GSE229169_intra_v1_20261023/GSE229169'
RESULT=ROOT/'result_contain/83_gse229169_fullb2_preprocessing'
meta=pd.read_csv(BENCH/'metadata.csv',keep_default_na=False)
splits={s:np.load(BENCH/'splits.npz')[s+'_index'] for s in ('train','val','test')}
rows=[]
for s,ix in splits.items():
    sub=meta.iloc[ix]
    for lib in ['Mop_2C_rep1','Mop_2C_rep2','Mop_3C_rep1','Mop_3C_rep2','Mop_4B_rep1','Mop_4B_rep2','Mop_5D_rep1','Mop_5D_rep2']:
        ss=sub[sub.library==lib]
        rows.append(dict(split=s,library=lib,cells=len(ss),subclasses=int(ss.subclass.nunique())))
    rows.append(dict(split=s,library='ALL',cells=len(sub),subclasses=int(sub.subclass.nunique())))
pd.DataFrame(rows).to_csv(RESULT/'SPLIT_SUMMARY.csv',index=False)
audit=json.load(open(RESULT/'HVG_SPARSITY_AUDIT.json'))
bf=pd.DataFrame([dict(status='BLOCKED_HVG_SPARSITY',reason='1373/2000 HVGs detected in <1% of train cells; median 21 nonzero HVGs/cell; 10 all-zero cells',
    canonical_peaks=286979,exact_union_peaks=1533271,hvg_selected=2000,hvg_lt_1pct=audit['hvg_detection_train']['lt_1pct'],
    median_hvg_per_cell=audit['hvg_per_cell']['median'],zero_hvg_cells=audit['hvg_per_cell']['zeros'],
    B_status='NOT_BUILT',F_status='NOT_BUILT',cache_created=False,
    detection_filtered_1pct_median_hvg_per_cell=audit['detection_filtered_alternative']['0.01']['per_cell_median'],
    detection_filtered_5pct_median_hvg_per_cell=audit['detection_filtered_alternative']['0.05']['per_cell_median'])])
bf.to_csv(RESULT/'BF_QUALITY_AUDIT.csv',index=False)
print('wrote split/BF audit')
