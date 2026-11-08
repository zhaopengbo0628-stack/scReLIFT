"""Cross-fold structural consistency from exported B_D edges."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('/home/u202522802045/share/zpb/result_contain/pbmc_dmode_regulatory_biology_20261026')
FOLDS=['target_D1','target_D2','target_D3','target_D4']
edges={}
for fold in FOLDS:
    d=pd.read_csv(OUT/'edges'/f'{fold}_edges.tsv.gz',sep='\t')
    d['key']=list(zip(d.gene_id,d.peak_chr,d.peak_start,d.peak_end))
    edges[fold]=d.set_index('key')
pairs=[]
for i,a in enumerate(FOLDS):
    for b in FOLDS[i+1:]:
        A,B=edges[a],edges[b]; common=A.index.intersection(B.index)
        inter=set(common); union=set(A.index)|set(B.index)
        pairs.append(dict(fold_a=a,fold_b=b,edges_a=len(A),edges_b=len(B),common_edges=len(common),
            jaccard=len(inter)/len(union),precision=len(inter)/len(A),recall=len(inter)/len(B),
            sign_consistency=float(np.mean(A.loc[common,'sign']==B.loc[common,'sign'])) if len(common) else float('nan'),
            weight_spearman=float(stats.spearmanr(A.loc[common,'weight'],B.loc[common,'weight']).statistic) if len(common)>2 else float('nan')))
pd.DataFrame(pairs).to_csv(OUT/'stability/fold_pairwise_overlap.csv',index=False)
allkeys=set().union(*[set(e.index) for e in edges.values()])
cons=pd.DataFrame([dict(gene_id=k[0],peak_chr=k[1],peak_start=k[2],peak_end=k[3],n_folds=sum(k in edges[f].index for f in FOLDS),
    folds=','.join(f for f in FOLDS if k in edges[f].index)) for k in allkeys])
cons.to_csv(OUT/'stability/four_fold_consensus.csv',index=False)
mat=pd.DataFrame(np.eye(4),index=FOLDS,columns=FOLDS)
for r in pairs: mat.loc[r['fold_a'],r['fold_b']]=mat.loc[r['fold_b'],r['fold_a']]=r['jaccard']
fig,ax=plt.subplots(figsize=(3.4,3)); im=ax.imshow(mat.to_numpy(),vmin=0,vmax=max(.01,mat.to_numpy().max()),cmap='Blues')
ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels([x.replace('target_','') for x in FOLDS]); ax.set_yticklabels([x.replace('target_','') for x in FOLDS])
for i in range(4):
    for j in range(4): ax.text(j,i,f'{mat.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8)
fig.colorbar(im,ax=ax,label='Jaccard'); fig.tight_layout()
for ext in ('svg','png'): fig.savefig(OUT/'figures'/f'fold_stability_heatmap.{ext}',dpi=300,bbox_inches='tight')
print(pd.DataFrame(pairs).to_string(index=False)); print('consensus',len(cons))
