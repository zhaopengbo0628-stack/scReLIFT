"""Minimal real-data analyses for PBMC D-mode regulatory biology."""
import json, gzip
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/home/u202522802045/share/zpb')
OUT=ROOT/'result_contain/pbmc_dmode_regulatory_biology_20261026'
REF=ROOT/'data/reference/external_regulatory/ENCODE_SCREEN_V4/hg38'
BASE=ROOT/'result_contain/99_pbmc_lodo_dmode_20261024/full_b2/validated_source_artifacts'
FOLDS=['target_D1','target_D2','target_D3','target_D4']
CCRE={'PLS':'GRCh38-cCREs.PLS.bed','pELS':'GRCh38-cCREs.pELS.bed','dELS':'GRCh38-cCREs.dELS.bed'}
SEED=20020130

def intervals(path):
    if not path.exists(): return None
    df=pd.read_csv(path,sep='\t',header=None,usecols=[0,1,2],names=['chr','start','end'])
    d={}
    for c,g in df.groupby('chr'):
        g=g.sort_values('start'); d[c]=(g.start.to_numpy(np.int64),g.end.to_numpy(np.int64))
    return d

def overlaps(peaks, ref):
    if ref is None: return None
    hit=np.zeros(len(peaks),bool)
    for c,(st,en) in ref.items():
        idx=np.where(peaks.chr.to_numpy()==c)[0]
        if not len(idx): continue
        pst=peaks.start.to_numpy()[idx]; pen=peaks.end.to_numpy()[idx]
        lo=np.searchsorted(st,pen,side='left')-1
        ok=lo>=0
        ok[ok]&=en[lo[ok]]>pst[ok]
        hit[idx]=ok
    return hit

def stats_null(obs, null):
    null=np.asarray(null,float); p=(1+np.sum(null>=obs))/(len(null)+1)
    return dict(observed=float(obs),null_mean=float(null.mean()),null_sd=float(null.std(ddof=1)),
                fold_change=float((obs+1e-6)/(null.mean()+1e-6)),p_empirical=float(p),
                ci_low=float(np.percentile(null,2.5)),ci_high=float(np.percentile(null,97.5)))

peaks_all={}
for fold in FOLDS:
    names=pd.Series(np.load(BASE/fold/'harmonized/canonical_peak_names.npy',allow_pickle=True).astype(str))
    parts=names.str.extract(r'^(chr[^:]+):(\d+)-(\d+)$')
    peaks_all[fold]=pd.DataFrame(dict(peak_index=np.arange(len(names)),chr=parts[0],start=parts[1].astype(np.int64),end=parts[2].astype(np.int64)))
cc={k:intervals(REF/v) for k,v in CCRE.items()}
rows=[]; nullrows=[]
for fold in FOLDS:
    df=pd.read_csv(OUT/'edges'/f'{fold}_edges.tsv.gz',sep='\t')
    peaks=peaks_all[fold].reset_index(drop=True)
    sel=peaks.peak_index.isin(df.peak_index.unique()).to_numpy()
    sel_idx=np.where(sel)[0]
    # all peaks are those with edges; null universe is all B rows not directly available here; use peaks in all edge rows
    universe=peaks
    rng=np.random.default_rng(SEED)
    for label,ref in cc.items():
        if ref is None:
            rows.append(dict(fold=fold,cCRE=label,peaks_total=len(peaks),peaks_with_edges=int(sel.sum()),edges=len(df),
                peak_overlap=float('nan'),edge_overlap=float('nan'),null_mean=float('nan'),fold_change=float('nan'),p_empirical=float('nan')))
            continue
        hit=overlaps(peaks,ref); assert hit is not None
        obs_peak=float(hit[sel_idx].mean())
        edge_peak_idx=df.peak_index.to_numpy(); edge_hit=hit[edge_peak_idx]; obs_edge=float(edge_hit.mean())
        null_peak=[]; null_edge=[]
        for _ in range(200):
            ridx=rng.choice(len(peaks),size=len(sel_idx),replace=False)
            null_peak.append(float(hit[ridx].mean()))
            eidx=rng.choice(len(df),size=len(df),replace=False)
            null_edge.append(float(hit[df.peak_index.to_numpy()[eidx]].mean()))
        sp=stats_null(obs_peak,null_peak); se=stats_null(obs_edge,null_edge)
        rows.append(dict(fold=fold,cCRE=label,peaks_total=len(peaks),peaks_with_edges=int(sel.sum()),edges=len(df),
            peak_overlap=obs_peak,edge_overlap=obs_edge,peak_null_mean=sp['null_mean'],peak_fold_change=sp['fold_change'],peak_p=sp['p_empirical'],
            edge_null_mean=se['null_mean'],edge_fold_change=se['fold_change'],edge_p=se['p_empirical']))
        nullrows += [dict(fold=fold,cCRE=label,level='peak',value=v) for v in null_peak]
        nullrows += [dict(fold=fold,cCRE=label,level='edge',value=v) for v in null_edge]
pd.DataFrame(rows).to_csv(OUT/'enhancer/overlap_per_fold.csv',index=False)
pd.DataFrame(nullrows).to_csv(OUT/'enhancer/matched_null_controls.csv',index=False)
# simple figure
plot=pd.DataFrame(rows); plot=plot[plot.peak_overlap.notna()]
if len(plot):
    fig,ax=plt.subplots(figsize=(7,3.5))
    x=np.arange(len(plot)); ax.bar(x,plot.peak_overlap,color='#3b6ea5',label='observed')
    ax.scatter(x,plot.peak_null_mean,color='black',marker='_',s=120,label='matched null mean')
    ax.set_xticks(x); ax.set_xticklabels([f'{r.fold}\n{r.cCRE}' for r in plot.itertuples()],rotation=45,ha='right',fontsize=7)
    ax.set_ylabel('peak overlap fraction'); ax.legend(frameon=False,fontsize=7); fig.tight_layout()
    for ext in ('svg','png'): fig.savefig(OUT/'figures'/f'enhancer_enrichment.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
print('ENHANCER_DONE',flush=True)

# stability: exact peak coordinate + gene match
edges={}
for fold in FOLDS:
    d=pd.read_csv(OUT/'edges'/f'{fold}_edges.tsv.gz',sep='\t')
    d['key']=list(zip(d.gene_id,d.peak_chr,d.peak_start,d.peak_end))
    edges[fold]=d.set_index('key')
pairs=[]; signrows=[]
for i,a in enumerate(FOLDS):
    for b in FOLDS[i+1:]:
        A=edges[a]; B=edges[b]; common=A.index.intersection(B.index)
        inter=set(common); union=set(A.index)|set(B.index)
        jac=len(inter)/len(union) if union else float('nan')
        prec=len(inter)/len(A) if len(A) else float('nan'); rec=len(inter)/len(B) if len(B) else float('nan')
        sign=np.mean(A.loc[common,'sign'].to_numpy()==B.loc[common,'sign'].to_numpy()) if len(common) else float('nan')
        corr=stats.spearmanr(A.loc[common,'weight'],B.loc[common,'weight']).statistic if len(common)>2 else float('nan')
        pairs.append(dict(fold_a=a,fold_b=b,edges_a=len(A),edges_b=len(B),common_edges=len(common),jaccard=jac,precision=prec,recall=rec,sign_consistency=sign,weight_spearman=corr))
pd.DataFrame(pairs).to_csv(OUT/'stability/fold_pairwise_overlap.csv',index=False)
# consensus presence
allkeys=set().union(*[set(e.index) for e in edges.values()])
cons=[]
for k in allkeys:
    present=[f for f in FOLDS if k in edges[f]]
    cons.append(dict(gene_id=k[0],peak_chr=k[1],peak_start=k[2],peak_end=k[3],n_folds=len(present),folds=','.join(present)))
cons=pd.DataFrame(cons); cons.to_csv(OUT/'stability/four_fold_consensus.csv',index=False)
# heatmap
mat=pd.DataFrame(np.eye(4),index=FOLDS,columns=FOLDS)
for r in pairs: mat.loc[r['fold_a'],r['fold_b']]=mat.loc[r['fold_b'],r['fold_a']]=r['jaccard']
fig,ax=plt.subplots(figsize=(3.5,3)); im=ax.imshow(mat.to_numpy(),vmin=0,vmax=max(.01,mat.to_numpy().max()),cmap='Blues')
ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels([f.replace('target_','') for f in FOLDS]); ax.set_yticklabels([f.replace('target_','') for f in FOLDS])
for i in range(4):
    for j in range(4): ax.text(j,i,f'{mat.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8)
fig.colorbar(im,ax=ax,label='Jaccard'); fig.tight_layout()
for ext in ('svg','png'): fig.savefig(OUT/'figures'/f'fold_stability_heatmap.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)
print('STABILITY_DONE',flush=True)
