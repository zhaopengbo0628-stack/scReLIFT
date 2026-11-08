"""Original-count RNA interface: full-dictionary normalization, then feature selection."""
import numpy as np
import pandas as pd
import ad_runtime as a

LEGACY=a.ROOT/'result_contain/AD_Dmode_4Cohort_LODO_20261026_rev02/queue/ad_fourfold_prepare.py'
l=a.load_module(LEGACY,'count_readers')
P1=a.ROOT/'data/benchmark/AD/GSE214979/author_qc_v1_20261012_bundle/post_qc'
FEATURES=pd.read_csv(P1/'rna_features.tsv',sep='\t')
GENES=FEATURES.gene_id.astype(str).to_numpy()

def reader(name):
    meta=a.cells(name)
    if name=='GSE214979':return l.TenX(meta.count_file.iloc[0],meta.count_row.to_numpy(),np.load(P1/'rna_feature_index.npy'))
    if name=='GSE268609':
        _,var=l.read_obs(meta.count_file.iloc[0]);ids=var['id'].astype(str).to_numpy()
    elif name=='SEAAD_MTG':
        p=a.RESULTS/'AD_Dmode_Goal_20261024/rev02_three_source_seaad_holdout/preprocessing/seaad_paired_counts/rna_features.tsv'
        ids=pd.read_csv(p,sep='\t').gene_ids.astype(str).to_numpy()
    elif name=='GSE288063_GSE278576':
        parts=[]
        for _,sub in meta.groupby('library_id',sort=False):
            genes,_=l.tenx_features(sub.count_file.iloc[0]);assert np.array_equal(genes,GENES)
            parts.append(l.TenX(sub.count_file.iloc[0],sub.count_row.to_numpy(),np.arange(len(GENES))))
        return l.Concat(parts)
    else:raise ValueError(name)
    lookup={g:i for i,g in enumerate(ids)}
    return l.H5AD(meta.count_file.iloc[0],meta.count_row.to_numpy(),np.asarray([lookup[g] for g in GENES]))

def normalized_blocks(name,selected=None,log=True):
    """Denominator always includes the full aligned 36601 genes."""
    from scipy import sparse
    for first,m in reader(name).blocks():
        total=np.asarray(m.sum(1)).ravel()
        z=sparse.diags(10000/np.maximum(total,1)).dot(m).tocsr()
        if log:z.data=np.log1p(z.data)
        if selected is not None:z=z[:,selected]
        assert np.isfinite(z.data).all() and (z.data>=0).all()
        yield first,z

def selected_rows(name,rows,features):
    rows=np.asarray(rows);assert np.all(np.diff(rows)>0)
    out=np.empty((len(rows),len(features)),np.float32);filled=0
    for first,z in normalized_blocks(name,features):
        lo,hi=np.searchsorted(rows,[first,first+z.shape[0]])
        if hi>lo:out[lo:hi]=z[rows[lo:hi]-first].toarray();filled+=hi-lo
    assert filled==len(rows) and np.isfinite(out).all()
    return out
