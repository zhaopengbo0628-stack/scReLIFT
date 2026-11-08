import json, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
ROOT=Path('/home/u202522802045/share/zpb')
OUT=ROOT/'result_contain/85_brain_four_dataset_preprocessing'
D3=ROOT/'data/cache/s_mode/Mouse-brain/full_b2_intra_v1_20261022/D3'
D4=ROOT/'data/cache/s_mode/Mouse-brain/GSE229169_intra_v1_20261023/GSE229169'
GAO=ROOT/'data/cache/s_mode/Mouse-brain/adult_cortex_author_processed_intra_pre_final_20261023/GaoVIS'
FC=ROOT/'data/cache/s_mode/Mouse-brain/FC_author_labeled_v1_20261023/FC'
DS={
 'D3':dict(cache=D3,meta=ROOT/'data/benchmark/Mouse-brain/b2_dmode_cohort_v1_20261022/D3/metadata.csv',label='reference_label',lib='author_sample_id',accession='GSE210747'),
 'D4':dict(cache=D4,meta=ROOT/'data/benchmark/Mouse-brain/GSE229169_intra_v1_20261023/GSE229169/metadata.csv',label='subclass',lib='library',accession='GSE229169'),
 'Gao':dict(cache=GAO,meta=ROOT/'data/benchmark/Mouse-brain/adult_cortex_author_processed_intra_pre_final_20261023/GaoVIS/metadata.csv',label='subclass_label',lib='library_prep',accession='NeMO dat-5y9mf0h'),
 'FC':dict(cache=FC,meta=ROOT/'data/benchmark/Mouse-brain/FC_author_labeled_v1_20261023/FC/metadata.csv',label='reference_label',lib='library',accession='GSE294772')}
OUT.mkdir(parents=True,exist_ok=True)
rows=[]; hvg_rows=[]; split_rows=[]; bf_rows=[]; details={}
for name,d in DS.items():
    c=d['cache']; meta=pd.read_csv(d['meta'],keep_default_na=False)
    tr=np.load(c/'train_index.npy'); va=np.load(c/'val_index.npy'); te=np.load(c/'test_index.npy')
    peaks=np.load(c/'peak_names.npy',allow_pickle=True); bs=sparse.load_npz(c/'B_S.npz')
    rtr=np.load(c/'rna_train.npy'); rva=np.load(c/'rna_val.npy'); rte=np.load(c/'rna_test.npy')
    ftr=np.load(c/'F_train.npy'); fva=np.load(c/'F_val.npy')
    with np.load(c/'field_state.npz') as z: fstd=z['field_std']
    assert len(meta)==len(tr)+len(va)+len(te)
    assert bs.shape==(len(peaks),2000), (name,bs.shape,len(peaks))
    assert ftr.shape==(len(tr),2000) and fva.shape==(len(va),2000)
    assert rtr.shape==(len(tr),2000) and rva.shape==(len(va),2000) and rte.shape==(len(te),2000)
    assert np.isfinite(rtr).all() and np.isfinite(ftr).all() and np.isfinite(bs.data).all()
    labels=meta[d['label']].fillna('').astype(str)
    cov=float((labels!='').mean())
    nclass=int(labels[labels!=''].nunique())
    nlib=int(meta[d['lib']].nunique())
    det=np.asarray((rtr>0).mean(0)).ravel(); per=np.asarray((rtr>0).sum(1)).ravel(); perte=np.asarray((rte>0).sum(1)).ravel()
    # minibatch
    mb=dict(rna_shape=list(rtr[:8].shape),f_shape=list(ftr[:8].shape),b_shape=list(bs[:,:8].shape),
        rna_finite=bool(np.isfinite(rtr[:8]).all()),f_finite=bool(np.isfinite(ftr[:8]).all()),b_finite=bool(np.isfinite(bs[:,:8].data).all()))
    rows.append(dict(dataset=name,accession=d['accession'],cells=len(meta),label_coverage=cov,classes=nclass,libraries=nlib,
        peaks=len(peaks),hvg=2000,B_edges=int(bs.nnz),F_nonzero=float((ftr!=0).mean()),F_zero_std=int((fstd<1e-6).sum()),
        ready=True,minibatch=json.dumps(mb)))
    hvg_rows.append(dict(dataset=name,hvg=2000,median_detection=float(np.median(det)),lt_1pct=int((det<0.01).sum()),
        median_nonzero_train=float(np.median(per)),zero_train=int((per==0).sum()),zero_test=int((perte==0).sum()),
        min_nonzero=int(per.min()),max_nonzero=int(per.max())))
    for split,idx in [('train',tr),('val',va),('test',te)]:
        sub=meta.iloc[idx]; labs=sub[d['label']].fillna('').astype(str)
        split_rows.append(dict(dataset=name,split=split,cells=len(idx),classes=int(labs[labs!=''].nunique()),label_coverage=float((labs!='').mean()),
            libraries=int(sub[d['lib']].nunique())))
    bf_rows.append(dict(dataset=name,peaks=len(peaks),hvg=2000,candidate_edges=None,B_edges=int(bs.nnz),B_retained_genes=int((np.asarray(abs(bs).sum(0)).ravel()>0).sum()),
        F_shape=f'{ftr.shape[0]+fva.shape[0]} x 2000',F_nonzero=float((ftr!=0).mean()),F_zero_std=int((fstd<1e-6).sum())))
    details[name]=dict(minibatch=mb,hvg=dict(median_detection=float(np.median(det)),lt_1pct=int((det<0.01).sum()),median_nonzero=float(np.median(per)),zero_train=int((per==0).sum()),zero_test=int((perte==0).sum())))
pd.DataFrame(rows).to_csv(OUT/'FOUR_DATASET_FINAL_COUNTS.csv',index=False)
pd.DataFrame(hvg_rows).to_csv(OUT/'FOUR_DATASET_HVG_AUDIT.csv',index=False)
pd.DataFrame(split_rows).to_csv(OUT/'FOUR_DATASET_SPLIT_AUDIT.csv',index=False)
pd.DataFrame(bf_rows).to_csv(OUT/'FOUR_DATASET_BF_QUALITY.csv',index=False)
(OUT/'FOUR_DATASET_PREFLIGHT.json').write_text(json.dumps(details,indent=2)+'\n')
print(json.dumps(details,indent=2))
print(pd.DataFrame(rows).to_string(index=False))
