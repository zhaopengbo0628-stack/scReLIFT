import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd, h5py, anndata as ad
from scipy import sparse
ROOT=Path('/home/u202522802045/share/zpb')
RAW=ROOT/'data/raw/BMMC/GSE194122'
H5=RAW/'GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad'
OUT=ROOT/'result_contain/96_bmmc_multiome_recovery'
OUT.mkdir(parents=True,exist_ok=True); (OUT/'site_indices').mkdir(exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(32<<20),b''): h.update(b)
    return h.hexdigest()
a=ad.read_h5ad(H5,backed='r')
obs=a.obs.copy(); obs.index=obs.index.astype(str)
var=a.var.copy(); ft=var['feature_types'].astype(str).to_numpy()
rna_mask=ft=='GEX'; atac_mask=ft=='ATAC'
assert rna_mask.sum()>0 and atac_mask.sum()>0
with h5py.File(H5,'r') as f:
    g=f['layers/counts']; shape=tuple(int(x) for x in g.attrs['shape']); enc=g.attrs['encoding-type']
    data=g['data'][:]; indices=g['indices'][:]; indptr=g['indptr'][:]
M=sparse.csr_matrix((data,indices,indptr),shape=shape)
counts_integer=bool(np.all(data>=0) and np.all(np.abs(data-np.round(data))<1e-6))
counts_min=float(data.min()); counts_max=float(data.max())
rna=M[:,rna_mask]; atac=M[:,atac_mask]
rna_total=np.asarray(rna.sum(axis=1)).ravel(); rna_det=np.asarray(rna.getnnz(axis=1)).ravel()
atac_total=np.asarray(atac.sum(axis=1)).ravel(); atac_det=np.asarray(atac.getnnz(axis=1)).ravel()
rna_nnz=int(rna.nnz); atac_nnz=int(atac.nnz)
del rna,atac,M
site_raw=obs['Site'].astype(str).to_numpy()
site=np.array(['BMMC_'+s.capitalize().replace('Site','Site') for s in site_raw])
# normalize site1 -> BMMC_Site1
site=np.array(['BMMC_Site'+s.replace('site','') for s in site_raw])
ct=obs['cell_type'].astype(str).to_numpy()
batch=obs['batch'].astype(str).to_numpy(); donor=obs['DonorID'].astype(str).to_numpy()
donor_num=obs['DonorNumber'].astype(str).to_numpy(); sample_name=obs['Samplename'].astype(str).to_numpy()
cell_id=obs.index.to_numpy()
meta=pd.DataFrame(dict(cell_id=cell_id,cell_type=ct,Site=site,batch=batch,DonorID=donor,DonorNumber=donor_num,
    Samplename=sample_name,RNA_total=rna_total,RNA_detected=rna_det,ATAC_total=atac_total,ATAC_detected=atac_det))
meta.to_csv(OUT/'CELL_METADATA.csv',index=False)
# label audit
label_rows=[]
for c,n in pd.Series(ct).value_counts().items():
    m=ct==c; label_rows.append(dict(cell_type=c,n_cells=int(n),site_count=int(pd.Series(site[m]).nunique()),
        donor_count=int(pd.Series(donor[m]).nunique()),fraction=float(n/len(ct))))
label_rows.append(dict(cell_type='__MISSING__',n_cells=int(pd.isna(obs['cell_type']).sum()+
    pd.Series(ct).isin(['','nan','None']).sum()),site_count=0,donor_count=0,fraction=0.0))
low=pd.Series(ct).str.lower()
label_rows.append(dict(cell_type='__DOUBLET_LIKE__',n_cells=int(low.str.contains('doublet').sum()),site_count=0,donor_count=0,fraction=0.0))
label_rows.append(dict(cell_type='__UNASSIGNED__',n_cells=int(low.isin(['unknown','unassigned','na','n/a']).sum()),site_count=0,donor_count=0,fraction=0.0))
label_rows.append(dict(cell_type='__DUPLICATE_CELL_ID__',n_cells=int(pd.Series(cell_id).duplicated().sum()),site_count=0,donor_count=0,fraction=0.0))
pd.DataFrame(label_rows).to_csv(OUT/'AUTHOR_LABEL_AUDIT.csv',index=False)
# site counts
rows=[]
for s in sorted(set(site)):
    m=site==s
    rows.append(dict(site=s,n_cells=int(m.sum()),n_donors=int(pd.Series(donor[m]).nunique()),n_cell_types=int(pd.Series(ct[m]).nunique()),
        RNA_total_median=float(np.median(rna_total[m])),RNA_detected_median=float(np.median(rna_det[m])),
        ATAC_total_median=float(np.median(atac_total[m])),ATAC_detected_median=float(np.median(atac_det[m])),
        RNA_zero_cells=int((rna_total[m]==0).sum()),ATAC_zero_cells=int((atac_total[m]==0).sum()),
        all_zero_cells=int(((rna_total[m]==0)&(atac_total[m]==0)).sum()),
        genes=int(rna_mask.sum()),peaks=int(atac_mask.sum())))
pd.DataFrame(rows).to_csv(OUT/'SITE_CELL_COUNTS.csv',index=False)
# distributions
dist=meta.groupby(['Site','cell_type']).size().rename('n_cells').reset_index()
dist['fraction']=dist.n_cells/dist.groupby('Site').n_cells.transform('sum')
dist.to_csv(OUT/'SITE_CELLTYPE_DISTRIBUTION.csv',index=False)
dd=meta.groupby(['Site','DonorID','DonorNumber']).size().rename('n_cells').reset_index()
dd.to_csv(OUT/'SITE_DONOR_DISTRIBUTION.csv',index=False)
# overlap
ov=meta.pivot_table(index=['DonorID','DonorNumber'],columns='Site',values='cell_id',aggfunc='count',fill_value=0)
ov['n_sites']=(ov[sorted(set(site))]>0).sum(axis=1); ov['total']=ov[sorted(set(site))].sum(axis=1)
ov.reset_index().to_csv(OUT/'DONOR_OVERLAP_MATRIX.csv',index=False)
# site indexes
sites={}
for s in sorted(set(site)):
    idx=np.where(site==s)[0].astype(np.int32); np.save(OUT/'site_indices'/f'{s}.npy',idx)
    sites[s]=dict(n_cells=int(len(idx)),row_indices_file=f'site_indices/{s}.npy')
mod=dict(dataset='GSE194122',h5ad=str(H5),shape=[int(x) for x in a.shape],obs_names_unique=bool(pd.Series(cell_id).is_unique),
    feature_types={'GEX':int(rna_mask.sum()),'ATAC':int(atac_mask.sum())},counts_layer='layers/counts',counts_encoding=enc,
    counts_integer_sample_only=counts_integer,counts_sample_min=counts_min,counts_sample_max=counts_max,
    RNA_nnz=rna_nnz,ATAC_nnz=atac_nnz,RNA_zero_cells=int((rna_total==0).sum()),ATAC_zero_cells=int((atac_total==0).sum()),
    all_zero_cells=int(((rna_total==0)&(atac_total==0)).sum()),paired_cells=int(a.n_obs),modalities=sorted(set(obs['Modality'].astype(str))),
    X_shape=list(shape),X_encoding='csr_matrix',layers=list(a.layers.keys()))
(OUT/'MODALITY_AUDIT.json').write_text(json.dumps(mod,indent=2)+'\n')
man=dict(dataset='GSE194122',file=H5.name,compressed_file=str(RAW/'GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz'),
    url='https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz',
    content_length=2917117242,last_modified='Tue, 01 Feb 2022 20:14:15 GMT',compressed_sha256=sha(RAW/'GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz'),
    h5ad_bytes=H5.stat().st_size,h5ad_sha256=sha(H5),gzip_ok=True,source='official GEO; downloaded locally with curl after server-side NCBI 503, then scp to server')
(OUT/'DOWNLOAD_MANIFEST.json').write_text(json.dumps(man,indent=2)+'\n')
idxman=dict(dataset='GSE194122',h5ad=str(H5),shape=[int(x) for x in a.shape],sites=sites,feature_counts={'GEX':int(rna_mask.sum()),'ATAC':int(atac_mask.sum())},
    note='Main H5AD holds all sites; site_indices are row indices; no duplicate matrices.')
(OUT/'SITE_INDEX_MANIFEST.json').write_text(json.dumps(idxman,indent=2)+'\n')
print(json.dumps(dict(shape=a.shape,cell_types=int(pd.Series(ct).nunique()),sites=pd.Series(site).value_counts().to_dict(),
    missing=int(pd.isna(obs['cell_type']).sum()),dups=int(pd.Series(cell_id).duplicated().sum()),rna_nnz=rna_nnz,atac_nnz=atac_nnz,
    rna_zero=int((rna_total==0).sum()),atac_zero=int((atac_total==0).sum())),indent=2))
