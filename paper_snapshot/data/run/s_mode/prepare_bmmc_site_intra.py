"""Prepare four BMMC Site caches for Full B2 and RNA-only intra baselines."""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import sparse
import h5py, anndata as ad
from sklearn.model_selection import train_test_split

ROOT=Path('/home/u202522802045/share/zpb')
sys.path[:0]=[str(ROOT),str(ROOT/'main/run/d_mode'),str(ROOT/'main/core/regulatory/common/legacy_support')]
import prepare_updated_b2_sources as common
from main.core.regulatory.s_mode.single_source import build_bs, build_field, normalize_counts, select_hvg
from v5d.regulatory_graph import read_tss

H5=ROOT/'data/raw/BMMC/GSE194122/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad'
GTF=ROOT/'data/reference/genome_annotation/Homo_sapiens/GRCh38/gencode_32/gencode.v32.primary_assembly.annotation.gtf.gz'
CACHE=ROOT/'data/cache/s_mode/BMMC/bmmc_site_intra_v1_20261024'
SITES=['BMMC_Site1','BMMC_Site2','BMMC_Site3','BMMC_Site4']


def load():
    a=ad.read_h5ad(H5,backed='r')
    obs=a.obs.copy(); obs.index=obs.index.astype(str)
    var=a.var.copy()
    ft=var['feature_types'].astype(str).to_numpy()
    gex=ft=='GEX'; atac=ft=='ATAC'
    genes=var.loc[gex,'gene_id'].astype(str).to_numpy()
    peaks=var.index.to_numpy().astype(str)[atac]
    with h5py.File(H5,'r') as f:
        g=f['layers/counts']; shape=tuple(int(x) for x in g.attrs['shape'])
        M=sparse.csr_matrix((g['data'][:],g['indices'][:],g['indptr'][:]),shape=shape)
    return obs,genes,peaks,M[:,gex].tocsr(),M[:,atac].tocsr()


def prepare(site,obs,genes,peaks,rna_all,atac_all):
    cache=CACHE/site
    if (cache/'manifest.json').exists():
        return common.checked(cache/'manifest.json')
    if cache.exists() and any(cache.iterdir()):
        raise RuntimeError(f'Incomplete cache requires inspection: {cache}')
    started=time.perf_counter()
    mask=(obs['Site'].astype(str).to_numpy()==site.replace('BMMC_','').lower())
    if mask.sum()==0:
        # obs Site values are site1..site4
        mask=(obs['Site'].astype(str).to_numpy()==site.replace('BMMC_','').replace('Site','site'))
    idx=np.where(mask)[0]
    labels=obs['cell_type'].astype(str).to_numpy()[idx]
    counts=rna_all[idx].tocsr()
    atac=atac_all[idx].tocsr()
    # stratified 70/15/15 split, fixed split seed
    tr,tmp=train_test_split(np.arange(len(idx)),test_size=0.30,stratify=labels,random_state=20020130)
    va,te=train_test_split(tmp,test_size=0.50,stratify=labels[tmp],random_state=20020130)
    tr=np.sort(tr); va=np.sort(va); te=np.sort(te)
    assert len(set(tr)|set(va)|set(te))==len(idx)
    train_log=normalize_counts(counts[tr])
    hvg,_=select_hvg(train_log)
    train_rna=train_log[:,hvg].toarray().astype(np.float32)
    val_rna=normalize_counts(counts[va])[:,hvg].toarray().astype(np.float32)
    test_rna=normalize_counts(counts[te])[:,hvg].toarray().astype(np.float32)
    mean,std=train_rna.mean(0),train_rna.std(0)
    genes_hvg=genes[hvg]
    atac_bin=atac.astype(bool).astype(np.float32)
    atac_tv=atac_bin[np.concatenate([tr,va])]
    graph=common.graph_fast(genes_hvg,peaks,read_tss(GTF))
    graph['peak_names']=peaks
    t0=time.perf_counter(); B_S,diag=build_bs(train_rna,atac_bin[tr],graph); selector=time.perf_counter()-t0
    t0=time.perf_counter(); field,state,report=build_field(atac_tv,B_S,np.arange(len(tr))); field_seconds=time.perf_counter()-t0
    assert B_S.nnz>0 and field.shape==(len(tr)+len(va),len(hvg))
    cache.mkdir(parents=True,exist_ok=False)
    np.save(cache/'gene_ids.npy',genes_hvg.astype(str)); np.save(cache/'peak_names.npy',peaks.astype(str))
    np.save(cache/'rna_train.npy',train_rna); np.save(cache/'rna_val.npy',val_rna); np.save(cache/'rna_test.npy',test_rna)
    np.save(cache/'F_train.npy',field[:len(tr)].astype(np.float32)); np.save(cache/'F_val.npy',field[len(tr):].astype(np.float32))
    np.save(cache/'train_index.npy',tr); np.save(cache/'val_index.npy',va); np.save(cache/'test_index.npy',te)
    np.save(cache/'labels.npy',labels.astype(str)); np.save(cache/'barcodes.npy',obs.index.to_numpy()[idx].astype(str))
    np.savez_compressed(cache/'rna_state.npz',mean=mean,std=std,hvg_index=hvg,fit_cell_index=tr)
    np.savez_compressed(cache/'field_state.npz',**state)
    sparse.save_npz(cache/'B_S.npz',B_S)
    (cache/'barcodes.txt').write_text('\n'.join(obs.index.to_numpy()[idx].astype(str))+'\n')
    deps=[H5,GTF,Path(__file__),ROOT/'main/core/regulatory/s_mode/single_source.py',ROOT/'main/run/d_mode/prepare_updated_b2_sources.py']
    manifest=dict(status='PREPARED',dataset=site,cells=len(idx),genes=len(hvg),peaks=len(peaks),
        split_counts={'train':len(tr),'val':len(va),'test':len(te)},cell_types=int(len(set(labels))),
        hvg_formula='variance/mean on full-library log1p Site train; stable ascending argsort tail',
        fitting_cells='train only',labels_used_for_split_stratification=True,labels_used_for_model_fitting=False,
        test_atac_used=False,selector_seconds=selector,field_seconds=field_seconds,
        preprocessing_seconds=time.perf_counter()-started,candidate_edges=int(graph['candidate_mask'].sum()),
        retained_edges=int(B_S.nnz),retained_genes=int((np.asarray(abs(B_S).sum(0)).ravel()>0).sum()),
        hashes={str(p):common.sha(p) for p in deps+list(cache.iterdir()) if p.is_file()})
    common.write(cache/'manifest.json',manifest)
    print('SITE_CACHE_READY',site,manifest['split_counts'],'edges',B_S.nnz,flush=True)
    return manifest


if __name__=='__main__':
    obs,genes,peaks,rna,atac=load()
    for site in SITES: prepare(site,obs,genes,peaks,rna,atac)
