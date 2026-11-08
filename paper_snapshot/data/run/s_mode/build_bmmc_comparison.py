"""Build the four BMMC Site comparison caches from Task97 splits and the official H5AD."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, h5py, anndata as ad
from scipy import sparse

ROOT=Path('/home/u202522802045/share/zpb')
BMMC=ROOT/'data/cache/s_mode/BMMC/bmmc_site_intra_v1_20261024'
CACHE=ROOT/'data/cache/s_mode/BMMC/bmmc_comparison_intra_v1'
BENCH=ROOT/'benchmark/BMMC'
H5=ROOT/'data/raw/BMMC/GSE194122/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad'
SITES=['BMMC_Site1','BMMC_Site2','BMMC_Site3','BMMC_Site4']


def build():
    a=ad.read_h5ad(H5,backed='r')
    obs_names=obs=a.obs_names.astype(str)
    var=a.var.copy(); ft=var['feature_types'].astype(str).to_numpy()
    gene_ids=var.loc[ft=='GEX','gene_id'].astype(str).to_numpy()
    peak_names=var.index.to_numpy().astype(str)[ft=='ATAC']
    with h5py.File(H5,'r') as f:
        sh=tuple(int(x) for x in f['layers/counts'].attrs['shape'])
        M=sparse.csr_matrix((f['layers/counts/data'][:],f['layers/counts/indices'][:],f['layers/counts/indptr'][:]),shape=sh)
    rna_all=M[:,ft=='GEX'].tocsr(); atac_all=M[:,ft=='ATAC'].tocsr()
    del M
    gpos={g:i for i,g in enumerate(gene_ids)}; ppos={p:i for i,p in enumerate(peak_names)}
    row_lookup={c:i for i,c in enumerate(obs_names)}
    for site in SITES:
        sc=BMMC/site
        cells=np.load(sc/'barcodes.npy',allow_pickle=True).astype(str)
        rows=np.asarray([row_lookup[c] for c in cells])
        genes=np.load(sc/'gene_ids.npy',allow_pickle=True).astype(str)
        gcols=np.asarray([gpos[g] for g in genes])
        raw_rna=rna_all[rows][:,gcols].tocsr()
        splits={s:np.load(sc/f'{s}_index.npy') for s in ('train','val','test')}
        labels=np.load(sc/'labels.npy',allow_pickle=True).astype(str)
        peaks=np.load(sc/'peak_names.npy',allow_pickle=True).astype(str)
        pcols=np.asarray([ppos[p] for p in peaks])
        raw_atac=atac_all[rows][:,pcols].tocsr()
        tr,va,te=splits['train'],splits['val'],splits['test']
        train_bin=raw_atac[tr].astype(bool).astype(np.float32)
        keep=np.asarray(train_bin.sum(0)).ravel()>=np.ceil(.005*len(tr))
        sel=np.flatnonzero(keep); peak_sel=peaks[sel]
        atac=raw_atac[:,sel].astype(bool).astype(np.float32)
        idf=np.log1p(len(tr)/np.maximum(np.asarray(atac[tr].sum(0)).ravel(),1)).astype(np.float32)
        tfidf=atac.multiply(1/np.maximum(np.asarray(atac.sum(1)).ravel(),1)[:,None]).multiply(idf).tocsr()
        scale=max(float(tfidf[tr].max()),1e-8); tfidf.data=np.clip(tfidf.data/scale,0,1)
        out=CACHE/site; out.mkdir(parents=True,exist_ok=False)
        for split,idx in [('train',tr),('val',va),('test',te)]:
            sparse.save_npz(out/f'rna_{split}.npz',raw_rna[idx].astype(np.float32))
            sparse.save_npz(out/f'rna_log_{split}.npz',sparse.csr_matrix(np.load(sc/f'rna_{split}.npy').astype(np.float32)))
            np.save(out/f'{split}_index.npy',idx)
        for split in ('train','val'):
            sparse.save_npz(out/f'atac_{split}.npz',atac[splits[split]].tocsr())
            sparse.save_npz(out/f'atac_tfidf_{split}.npz',tfidf[splits[split]].tocsr())
        np.save(out/'gene_ids.npy',genes); np.save(out/'peak_ids.npy',peak_sel)
        (out/'barcodes.txt').write_text('\n'.join(cells)+'\n')
        pd.DataFrame(dict(barcode=cells,cell_type=labels)).to_csv(BENCH/site/'label.csv',index=False)
        np.savez_compressed(out/'atac_state.npz',peak_columns=np.arange(len(peak_sel)),train_idf=idf,train_max=scale)
        manifest=dict(status='PREPARED',identity=site,cells=len(cells),rna_genes=len(genes),atac_peaks=len(peak_sel),
            split_counts={s:len(v) for s,v in splits.items()},preprocessing_seconds=0.,
            test_atac_loaded=False,labels_used_for_fitting=False,cache_sha256={})
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('CACHE',site,manifest['split_counts'],'peaks',len(peak_sel),flush=True)

if __name__=='__main__':
    for s in SITES: (BENCH/s).mkdir(parents=True,exist_ok=True)
    build()
