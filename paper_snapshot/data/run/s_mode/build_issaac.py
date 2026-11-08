"""Build ISSAAC E-MTAB-11264 paired benchmark and S-mode intra cache."""
import gzip, json, shutil, sys, time
from bisect import bisect_left, bisect_right
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
ROOT=Path('/home/u202522802045/share/zpb')
sys.path[:0]=[str(ROOT),str(ROOT/'main/core/regulatory/common/legacy_support')]
from main.core.regulatory.s_mode.single_source import normalize_counts, select_hvg, build_bs, build_field
from v5d.regulatory_graph import read_tss
RAW=ROOT/'data/raw/Mouse-brain/ISSAAC_E-MTAB-11264/downloads'
ST=ROOT/'data/staging/Mouse-brain/ISSAAC_E-MTAB-11264_v1_20261023'
BENCH=ROOT/'data/benchmark/Mouse-brain/ISSAAC_E-MTAB-11264_v1_20261023/ISSAAC'
CACHE=ROOT/'data/cache/s_mode/Mouse-brain/ISSAAC_E-MTAB-11264_v1_20261023/ISSAAC'
RESULT=ROOT/'result_contain/89_brain_two_new_sources'
GTF=ROOT/'data/reference/genome_annotation/Mus_musculus/GRCm38/gencode_M25/gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz'
D4=ROOT/'data/benchmark/Mouse-brain/GSE229169_intra_v1_20261023/GSE229169'
def sha(p):
    import hashlib
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def parse_peak(name):
    c,se=str(name).rsplit(':',1); s,e=se.split('-'); return c,int(s),int(e)
def norm_chrom(c):
    c=str(c); return c if c.startswith('chr') else 'chr'+c
def graph_fast(genes,peaks,tss,window=250000,cap=50):
    by={}
    for i,p in enumerate(peaks):
        c,s,e=parse_peak(p); by.setdefault(c,[]).append(((s+e)//2,i))
    indexed={}
    for c,v in by.items(): v.sort(); indexed[c]=(v,[x[0] for x in v])
    index=np.zeros((len(genes),cap),np.int64); mask=np.zeros_like(index,bool); dist=np.zeros_like(index,np.float32)
    for i,g in enumerate(genes):
        c,pos=tss[str(g).split('.')[0]]; vals,positions=indexed.get(norm_chrom(c),([],[]))
        l,r=bisect_left(positions,pos-window),bisect_right(positions,pos+window)
        cand=sorted((abs(mid-pos),j) for mid,j in vals[l:r])[:cap]
        for k,(d,j) in enumerate(cand): index[i,k],mask[i,k],dist[i,k]=j,True,d
    return dict(candidate_index=index,candidate_mask=mask,distance_bp=dist)
def main():
    t=time.time(); RESULT.mkdir(parents=True,exist_ok=True)
    assert not BENCH.exists() and not CACHE.exists()
    BENCH.mkdir(parents=True)
    rna=sparse.load_npz(ST/'rna_counts.npz').tocsr()
    bcs=np.load(ST/'barcodes.npy',allow_pickle=True).astype(str)
    genes=np.load(ST/'gene_names.npy',allow_pickle=True).astype(str)
    meta=pd.read_csv(ST/'metadata.csv',keep_default_na=False)
    if 'barcode' not in meta.columns:
        meta=meta.rename(columns={meta.columns[0]:'barcode'})
    assert meta.barcode.tolist()==bcs.tolist() and rna.shape==(len(bcs),len(genes))
    # parse ATAC
    with gzip.open(RAW/'mCortex_all_ATAC_peak_matrix.csv.gz','rt') as f: h=f.readline().rstrip('\n').split(',')
    abcs=np.array(h[1:]); assert abcs.tolist()==bcs.tolist(), 'ATAC/RNA barcode order mismatch'
    chunks=[]; peaks=[]
    for ch in pd.read_csv(RAW/'mCortex_all_ATAC_peak_matrix.csv.gz',index_col=0,chunksize=4000):
        peaks.extend(ch.index.astype(str)); chunks.append(sparse.csr_matrix(ch.to_numpy(dtype=np.int32)))
    atac=sparse.vstack(chunks).tocsr().T.tocsr()
    atac.data[:]=1.0
    print('atac',atac.shape,atac.nnz,'peaks',len(peaks),flush=True)
    assert atac.shape==(len(bcs),len(peaks))
    # canonical peak universe: one aggregated peak set in the author matrix
    native=np.array(peaks,dtype=str); uniq,inv=np.unique(native,return_inverse=True)
    if len(uniq)!=len(native):
        atac=atac@sparse.csr_matrix((np.ones(len(native),np.float32),(np.arange(len(native)),inv)),shape=(len(native),len(uniq)))
        atac.data[:]=1.0
    else:
        atac=atac[:,np.argsort(inv)].tocsr()
    canonical=uniq; mapping=inv
    assert atac.shape[1]==len(canonical)
    # gene mapping symbols -> Ensembl via D4 32,285 gene table
    d4n=np.load(D4/'gene_names.npy',allow_pickle=True).astype(str); d4e=np.load(D4/'gene_ids.npy',allow_pickle=True).astype(str)
    lookup={}
    for s,e in zip(d4n,d4e): lookup.setdefault(s,set()).add(e)
    gene_ids=np.array([next(iter(lookup[s])) if s in lookup and len(lookup[s])==1 else '' for s in genes])
    print('mapped genes',int((gene_ids!='').sum()),'of',len(genes),flush=True)
    # save benchmark
    sparse.save_npz(BENCH/'rna_counts.npz',rna); sparse.save_npz(BENCH/'atac_binary.npz',atac)
    np.save(BENCH/'peak_names.npy',canonical); np.save(BENCH/'barcodes.npy',bcs)
    np.save(BENCH/'gene_names.npy',genes); np.save(BENCH/'gene_ids.npy',gene_ids)
    np.save(BENCH/'native_peak_names.npy',native); np.save(BENCH/'native_to_canonical.npy',mapping)
    sparse.save_npz(BENCH/'peak_availability.npz',sparse.csr_matrix(np.ones((1,len(canonical)),np.float32)))
    idx=np.arange(len(meta)); tr,held=train_test_split(idx,train_size=.7,random_state=20020130); va,te=train_test_split(held,test_size=.5,random_state=20020130)
    splits={k:np.sort(v) for k,v in [('train',tr),('val',va),('test',te)]}
    meta['split']=''
    for k,v in splits.items(): meta.loc[v,'split']=k
    meta.to_csv(BENCH/'metadata.csv',index=False)
    np.savez_compressed(BENCH/'splits.npz',**{k+'_index':v for k,v in splits.items()})
    # audits
    labels=meta['RNA_Cluster_Annotation'].fillna('').astype(str)
    pd.DataFrame(dict(barcode=bcs,batch=meta.batch,label=labels,
        nCount_RNA=meta.nCount_RNA,nFeature_RNA=meta.nFeature_RNA,percent_mt=meta['percent.mt'],
        nCount_ATAC=meta.nCount_ATAC,nFeature_ATAC=meta.nFeature_ATAC,TSS_enrichment=meta['TSS.enrichment'],
        nucleosome_signal=meta['nucleosome_signal'],blacklist_fraction=meta['blacklist_fraction'])).to_csv(RESULT/'ISSAAC_CELL_QC.csv',index=False)
    # cache
    full=normalize_counts(rna); tss=read_tss(GTF)
    mapped=np.flatnonzero(gene_ids!='')
    mapped=mapped[np.array([g in tss for g in gene_ids[mapped]])]
    selected,scores=select_hvg(full[splits['train']][:,mapped],2000); columns=mapped[selected]; genes2000=gene_ids[columns]
    assert len(genes2000)==len(set(genes2000))==2000
    rna2000=full[:,columns].toarray().astype(np.float32); del full
    graph=graph_fast(genes2000,canonical,tss); graph['peak_names']=canonical
    bs,diagnostics=build_bs(rna2000[splits['train']],atac[splits['train']],graph)
    field,field_state,field_report=build_field(atac[np.concatenate([splits['train'],splits['val']])],bs,np.arange(len(splits['train'])))
    CACHE.mkdir(parents=True)
    for k,v in splits.items(): np.save(CACHE/f'rna_{k}.npy',rna2000[v]); np.save(CACHE/f'{k}_index.npy',v)
    np.save(CACHE/'F_train.npy',field[:len(splits['train'])]); np.save(CACHE/'F_val.npy',field[len(splits['train']):])
    np.save(CACHE/'gene_ids.npy',genes2000); np.save(CACHE/'peak_names.npy',canonical)
    np.savez_compressed(CACHE/'rna_state.npz',mean=rna2000[splits['train']].mean(0),std=rna2000[splits['train']].std(0),fit_cell_index=splits['train'])
    np.savez_compressed(CACHE/'field_state.npz',**field_state); np.savez_compressed(CACHE/'regulatory_stability.npz',**diagnostics); np.savez_compressed(CACHE/'candidate_graph.npz',**graph)
    sparse.save_npz(CACHE/'B_S.npz',bs)
    det=np.asarray((rna2000[splits['train']]>0).mean(0)).ravel(); per=np.asarray((rna2000>0).sum(1)).ravel()
    pd.DataFrame(dict(hvg_rank=np.arange(2000),gene_symbol=genes[columns],ensembl_gene_id=genes2000,detection_rate_train=det)).to_csv(CACHE/'hvg.csv',index=False)
    quality=dict(train_median_detected_HVG=float(np.median(per[splits['train']])),test_all_zero_HVG_cells=int((per[splits['test']]==0).sum()),
        HVG_detected_in_under_1pct_train=int((det<0.01).sum()),unsupported_field_genes=int((field_state['field_std']<1e-6).sum()))
    (CACHE/'input_quality.json').write_text(json.dumps(quality,indent=2)+'\n')
    manifest=dict(status='S_MODE_INTRA_CACHE_COMPLETE',dataset='ISSAAC_E-MTAB-11264',cells=len(meta),genes=2000,peaks=len(canonical),
        split_counts={k:len(v) for k,v in splits.items()},split_seed=20020130,RNA='native counts; 1e4+log1p; train-only variance/mean Top2000',
        labels=23,label_column='RNA_Cluster_Annotation',author_QC_reused=True,new_QC_filters=False,HVG_intersection_performed=False,
        B_edges=int(bs.nnz),F_shape=list(field.shape),F_variant='frozen_fit_dmode_field',labels_used_for_model_fitting=False,
        test_cells_used_for_fitting=False,test_atac_used_for_fitting=False,quality=quality,fragments_available=True,
        fragments_used=False,peak_mode='author peak counts -> binary any-hit',hashes={str(p):sha(p) for p in [BENCH/'metadata.csv',BENCH/'rna_counts.npz',BENCH/'atac_binary.npz',BENCH/'peak_names.npy']})
    (CACHE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (BENCH/'manifest.json').write_text(json.dumps(dict(status='ISSAAC_BENCHMARK_COMPLETE',cells=len(meta),genes=len(genes),peaks=len(canonical),labels=23,split_counts={k:len(v) for k,v in splits.items()}),indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)
if __name__=='__main__': main()
