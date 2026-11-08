"""Build FC author-RNA-labeled cohort and S-mode intra cache."""
import json, shutil, sys, time
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
SRC=ROOT/'data/benchmark/Mouse-brain/GSE294772_fullb2_standard_v1_20261023/GSE294772'
ANN=ROOT/'data/benchmark/Mouse-brain/GSE294772_author_annotation_v1_20261023/cell_annotations.csv'
BENCH=ROOT/'data/benchmark/Mouse-brain/FC_author_labeled_v1_20261023/FC'
CACHE=ROOT/'data/cache/s_mode/Mouse-brain/FC_author_labeled_v1_20261023/FC'
RESULT=ROOT/'result_contain/85_brain_four_dataset_preprocessing'
GTF=ROOT/'data/reference/genome_annotation/Mus_musculus/GRCm38/gencode_M25/gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz'

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
    started=time.perf_counter()
    assert not BENCH.exists(), BENCH
    assert not CACHE.exists(), CACHE
    RESULT.mkdir(parents=True,exist_ok=True)
    BENCH.mkdir(parents=True)
    rna=sparse.load_npz(SRC/'rna_counts.npz').tocsr()
    atac=sparse.load_npz(SRC/'atac_counts.npz').tocsr()
    peaks=np.load(SRC/'peak_names.npy',allow_pickle=True).astype(str)
    bcs=np.load(SRC/'barcodes.npy',allow_pickle=True).astype(str)
    meta=pd.read_csv(SRC/'metadata.csv',keep_default_na=False)
    ann=pd.read_csv(ANN,keep_default_na=False)
    assert bcs.tolist()==meta.barcode.tolist()==ann.barcode.tolist()
    assert rna.shape==(12416,32285) and atac.shape==(12416,len(peaks))
    keep=ann.author_RNA_label.fillna('').str.strip().ne('')
    assert int(keep.sum())==9821, int(keep.sum())
    meta=meta.loc[keep].reset_index(drop=True)
    meta['reference_label']=ann.loc[keep,'author_RNA_label'].to_numpy()
    meta['author_ATAC_label']=ann.loc[keep,'author_ATAC_label'].to_numpy()
    meta['label_conflict']=ann.loc[keep,'author_labels_conflict'].to_numpy()
    meta['library']=ann.loc[keep,'sample'].to_numpy()
    meta['RNA_GSM']=ann.loc[keep,'rna_gsm'].to_numpy(); meta['ATAC_GSM']=ann.loc[keep,'atac_gsm'].to_numpy()
    keep_idx=np.flatnonzero(keep.to_numpy())
    sparse.save_npz(BENCH/'rna_counts.npz',rna[keep_idx])
    sparse.save_npz(BENCH/'atac_counts.npz',atac[keep_idx])
    sparse.save_npz(BENCH/'peak_availability.npz',sparse.load_npz(SRC/'peak_availability.npz'))
    np.save(BENCH/'peak_names.npy',peaks); np.save(BENCH/'barcodes.npy',meta.barcode.to_numpy(str))
    np.save(BENCH/'gene_ids.npy',np.load(SRC/'gene_ids.npy',allow_pickle=True))
    np.save(BENCH/'gene_names.npy',np.load(SRC/'gene_names.npy',allow_pickle=True))
    shutil.copytree(SRC/'identity_map',BENCH/'identity_map')
    # split
    idx=np.arange(len(meta)); train,held=train_test_split(idx,train_size=.7,random_state=20020130); val,test=train_test_split(held,test_size=.5,random_state=20020130)
    splits={k:np.sort(v) for k,v in [('train',train),('val',val),('test',test)]}
    assert np.array_equal(np.sort(np.concatenate(list(splits.values()))),idx)
    meta['split']=''
    for k,v in splits.items(): meta.loc[v,'split']=k
    meta.to_csv(BENCH/'metadata.csv',index=False)
    np.savez_compressed(BENCH/'splits.npz',**{k+'_index':v for k,v in splits.items()})
    # label/exclusion audits
    audit=ann.copy(); audit['kept']=keep.to_numpy(); audit['exclusion_reason']=np.where(keep,'','NO_AUTHOR_RNA_LABEL')
    audit.to_csv(RESULT/'FC_AUTHOR_LABEL_AUDIT.csv',index=False)
    audit.loc[~keep,['barcode','sample','author_RNA_label','author_ATAC_label','label_conflict' if 'label_conflict' in audit.columns else 'author_labels_conflict']].assign(reason='NO_AUTHOR_RNA_LABEL').to_csv(RESULT/'FC_FILTER_EXCLUSIONS.csv',index=False)
    # cache prep
    full=normalize_counts(rna)
    mapping=pd.read_csv(BENCH/'identity_map/gene_mapping.tsv',sep='\t',keep_default_na=False)
    gene_names=np.load(BENCH/'gene_names.npy',allow_pickle=True).astype(str)
    ens=np.load(BENCH/'gene_ids.npy',allow_pickle=True).astype(str)
    assert mapping.input_feature.tolist()==ens.tolist()
    tss=read_tss(GTF)
    eligible=np.flatnonzero(mapping.status.eq('unique').to_numpy() & mapping.ensembl_gene_id.isin(tss).to_numpy())
    selected,scores=select_hvg(full[splits['train']][:,eligible],2000)
    columns=eligible[selected]; genes=ens[columns]
    assert len(genes)==len(set(genes))==2000
    rna2000=full[:,columns].toarray().astype(np.float32)
    del full
    graph=graph_fast(genes,peaks,tss); graph['peak_names']=peaks
    atac_bin=atac[keep_idx].tocsr().astype(bool).astype(np.float32)
    bs,diagnostics=build_bs(rna2000[splits['train']],atac_bin[splits['train']],graph)
    paired=np.concatenate([splits['train'],splits['val']])
    field,field_state,field_report=build_field(atac_bin[paired],bs,np.arange(len(splits['train'])))
    CACHE.mkdir(parents=True)
    for k,v in splits.items(): np.save(CACHE/f'rna_{k}.npy',rna2000[v]); np.save(CACHE/f'{k}_index.npy',v)
    np.save(CACHE/'F_train.npy',field[:len(splits['train'])]) ; np.save(CACHE/'F_val.npy',field[len(splits['train']):])
    np.save(CACHE/'gene_ids.npy',genes); np.save(CACHE/'peak_names.npy',peaks)
    np.savez_compressed(CACHE/'rna_state.npz',mean=rna2000[splits['train']].mean(0),std=rna2000[splits['train']].std(0),fit_cell_index=splits['train'])
    np.savez_compressed(CACHE/'field_state.npz',**field_state)
    np.savez_compressed(CACHE/'regulatory_stability.npz',**diagnostics)
    np.savez_compressed(CACHE/'candidate_graph.npz',**graph)
    sparse.save_npz(CACHE/'B_S.npz',bs)
    det=np.asarray((rna2000[splits['train']]>0).mean(0)).ravel(); per=np.asarray((rna2000>0).sum(1)).ravel()
    pd.DataFrame(dict(hvg_rank=np.arange(2000),gene_symbol=gene_names[columns],ensembl_gene_id=genes,detection_rate_train=det)).to_csv(CACHE/'hvg.csv',index=False)
    quality=dict(train_median_detected_HVG=float(np.median(per[splits['train']])),test_all_zero_HVG_cells=int((per[splits['test']]==0).sum()),
                 HVG_detected_in_under_1pct_train=int((det<0.01).sum()),unsupported_field_genes=int((field_state['field_std']<1e-6).sum()))
    (CACHE/'input_quality.json').write_text(json.dumps(quality,indent=2)+'\n')
    manifest=dict(status='S_MODE_INTRA_CACHE_COMPLETE',dataset='FC_author_labeled_v1',cells=len(meta),genes=2000,peaks=len(peaks),
        split_counts={k:len(v) for k,v in splits.items()},split_seed=20020130,RNA='native 1e4+log1p; train-only variance/mean Top2000',
        author_QC_reused=False,new_QC_filters=False,label_source='author RNA supplement; RNA primary; ATAC separate',
        label_conflicts=85,label_conflict_policy='keep RNA primary; flag',HVG_intersection_performed=False,
        candidates=int(graph['candidate_mask'].sum()),B_edges=int(bs.nnz),F_shape=list(field.shape),F_variant='frozen_fit_dmode_field',
        labels_used_for_model_fitting=False,test_cells_used_for_fitting=False,test_atac_used_for_fitting=False,
        quality=quality,hashes={str(p):sha(p) for p in [BENCH/'metadata.csv',BENCH/'rna_counts.npz',BENCH/'atac_counts.npz',BENCH/'peak_names.npy']})
    (CACHE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (BENCH/'manifest.json').write_text(json.dumps(dict(status='FC_AUTHOR_LABELED_BENCHMARK_COMPLETE',cells=len(meta),genes=32285,peaks=len(peaks),
        labels=37,conflicts=85,excluded=int((~keep).sum()),splits={k:len(v) for k,v in splits.items()}),indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)
    print('FC_DONE',flush=True)
if __name__=='__main__': main()
