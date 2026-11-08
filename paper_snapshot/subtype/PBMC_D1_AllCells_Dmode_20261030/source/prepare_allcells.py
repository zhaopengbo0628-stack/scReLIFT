"""Rebuild all-source artifacts using unchanged frozen Full B2 operations."""
import csv
import time
import shutil
from dataclasses import asdict
import h5py
import pandas as pd
from scipy.io import mmread
from common import *
from v5d.dataset_spec import DatasetSpec
from v5d.harmonize import prepare_harmonized
from v5d.regulatory_graph import read_tss,build_candidate_graph
from v5d.dmode_reliable_selector import DModeReliabilityConfig,build_source_only_b
from v5d.dmode_teacher import fit_dmode_field
from v5d.data import V5DData

ART=RUN/'artifacts'
GTF=Path('/home/u202522802045/glue_156478/graph_files/gencode.v32.primary_assembly.annotation.gtf.gz')
EXPECTED={'D1':9347,'D2':6317,'D3':8105,'D4':2413}
OLD=SHARED/'data/cache/d_mode/PBMC/legacy_full_b2_corrected/_historical_dependencies'

def staged(key):
    p=SHARED/'data/staging/PBMC/legacy_dmode_data_20261004/derived/common'/key
    x=sparse.load_npz(p/'rna_counts.npz').tocsr()
    a=sparse.load_npz(p/'atac_counts.npz').tocsr()
    genes=pd.read_csv(p/'genes.tsv',sep='\t')
    ids=genes.iloc[:,0].astype(str).to_numpy()
    names=genes.iloc[:,1].astype(str).to_numpy()
    cells=np.asarray((p/'barcodes.tsv').read_text().splitlines())
    peaks=np.asarray(['%s:%s-%s'%tuple(line.split('\t')[:3]) for line in (p/'peaks.bed').read_text().splitlines()])
    rawroot=SHARED/'data/raw/PBMC'/f'PBMC-MO-00{key[-1]}'
    h5=next(rawroot.glob('*feature_bc_matrix.h5'))
    ix=np.load(p/'raw_h5_cell_indices.npy')
    with h5py.File(h5) as h:
        m=h['matrix'];ft=m['features/feature_type'][:].astype(str)
        assert np.array_equal(cells,m['barcodes'][:].astype(str)[ix])
        assert np.array_equal(ids,m['features/id'][:].astype(str)[ft=='Gene Expression'])
        assert np.array_equal(peaks,m['features/id'][:].astype(str)[ft=='Peaks'])
        # Verify full sparse matrices, not just filenames or dimensions.
        full=sparse.csc_matrix((m['data'][:],m['indices'][:],m['indptr'][:]),shape=tuple(m['shape'][:])).T.tocsr()[ix]
        assert (x-full[:,ft=='Gene Expression']).nnz==0
        assert (a-full[:,ft=='Peaks']).nnz==0
    return x,a,cells,ids,names,peaks,dict(raw_h5=str(h5),raw_h5_sha256=sha(h5),
             rna_sha256=sha(p/'rna_counts.npz'),atac_sha256=sha(p/'atac_counts.npz'),full_matrix_match=True)

def corrected_d2():
    p=OLD/'source_inputs/D2'
    dense=np.load(p/'rna.npy',mmap_mode='r')
    assert dense.shape==(6317,36601)
    author=RUN/'input/D2_author'
    x=mmread(author/'rna_counts_genes_by_cells.mtx').T.tocsr().astype(np.int64)
    assert (x-sparse.csr_matrix(dense)).nnz==0, 'Full-count cache differs from original author RDS'
    assert np.isfinite(x.data).all() and np.all(x.data>=0) and np.array_equal(x.data,np.round(x.data))
    a=sparse.load_npz(p/'atac_binary.npz').tocsr()
    reference=SHARED/'data/raw/PBMC/PBMC-MO-003/pbmc_unsorted_10k_filtered_feature_bc_matrix.h5'
    with h5py.File(reference) as h:
        ft=h['matrix/features/feature_type'][:].astype(str)=='Gene Expression'
        ids=h['matrix/features/id'][:].astype(str)[ft]
        names=h['matrix/features/name'][:].astype(str)[ft]
    supplied=np.asarray((author/'gene_names.txt').read_text().splitlines())
    # Author RDS has ordered feature symbols; verify all 36601, including duplicate suffixes.
    seen=set();unique=[]
    for name in names:
        v=name;suffix=1
        while v in seen:
            v=f'{name}.{suffix}';suffix+=1
        unique.append(v);seen.add(v)
    assert np.array_equal(supplied,np.asarray(unique)), 'Author gene order differs from complete 10x reference'
    peaks=np.load(p/'peak_names.npy').astype(str)
    bc=SHARED/'data/cache/s_mode/PBMC/PBMC-MO-002/pbmc_corrected_s10x20_v1_20261012/barcodes.txt'
    cells=np.asarray(bc.read_text().splitlines())
    assert np.array_equal(cells,np.asarray((author/'barcodes.txt').read_text().splitlines()))
    assert len(cells)==6317 and len(set(cells))==6317 and all(c.startswith('HDa_') for c in cells)
    raw=SHARED/'data/raw/PBMC/PBMC-MO-002'
    identity=json.loads((raw/'DATASET_IDENTITY.json').read_text())
    assert identity['canonical_identity']=='GEO:GSE196990:GSE196988+GSE196987:HDa'
    rmeta=raw/'GSE196988_RNA_Metadata_Final_Multiome.txt/GSE196988_RNA_Metadata_Final_Multiome.txt'
    ameta=raw/'GSE196987_ATAC_Metadata_Final_Multiome.txt/GSE196987_ATAC_Metadata_Final_Multiome.txt'
    rm=pd.read_csv(rmeta,sep='\t',index_col=0,low_memory=False).loc[cells]
    am=pd.read_csv(ameta,sep='\t',index_col=0,low_memory=False).loc[cells]
    assert np.all(rm['sample']=='HDa') and np.all(am['sample']=='HDa')
    assert np.array_equal(rm['barcodes'].astype(str),cells) and np.array_equal(am['barcodes'].astype(str),cells)
    assert np.allclose(np.asarray(x.sum(1)).ravel(),rm['nCount_RNA'])
    assert np.array_equal(x.getnnz(axis=1),rm['nFeature_RNA'].to_numpy())
    assert np.array_equal(a.getnnz(axis=1),am['nFeature_ATAC'].to_numpy())
    # Exact expression reproduction against frozen historical 2000-gene object.
    prev=OLD/'validated_prepared/D2'
    oldids=np.load(prev/'gene_ids.npy').astype(str)
    lookup=dict(zip(ids,range(len(ids))))
    selected=np.asarray([lookup[g] for g in oldids])
    historical=np.load(prev/'rna.npy')
    new=normalize_log1p(x)[:,selected].toarray().astype(np.float32)
    assert np.array_equal(new,historical),'Full-gene raw counts do not reproduce frozen D2 expression'
    oldpeaks=np.load(prev/'peak_names.npy').astype(str)
    lookup=dict(zip(peaks,range(len(peaks))))
    cols=np.asarray([lookup[p] for p in oldpeaks])
    assert (a[:,cols]-sparse.load_npz(prev/'atac_binary.npz')).nnz==0
    sparse.save_npz(author/'rna_counts.npz',x)
    rm.to_csv(RUN/'audit/D2_author_metadata.csv')
    return x,a,cells,ids,names,peaks,dict(identity=identity['canonical_identity'],
        raw_integer_counts=True,author_rna_umi_and_detected_genes_match_all_cells=True,
        author_atac_detected_peaks_match_all_cells=True,paired_barcode_match=True,
        old_2000_gene_rna_bitwise_reproduced=True,old_atac_binary_subset_exactly_reproduced=True,
        gene_mapping='All 36601 author RDS rownames exactly match ordered 10x unique symbols; use corresponding Ensembl IDs, not legacy source_inputs gene_ids',
        original_rds_counts_equal_full_count_cache=True,rna_sha256=sha(author/'rna_counts.npz'),
        atac_sha256=sha(p/'atac_binary.npz'),rna_metadata_sha256=sha(rmeta),atac_metadata_sha256=sha(ameta),
        counts_cache=str(p),note='ATAC binary cache is sufficient for unchanged Full B2 protocol; no ATAC count claim')

def main():
    (RUN/'audit').mkdir(exist_ok=True)
    if (ART/'PREPARATION_COMPLETE.json').exists():
        manifest=json.loads((ART/'PREPARATION_COMPLETE.json').read_text())
        for p,h in manifest['hashes'].items(): assert sha(RUN/p)==h
        print('PREPARATION_ALREADY_VERIFIED',flush=True);return
    ART.mkdir(exist_ok=True)
    assert sha(GTF)=='0c36c37abb0c88637b04a8e401c93f0199a65bbdd0702c9e13bdfbadc053fe1e'
    tss=read_tss(GTF)
    readiness=[]
    # All source artifacts completed before target expression/labels are loaded.
    for key in ['D2','D3','D4']:
        print('READ_SOURCE',key,flush=True)
        x,a,cells,ids,names,peaks,audit=corrected_d2() if key=='D2' else staged(key)
        assert x.shape==(EXPECTED[key],36601) and a.shape[0]==len(cells)
        mapped=np.flatnonzero([g.split('.')[0] in tss for g in ids])
        graph=graph_fast(ids[mapped],peaks,tss)
        eligible=mapped[np.flatnonzero(graph['candidate_mask'].any(1))]
        selected=hvg_train_indices(x,np.arange(x.shape[0]),eligible,2000)
        assert len(selected)==2000
        g=graph_fast(ids[selected],peaks,tss)
        ref,_=build_candidate_graph(ids[selected[:8]],peaks,tss,250000,50)
        assert all(np.array_equal(g[k][:8],ref[k]) for k in ref)
        kept=np.unique(g['candidate_index'][g['candidate_mask']])
        dest=ART/'native'/key;dest.mkdir(parents=True,exist_ok=True)
        np.save(dest/'rna.npy',normalize_log1p(x)[:,selected].toarray().astype(np.float32))
        np.save(dest/'gene_ids.npy',ids[selected]);np.save(dest/'gene_names.npy',names[selected])
        np.save(dest/'peak_names.npy',peaks[kept])
        sparse.save_npz(dest/'atac_binary.npz',a[:,kept].astype(bool).astype(np.float32))
        np.savez_compressed(dest/'metadata.npz',train_index=np.arange(len(cells)),val_index=np.asarray([],dtype=np.int64),test_index=np.asarray([],dtype=np.int64))
        (dest/'barcodes.tsv').write_text('\n'.join(cells)+'\n')
        row=dict(dataset=key,role='SOURCE_ALL_TRAIN',cells=len(cells),hvg=2000,eligible_genes=len(eligible),**audit)
        readiness.append(row);write(RUN/'audit'/f'{key}_data_audit.json',row)
        del x,a
    specs=[DatasetSpec(key,ART/'native'/key) for key in ['D2','D3','D4']]
    hm=prepare_harmonized(specs,ART/'harmonized',feature_reference_names=['D2','D3','D4'],held_out_target_name='D1')
    genes=np.load(ART/'harmonized/common_gene_ids.npy').astype(str)
    peaks=np.load(ART/'harmonized/canonical_peak_names.npy').astype(str)
    graph=graph_fast(genes,peaks,tss);graph['peak_names']=peaks
    np.savez_compressed(ART/'candidate_graph.npz',**graph)
    datasets=[V5DData(ART/'harmonized',key) for key in ['D2','D3','D4']]
    started=time.perf_counter();config=DModeReliabilityConfig()
    print('BUILD_B_D',len(genes),'genes',len(peaks),'peaks',flush=True)
    selection=build_source_only_b(datasets,graph,genes,config,20020130)
    sparse.save_npz(ART/'B_D.npz',selection['B_D'])
    for key,value in selection.items():
        if isinstance(value,np.ndarray): np.save(ART/f'{key}.npy',value)
        elif isinstance(value,list) and value and isinstance(value[0],np.ndarray):
            np.savez_compressed(ART/f'{key}.npz',**{s.name:v for s,v in zip(datasets,value)})
    write(ART/'regulatory_manifest.json',dict(config=asdict(config),selection=selection['manifest'],seconds=time.perf_counter()-started,target_loaded=False))
    for data in datasets:
        field,state,fm=fit_dmode_field(data.atac,selection['B_D'],data.train)
        np.save(ART/f'{data.name}_F.npy',field)
        np.savez_compressed(ART/f'{data.name}_field_state.npz',**state)
        write(ART/f'{data.name}_field_manifest.json',fm)
    blocks=[np.asarray(d.rna,dtype=np.float32) for d in datasets]
    means=np.stack([x.mean(0) for x in blocks]);mu=means.mean(0).astype(np.float32)
    std=np.sqrt(np.mean(np.stack([x.var(0) for x in blocks])+(means-mu)**2,axis=0)).astype(np.float32)
    np.savez_compressed(ART/'source_scaler.npz',mean=mu,std=std)
    np.save(ART/'gene_ids.npy',genes)
    # Target RNA only. Staging pairing was inherited QC; target ATAC never read here.
    p=SHARED/'data/staging/PBMC/legacy_dmode_data_20261004/derived/common/D1'
    x=sparse.load_npz(p/'rna_counts.npz').tocsr()
    ids=pd.read_csv(p/'genes.tsv',sep='\t').iloc[:,0].astype(str).to_numpy()
    cells=np.asarray((p/'barcodes.tsv').read_text().splitlines())
    assert len(cells)==9347 and x.shape==(9347,36601)
    lookup={g:i for i,g in enumerate(ids)}
    selected=np.asarray([lookup[g] for g in genes])
    target=normalize_log1p(x)[:,selected].toarray().astype(np.float32)
    np.save(ART/'D1_rna_all.npy',target)
    (ART/'D1_barcodes.tsv').write_text('\n'.join(cells)+'\n')
    labs=pd.read_csv(RUN/'input/D1_label_harmonized.csv').set_index('barcode')
    assert labs.index.is_unique and set(labs.index)==set(cells)
    labels=labs.loc[cells,'cell_type'].astype(str)
    labels.to_frame().to_csv(RUN/'audit/D1_allcell_labels.csv')
    readiness.append(dict(dataset='D1',role='TARGET_INFERENCE_ONLY',cells=9347,
        cd8_tem_cells=int((labels=='CD8 TEM').sum()),subclasses=labels.nunique(),rna_sha256=sha(p/'rna_counts.npz'),
        target_atac_loaded=False,hvg_fit=False,scaler_fit=False))
    pd.DataFrame(readiness).to_csv(RUN/'DATA_READINESS.csv',index=False)
    write(ART/'PREPARATION_COMPLETE.json',dict(status='COMPLETE',sources=['D2','D3','D4'],
        target='D1',source_cells=16835,target_cells=9347,gene_count=len(genes),edges=int(selection['B_D'].nnz),
        source_all_cells=True,target_training=False,readiness=readiness,
        hashes={str(p.relative_to(RUN)):sha(p) for p in ART.rglob('*') if p.is_file()}))
    print('PREPARATION_COMPLETE',len(genes),readiness[-1],flush=True)

if __name__=='__main__': main()
