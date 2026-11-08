"""Full-library RNA cell-level tests; all candidate seeds and all predefined contrasts."""
import itertools,time
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
from analysis_common import *

MODULES={
 'Naive_memory':['CCR7','TCF7','LEF1','SELL','IL7R','LTB','MAL'],
 'Cytotoxic':['NKG7','GNLY','GZMB','GZMH','PRF1','FGFBP2','CTSW'],
 'GZMK_effector':['GZMK','CCL5','GZMA','CST7','KLRG1'],
 'Interferon':['ISG15','IFIT1','IFIT2','IFIT3','IFI6','MX1','OAS1','STAT1'],
 'Immediate_early':['FOS','JUN','JUNB','DUSP1','IER2','NR4A1'],
 'Cell_cycle':['MKI67','TOP2A','STMN1','TYMS','HMGB2','PCNA','MCM5','CCNB1','CDC20'],
 'Myeloid':['LYZ','S100A8','S100A9','FCN1','CTSS','LST1','TYROBP'],
 'B_cell':['MS4A1','CD79A','CD79B','CD74','CD37','HLA-DRA'],
 'T_cell':['CD3D','CD3E','CD3G','TRAC','CD247','LCK'],
 'CD8':['CD8A','CD8B'],'CD4':['CD4','IL7R','MAL'],
}

def main():
    while not (RUN/'audit/INVENTORY_COMPLETE.json').exists(): time.sleep(15)
    cells,labels=context()
    root=SHARED/'data/staging/PBMC/legacy_dmode_data_20261004/derived/common/D1'
    x=sparse.load_npz(root/'rna_counts.npz').tocsr()
    ids=pd.read_csv(root/'genes.tsv',sep='\t').iloc[:,0].astype(str).to_numpy(dtype=str)
    symbols=pd.read_csv(root/'genes.tsv',sep='\t').iloc[:,1].astype(str).to_numpy(dtype=str)
    assert x.shape==(9347,36601) and np.array_equal(cells,np.asarray((root/'barcodes.tsv').read_text().splitlines()))
    assert np.isfinite(x.data).all() and np.all(x.data>=0) and np.array_equal(x.data,np.round(x.data))
    totals=np.asarray(x.sum(1)).ravel().astype(float)
    detected=x.getnnz(1)
    mt=np.char.startswith(symbols,'MT-')
    qc=pd.DataFrame(dict(barcode=cells,reference_label=labels,RNA_UMI=totals,detected_genes=detected,
       percent_mitochondrial=100*np.asarray(x[:,mt].sum(1)).ravel()/np.maximum(1,totals)))
    log=x.multiply((1e4/np.maximum(totals,1))[:,None]).tocsr().astype(float);log.data=np.log1p(log.data)
    gene_mu=np.asarray(log.mean(0)).ravel();gene_std=np.sqrt(np.maximum(np.asarray(log.power(2).mean(0)).ravel()-gene_mu**2,0))
    module_rows=[]
    for name,genes in MODULES.items():
        ix=np.flatnonzero(np.isin(symbols,genes));assert len(ix)>0
        values=log[:,ix].toarray()
        qc['module_'+name]=((values-gene_mu[ix])/(gene_std[ix]+1e-6)).mean(1)
        module_rows.append(dict(module=name,matched_genes=';'.join(symbols[ix]),requested_genes=';'.join(genes),
              scoring='mean per-gene z(log1p CP10K), descriptive; no control-gene subtraction, not Seurat AddModuleScore'))
    # Cell-level metadata and actual ATAC total counts only in posthoc analysis of frozen embeddings.
    atac=sparse.load_npz(root/'atac_counts.npz').tocsr()
    assert atac.shape[0]==9347
    qc['ATAC_peak_read_counts']=np.asarray(atac.sum(1)).ravel()
    qc['ATAC_detected_peaks']=atac.getnnz(1)
    qc.to_csv(RUN/'tables/cell_qc_and_module_scores.csv',index=False)
    pd.DataFrame(module_rows).to_csv(RUN/'tables/module_definitions.csv',index=False)
    memberships=pd.read_csv(RUN/'tables/candidate_membership.csv.gz')
    screen=pd.read_csv(RUN/'tables/candidate_screen_by_seed.csv')
    (RUN/'deg').mkdir(exist_ok=True)
    manifest=[];qcrows=[];descriptive=[];all_p=[]
    for (cid,seed),mem in memberships.groupby(['candidate_id','seed'],sort=False):
        method=mem.method.iloc[0];label=mem.reference_subclass.iloc[0]
        groups=pd.Series('',index=cells);groups.loc[mem.barcode]=mem.group.to_numpy()
        a=groups.eq('A').to_numpy();b=groups.eq('B').to_numpy();same=labels==label
        # Groups contain only the named Reference subclass; no other label is silently included.
        assert np.all(same[a|b]) and min(a.sum(),b.sum())>=10
        for contrast,left,right in [('A_vs_B',a,b),('A_vs_subclass_rest',a,same&~a),('B_vs_subclass_rest',b,same&~b)]:
            if min(left.sum(),right.sum())<10:
                manifest.append(dict(candidate_id=cid,method=method,seed=int(seed),contrast=contrast,status='INSUFFICIENT_CELLS',n_A=int(left.sum()),n_B=int(right.sum())));continue
            path=RUN/'deg'/f'{cid}__{seed}__{contrast}.csv.gz'
            if path.exists():d=pd.read_csv(path)
            else:
                d=frozen.rank_genes(x,left,right,symbols,ids,cid,'D1',int(seed),contrast)
                d['method']=method;d['reference_subclass']=label
                d['expressed_in_comparison']=(d.pct_detected_A>0)|(d.pct_detected_B>0)
                d.to_csv(path,index=False,compression='gzip')
            assert len(d)==36601 and np.isfinite(d.p_value).all()
            all_p.append(d.p_value.to_numpy())
            manifest.append(dict(candidate_id=cid,method=method,seed=int(seed),reference_subclass=label,contrast=contrast,
                status='COMPLETE',n_A=int(left.sum()),n_B=int(right.sum()),tested_genes=len(d),
                expressed_genes=int(d.expressed_in_comparison.sum()),path=str(path.relative_to(RUN)),
                within_FDR_lt_005=int((d.FDR_BH_within_comparison<.05).sum()),
                within_strict_effect025=int(d.strict_DEG.sum())))
            print('DEG_DONE',cid,seed,contrast,'n',left.sum(),right.sum(),'DEG',int(d.strict_DEG.sum()),flush=True)
        # Unified cell-level confounding and module evidence, with effect sizes and missing fields explicit.
        for col in [c for c in qc.columns if c not in ['barcode','reference_label']]:
            va=qc.loc[a,col].to_numpy();vb=qc.loc[b,col].to_numpy()
            result=stats.mannwhitneyu(va,vb,alternative='two-sided',method='asymptotic')
            effect=2*result.statistic/(len(va)*len(vb))-1
            qcrows.append(dict(candidate_id=cid,method=method,seed=int(seed),reference_subclass=label,variable=col,
                    n_A=len(va),n_B=len(vb),median_A=float(np.median(va)),median_B=float(np.median(vb)),
                    mean_A=float(va.mean()),mean_B=float(vb.mean()),rank_biserial=float(effect),p_value=float(result.pvalue),
                    evidence_scope='same Target cell-level posthoc; donor not independent'))
            for name,values in [('A',va),('B',vb)]:
                descriptive.append(dict(candidate_id=cid,method=method,seed=int(seed),group=name,variable=col,
                    n=len(values),mean=float(values.mean()),median=float(np.median(values)),sd=float(values.std(ddof=1))))
    # Global BH includes every tested gene of every predefined candidate, seed and contrast, including p=1 all-zero genes.
    qglobal=bh(np.concatenate(all_p));offset=0;summary=[];markerframes=[];anchor_manifest=[]
    inv=pd.read_csv(RUN/'candidate_inventory.csv').set_index('candidate_id')
    for r in manifest:
        if r['status']!='COMPLETE':continue
        path=RUN/r['path'];d=pd.read_csv(path);n=len(d)
        d['FDR_BH_global_all_candidate_seed_contrast_tests']=qglobal[offset:offset+n];offset+=n
        d['strict_DEG_global']=(d.FDR_BH_global_all_candidate_seed_contrast_tests<.05)&(d.log2FC_meanCP10K.abs()>.25)
        d.to_csv(path,index=False,compression='gzip')
        r.update(global_FDR_lt_005=int((d.FDR_BH_global_all_candidate_seed_contrast_tests<.05).sum()),
             global_strict_effect025=int(d.strict_DEG_global.sum()),sha256=sha(path))
        summary.append(r)
        markerframes.append(d[d.gene_symbol.isin(set(itertools.chain.from_iterable(MODULES.values())))])
        if int(r['seed'])==int(inv.loc[r['candidate_id'],'anchor_seed']):anchor_manifest.append(r)
    pd.DataFrame(manifest).to_csv(RUN/'tables/deg_comparison_manifest.csv',index=False)
    pd.DataFrame(summary).to_csv(RUN/'all_candidate_deg_summary.csv',index=False)
    pd.DataFrame(anchor_manifest).to_csv(RUN/'tables/anchor_enrichment_inputs.csv',index=False)
    pd.concat(markerframes,ignore_index=True).to_csv(RUN/'tables/canonical_marker_statistics.csv.gz',index=False,compression='gzip')
    qct=pd.DataFrame(qcrows);qct['FDR_BH_all_qc_and_module_tests']=bh(qct.p_value.to_numpy())
    qct.to_csv(RUN/'tables/confounder_and_module_tests.csv',index=False)
    pd.DataFrame(descriptive).to_csv(RUN/'tables/confounder_descriptives.csv',index=False)
    # Require aligned effect directions, never average across unrelated seed partitions.
    directions=[]
    for cid,rr in pd.DataFrame(summary).query("contrast=='A_vs_B'").groupby('candidate_id',sort=False):
        dfs=[pd.read_csv(RUN/p).set_index('gene_id') for p in rr.path]
        effects=np.column_stack([d.log2FC_meanCP10K for d in dfs]);pvals=np.column_stack([d.p_value for d in dfs]);qvals=np.column_stack([d.FDR_BH_within_comparison for d in dfs])
        same=(np.sign(effects)==np.sign(effects[:,[0]])).all(1)
        for j,gene in enumerate(dfs[0].index):
            directions.append(dict(candidate_id=cid,gene_id=gene,gene_symbol=dfs[0].gene_symbol.iloc[j],seeds=len(dfs),
                direction_consistent=bool(same[j]),mean_log2FC=float(effects[j].mean()),min_abs_log2FC=float(abs(effects[j]).min()),
                nominal_seeds=int((pvals[j]<.05).sum()),within_FDR_seeds=int((qvals[j]<.05).sum()),
                marker_status='seed-consistent posthoc trend; not biological replication'))
    pd.DataFrame(directions).to_csv(RUN/'tables/gene_direction_consistency.csv.gz',index=False,compression='gzip')
    write(RUN/'audit/DEG_COMPLETE.json',dict(status='COMPLETE',comparisons=len(summary),global_gene_tests=len(qglobal),
        genes_per_comparison=36601,raw_counts_sha256=sha(root/'rna_counts.npz'),
        statistic='frozen Task99 two-sided asymptotic tie-corrected Mann-Whitney on full-library log1p CP10K',
        effect='log2 ratio of mean CP10K with 1e-4 pseudocount',effect_threshold=.25,
        global_BH_family='all candidate x seed x contrast x 36601 genes',
        missing_confounders=['Donor','Batch','Doublet score','ATAC fragments','TSS enrichment'],
        ATAC_counts_note='peak read-count sum is not fragment count; no invented TSS/fragment metadata',
        functional_testing_plan='anchor is lowest available seed selected before statistics; GO/Reactome for all 3 anchor contrasts; all-seed DEG direction reported',
        caveat='Cell-level P values are posthoc descriptive in one cohort; cluster-selection and correlated cells preclude independent discovery confirmation'))
    print('DEG_COMPLETE',len(summary),'comparisons',len(qglobal),'gene tests',flush=True)

if __name__=='__main__':main()
