"""All-seed candidate RNA effects and identical-membership UMAP comparisons."""
import os,sys,time,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

ROOT=Path(__file__).resolve().parents[1]
target=sys.argv[1];run=ROOT/'analyses'/target
while not (run/'audit/DEG_COMPLETE.json').exists():time.sleep(30)
cfg=json.loads((run/'analysis_config.json').read_text());base=Path(cfg['base'])
inv=pd.read_csv(run/'candidate_inventory.csv')
de=pd.read_csv(run/'all_candidate_deg_summary.csv')
membership=pd.read_csv(run/'tables/candidate_membership.csv.gz')
qc=pd.read_csv(run/'tables/confounder_and_module_tests.csv')
recovery=pd.read_csv(run/'tables/fixed_candidate_recovery_all_methods.csv')
genes=set(np.load(base/'artifacts/gene_ids.npy').astype(str))
methods=['FullB2','PCA_sourcefit','PCA_targetfit_fullnorm']
rows=[];markers=[];umaprows=[]
qname='FDR_BH_global_all_candidate_seed_contrast_tests'
for r in de[de.contrast.eq('A_vs_B')].itertuples():
    cid=r.candidate_id;seed=int(r.seed)
    row=inv[inv.candidate_id.eq(cid)].iloc[0].to_dict()
    row.update(fold=target,seed=seed,n_A=int(r.n_A),n_B=int(r.n_B),DEG_file=str(run/r.path))
    d=pd.read_csv(run/r.path)
    sig=(d[qname]<.05)&d.log2FC_meanCP10K.abs().gt(.25)
    measured=sig & d.log2FC_meanCP10K.abs().gt(.5)&((d.pct_detected_A>=.1)|(d.pct_detected_B>=.1))
    row['fold_global_DEG']=int(sig.sum());row['measured_effect_markers']=int(measured.sum())
    row['measured_effect_markers_outside_model']=int((measured&~d.gene_id.isin(genes)).sum())
    row['median_abs_log2FC_measured']=float(d.loc[measured,'log2FC_meanCP10K'].abs().median()) if measured.any() else 0.
    comparisons={}
    for contrast in ['A_vs_subclass_rest','B_vs_subclass_rest']:
        rr=de[de.candidate_id.eq(cid)&de.seed.eq(seed)&de.contrast.eq(contrast)]
        if len(rr):comparisons[contrast]=pd.read_csv(run/rr.iloc[0].path).set_index('gene_id')
    for direction,sign in [('A_high',1),('B_high',-1)]:
        dd=d[measured&(d.log2FC_meanCP10K*sign>0)].copy()
        row[direction+'_measured_markers']=len(dd)
        other='A_vs_subclass_rest' if sign==1 else 'B_vs_subclass_rest'
        if other in comparisons:
            rest=comparisons[other].reindex(dd.gene_id)
            row[direction+'_supported_vs_rest']=int(((rest[qname]<.05)&(rest.log2FC_meanCP10K>.25)).sum())
        else:row[direction+'_supported_vs_rest']=0
        dd['abs_effect']=dd.log2FC_meanCP10K.abs()
        dd=dd.sort_values(['abs_effect',qname],ascending=[False,True]).head(15)
        row[direction+'_top_markers']=';'.join(dd.gene_symbol)
        for gene in dd.itertuples():
            markers.append(dict(fold=target,candidate_id=cid,seed=seed,direction=direction,gene_id=gene.gene_id,
                gene_symbol=gene.gene_symbol,log2FC=gene.log2FC_meanCP10K,fold_global_FDR=getattr(gene,qname),
                pct_A=gene.pct_detected_A,pct_B=gene.pct_detected_B,outside_model=gene.gene_id not in genes))
    qq=qc[qc.candidate_id.eq(cid)&qc.seed.eq(seed)]
    physical=qq[~qq.variable.str.startswith('module_')]
    associated=physical[(physical.FDR_BH_all_qc_and_module_tests<.05)&physical.rank_biserial.abs().ge(.3)]
    row['large_QC_associations']=';'.join(associated.variable)
    row['QC_association_count']=len(associated)
    mm=membership[membership.candidate_id.eq(cid)&membership.seed.eq(seed)].set_index('barcode').group
    truth=(mm=='B').to_numpy(dtype=int)
    for method in methods:
        rr=recovery[recovery.candidate_id.eq(cid)&recovery.seed.eq(seed)&recovery.evaluated_method.eq(method)]
        row['recovery_ARI_'+method]=float(rr.ARI.iloc[0])
        p=base/'figures'/method/f'seed_{seed}/plot_data.csv'
        while not p.exists():time.sleep(15)
        z=pd.read_csv(p).set_index('barcode').loc[mm.index,['UMAP1','UMAP2']].to_numpy()
        sil=float(silhouette_score(z,truth))
        k=min(15,len(truth)-1)
        neighbors=NearestNeighbors(n_neighbors=k).fit(z).kneighbors(return_distance=False)
        correct=(truth[neighbors]==truth[:,None]).mean(1)
        balanced=float(np.mean([correct[truth==v].mean() for v in [0,1]]))
        row['UMAP_silhouette_'+method]=sil;row['UMAP_balanced_purity15_'+method]=balanced
        umaprows.append(dict(fold=target,candidate_id=cid,origin_method=r.method,seed=seed,evaluated_method=method,
            cells=len(mm),silhouette=sil,balanced_purity15=balanced,scope='identical fixed barcodes projected in whole-target UMAP; posthoc origin bias'))
    for comparator in methods[1:]:
        row['UMAP_silhouette_gain_vs_'+comparator]=row['UMAP_silhouette_FullB2']-row['UMAP_silhouette_'+comparator]
        row['UMAP_balanced_purity_gain_vs_'+comparator]=row['UMAP_balanced_purity15_FullB2']-row['UMAP_balanced_purity15_'+comparator]
    rows.append(row)
pd.DataFrame(rows).to_csv(run/'candidate_seed_RNA_UMAP_evidence.csv',index=False)
pd.DataFrame(markers).to_csv(run/'candidate_seed_top_markers.csv',index=False)
pd.DataFrame(umaprows).to_csv(run/'candidate_seed_UMAP_comparison.csv',index=False)
(run/'audit/RNA_UMAP_EVIDENCE_COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',fold=target,seed_candidates=len(rows),all_cells_preserved=True)))
print('RNA_UMAP_EVIDENCE_COMPLETE',target,len(rows),flush=True)
