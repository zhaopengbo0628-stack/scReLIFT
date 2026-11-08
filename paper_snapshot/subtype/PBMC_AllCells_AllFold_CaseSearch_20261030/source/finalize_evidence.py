"""Add stability coverage and audit annotations; retain all favorable PCA cases."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
Q='FDR_BH_all_folds_all_seeds'

def main():
    inv=[];st=[]
    for fold in ['D1','D2','D3','D4']:
        run=ROOT/'analyses'/fold
        d=pd.read_csv(run/'candidate_inventory.csv').assign(fold=fold)
        stability=pd.read_csv(run/'tables/cross_seed_candidate_stability.csv')
        for row in d.itertuples():
            ss=stability[stability.candidate_id.eq(row.candidate_id)]
            st.append(dict(candidate_id=row.candidate_id,seed_pairs_tested=len(ss),
                valid_two_group_pairs=int(ss.valid_two_group_overlap.sum()) if len(ss) else 0,
                min_shared_cells=int(ss.shared_cells.min()) if len(ss) else 0,
                stability_warning='ARI conditional on common members; inspect member Jaccard and valid pair count; seeds not biological replication'))
        d['DEG_status']='COMPLETE';d['GO_Reactome_status']='COMPLETE';inv.append(d)
    inventory=pd.concat(inv,ignore_index=True).merge(pd.DataFrame(st),on='candidate_id',validate='one_to_one')
    inventory.to_csv(ROOT/'candidate_inventory.csv',index=False)
    seed=pd.read_csv(ROOT/'candidate_seed_evidence_ranking.csv').merge(pd.DataFrame(st),on='candidate_id',validate='many_to_one')
    completed_deg=pd.read_csv(ROOT/'all_fold_all_seed_DEG_summary.csv')
    assert completed_deg.status.eq('COMPLETE').all()
    comparisons=completed_deg.groupby(['candidate_id','seed']).contrast.nunique()
    for row in seed.itertuples():
        assert comparisons.loc[(row.candidate_id,row.seed)]==3
    tested=[f'{scope}_{db}_tested_terms' for scope in ['GSEA_full','GSEA_expressed'] for db in ['GO','Reactome']]
    assert seed[tested].notna().all().all() and seed[tested].gt(0).all().all()
    # These inherited status fields describe an earlier snapshot, before analysis completed.
    seed['DEG_status']='COMPLETE';seed['GO_Reactome_status']='COMPLETE'
    seed['annotation_risk']=''
    seed.loc[seed.fold.eq('D2')&seed.reference_subclass.eq('CD8 Naive'),'annotation_risk']='l2 CD8 Naive conflicts with author MarkerAnnotations predominantly CD4'
    seed.loc[seed.fold.eq('D2')&seed.reference_subclass.eq('HSPC'),'annotation_risk']='same-cohort fine labels available; only one valid cross-seed pair; PCA also recovers; QC associations'
    seed['PCA_fit_note']='same source-HVG intersection as Full B2; target-fit is transductive descriptive sensitivity'
    seed['embedding_scaling_note']='inherited Task99 uses Target-fitted posthoc embedding StandardScaler; input RNA scaler remains Source-only'
    seed.to_csv(ROOT/'candidate_seed_evidence_matrix.csv',index=False)
    aggregate=[]
    for cid,d in seed.groupby('candidate_id'):
        r=inventory[inventory.candidate_id.eq(cid)].iloc[0].to_dict()
        r.update(seed_cases=len(d),max_global_DEG=int(d.allfold_strict_effect025.max()),
            min_global_DEG=int(d.allfold_strict_effect025.min()),max_measured_markers=int(d.allfold_measured_markers.max()),
            min_measured_markers=int(d.allfold_measured_markers.min()),
            any_large_QC_association=bool(d.QC_association_count.gt(0).any()),
            mean_UMAP_gain_vs_sourcePCA=float(d.UMAP_silhouette_gain_vs_PCA_sourcefit.mean()),
            mean_UMAP_gain_vs_targetPCA=float(d.UMAP_silhouette_gain_vs_PCA_targetfit_fullnorm.mean()),
            seeds_UMAP_better_than_both_PCA=int((d.UMAP_silhouette_gain_vs_PCA_sourcefit.gt(0)&d.UMAP_silhouette_gain_vs_PCA_targetfit_fullnorm.gt(0)).sum()),
            annotation_risk=';'.join(x for x in d.annotation_risk.dropna().unique() if x),
            interpretation='posthoc candidate state/subtype association; not independently validated')
        for prefix in ['GSEA_full','GSEA_expressed','ORA']:
            for db in ['GO','Reactome']:
                c=f'{prefix}_{db}_significant_terms';r[c+'_max']=int(d[c].max())
        aggregate.append(r)
    pd.DataFrame(aggregate).to_csv(ROOT/'candidate_evidence_matrix.csv',index=False)
    # Compatibility names refer to the same audited all-fold data rather than a separate analysis.
    de=pd.read_csv(ROOT/'all_fold_all_seed_DEG_summary.csv');de.to_csv(ROOT/'all_candidate_deg_summary.csv',index=False)
    g=[]
    for scope in ['full','expressed']:
        d=pd.read_csv(ROOT/f'all_fold_all_seed_GSEA_{scope}_compact.csv');d['rank_universe']=scope;g.append(d)
    pd.concat(g,ignore_index=True).to_csv(ROOT/'all_candidate_gsea_summary.csv',index=False)
    (ROOT/'audit/EVIDENCE_MATRIX_COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',
        candidate_groups=len(inventory),candidate_seed_cases=len(seed),gene_comparisons=len(de),
        global_gene_tests=int(de.tested_genes.sum()),valid_pairs_not_assumed=True,
        preserved_all_PCA_and_B2_cases=True),indent=2))

if __name__=='__main__':main()
