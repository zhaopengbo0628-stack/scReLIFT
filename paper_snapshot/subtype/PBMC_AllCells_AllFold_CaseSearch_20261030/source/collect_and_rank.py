"""Pool all folds/seeds for multiplicity accounting and select exploratory cases transparently."""
import time,sys,json,re,gzip
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
FOLDS=['D1','D2','D3','D4']
sys.path.insert(0,str(ROOT/'analyses/D1/source'))
from analysis_common import bh,sha,write
Q='FDR_BH_all_folds_all_seeds'
IMMUNE=r'cytotox|lymph|t cell|b cell|interferon|antigen|chemokin|leukocyte|myeloid|immun|monocyt|natural killer|inflamm'


def wait(name):
    for fold in FOLDS:
        p=ROOT/'analyses'/fold/'audit'/name
        while not p.exists():time.sleep(30)


def pool_gene_tests():
    comparisons=[];pvals=[]
    for fold in FOLDS:
        run=ROOT/'analyses'/fold
        manifest=pd.read_csv(run/'all_candidate_deg_summary.csv')
        for row in manifest.to_dict('records'):
            path=run/row['path']
            p=pd.read_csv(path,usecols=['p_value']).p_value.to_numpy()
            assert len(p)==36601 and np.isfinite(p).all()
            comparisons.append(dict(fold=fold,input_file=str(path),**row));pvals.append(p)
    q=bh(np.concatenate(pvals));out=ROOT/'deg_allfold_global';out.mkdir(exist_ok=True)
    offsets=0;summary=[];markers=[]
    for item in comparisons:
        d=pd.read_csv(item['input_file']);n=len(d)
        d[Q]=q[offsets:offsets+n];offsets+=n
        p=out/Path(item['input_file']).name
        d.to_csv(p,index=False,compression='gzip')
        item.update(allfold_global_file=str(p),allfold_global_sha256=sha(p),
            allfold_strict_effect025=int(((d[Q]<.05)&d.log2FC_meanCP10K.abs().gt(.25)).sum()))
        summary.append(item)
        if item['contrast']!='A_vs_B':continue
        good=(d[Q]<.05)&d.log2FC_meanCP10K.abs().gt(.5)&((d.pct_detected_A>=.1)|(d.pct_detected_B>=.1))
        for sign,direction in [(1,'A_high'),(-1,'B_high')]:
            dd=d[good&(d.log2FC_meanCP10K*sign>0)].copy()
            dd['abs_effect']=dd.log2FC_meanCP10K.abs()
            for gene in dd.sort_values(['abs_effect',Q],ascending=[False,True]).head(20).itertuples():
                markers.append(dict(fold=item['fold'],candidate_id=item['candidate_id'],seed=item['seed'],direction=direction,
                    gene_symbol=gene.gene_symbol,gene_id=gene.gene_id,log2FC=gene.log2FC_meanCP10K,global_FDR=getattr(gene,Q),
                    pct_A=gene.pct_detected_A,pct_B=gene.pct_detected_B))
        item['allfold_measured_markers']=int(good.sum())
        item['allfold_A_markers']=int((good&d.log2FC_meanCP10K.gt(0)).sum())
        item['allfold_B_markers']=int((good&d.log2FC_meanCP10K.lt(0)).sum())
    pd.DataFrame(summary).to_csv(ROOT/'all_fold_all_seed_DEG_summary.csv',index=False)
    pd.DataFrame(markers).to_csv(ROOT/'all_fold_all_seed_top_markers.csv',index=False)
    write(ROOT/'audit/ALL_FOLD_GENE_BH_COMPLETE.json',dict(status='COMPLETE',comparisons=len(comparisons),gene_tests=len(q),
        all_cells_full_gene_tests=True,all_available_seeds=True,original_per_fold_tables_unchanged=True))
    return pd.DataFrame(summary)


def pool_pathways(relative,output_name):
    inputs=[ROOT/'analyses'/fold/relative for fold in FOLDS]
    cols=['nominal_p'] if 'ora' in relative else ['nominal_p_floored']
    raw_pvalues=[pd.read_csv(p,usecols=cols).iloc[:,0].to_numpy() for p in inputs]
    # fgsea can return NA for an unestimable null tail. Keep the hypothesis in the
    # family with P=1; preserve raw NA and never turn it into positive evidence.
    pvalues=[np.where(np.isfinite(p)&(p>=0)&(p<=1),p,1.) for p in raw_pvalues]
    q=bh(np.concatenate(pvalues));offset=0;summaries=[];top=[]
    dest=ROOT/output_name
    with gzip.open(dest,'wt',encoding='utf-8',newline='') as handle:
        first=True
        for fold,path in zip(FOLDS,inputs):
            for d in pd.read_csv(path,chunksize=25000):
                d['fold']=fold;d[Q]=q[offset:offset+len(d)];offset+=len(d)
                d['pathway_test_estimable']=np.isfinite(d[cols[0]])&d[cols[0]].between(0,1)
                d['P_used_for_global_BH']=d[cols[0]].where(d.pathway_test_estimable,1.)
                if 'NES' in d:
                    d['pathway_test_estimable'] &= np.isfinite(d.NES)
                d.to_csv(handle,index=False,header=first);first=False
                keep=d[d.contrast.eq('A_vs_B')].copy()
                keep['immune_program_term']=keep.Term.str.contains(IMMUNE,case=False,regex=True)
                for keys,g in keep.groupby(['fold','candidate_id','seed','database']):
                    sig=g[Q]<.05
                    biological=sig&g.immune_program_term
                    summaries.append(dict(zip(['fold','candidate_id','seed','database'],keys),
                        tested_terms=len(g),significant_terms=int(sig.sum()),immune_terms=int(biological.sum())))
                    top.append(g.sort_values([Q,cols[0]]).head(12))
    compact=pd.DataFrame(summaries).groupby(['fold','candidate_id','seed','database'],as_index=False)[['tested_terms','significant_terms','immune_terms']].sum()
    compact.to_csv(ROOT/(output_name.replace('.csv.gz','_compact.csv')),index=False)
    top=pd.concat(top,ignore_index=True).sort_values([Q,cols[0]]).groupby(['fold','candidate_id','seed','database'],sort=False).head(12)
    top.to_csv(ROOT/(output_name.replace('.csv.gz','_top_terms.csv')),index=False)
    write(ROOT/'audit'/(output_name.replace('.csv.gz','_BH_AUDIT.json')),dict(status='COMPLETE',
        hypotheses=int(sum(map(len,pvalues))),unestimable_P=int(sum((~np.isfinite(p)).sum() for p in raw_pvalues)),
        policy='raw P/NES missing values preserved; unestimable tests included conservatively as P=1; not evidence',
        all_global_q_finite=bool(np.isfinite(q).all()),global_BH_family='all folds/candidates/seeds/contrasts/databases in this analysis universe'))
    return compact


def rank_candidates(deg,gsea,expressed,ora):
    ev=pd.concat([pd.read_csv(ROOT/'analyses'/fold/'candidate_seed_RNA_UMAP_evidence.csv') for fold in FOLDS],ignore_index=True)
    selected=deg[deg.contrast.eq('A_vs_B')][['candidate_id','seed','allfold_global_file','allfold_strict_effect025','allfold_measured_markers','allfold_A_markers','allfold_B_markers']]
    ev=ev.merge(selected,on=['candidate_id','seed'],validate='one_to_one')
    for table,prefix in [(gsea,'GSEA_full'),(expressed,'GSEA_expressed'),(ora,'ORA')]:
        for db,dbshort in [('GO_BP_2025','GO'),('Reactome_2022','Reactome')]:
            t=table[table.database.eq(db)].drop(columns=['database']).rename(columns={k:f'{prefix}_{dbshort}_{k}' for k in ['tested_terms','significant_terms','immune_terms']})
            ev=ev.merge(t,on=['fold','candidate_id','seed'],how='left',validate='one_to_one')
    ev['markers_both_directions']=ev[['allfold_A_markers','allfold_B_markers']].min(axis=1)
    ev['immune_ORA_terms']=ev.ORA_GO_immune_terms+ev.ORA_Reactome_immune_terms
    ev['immune_expressed_GSEA_terms']=ev.GSEA_expressed_GO_immune_terms+ev.GSEA_expressed_Reactome_immune_terms
    ev['functional_support_tier']=(ev.immune_ORA_terms.gt(0).astype(int)+ev.immune_expressed_GSEA_terms.gt(0).astype(int))
    ev['adequate_measured_RNA_effect']=ev.markers_both_directions.ge(3)&ev.allfold_measured_markers.ge(10)
    ev['no_large_measured_QC_association']=ev.QC_association_count.eq(0)
    ev['functional_program_count_capped']=ev.immune_ORA_terms.clip(upper=10)+ev.immune_expressed_GSEA_terms.clip(upper=10)
    ev['measured_marker_count_capped']=ev.allfold_measured_markers.clip(upper=100)
    # First molecular completeness and QC; then UMAP; then cross-seed reproducibility.
    sort=['adequate_measured_RNA_effect','functional_support_tier','no_large_measured_QC_association',
          'functional_program_count_capped','measured_marker_count_capped',
          'UMAP_silhouette_gain_vs_PCA_sourcefit','UMAP_silhouette_gain_vs_PCA_targetfit_fullnorm',
          'mean_partition_ARI','mean_member_Jaccard']
    ev=ev.sort_values(sort,ascending=False,na_position='last')
    ev.insert(0,'descriptive_evidence_rank',range(1,len(ev)+1))
    ev.to_csv(ROOT/'candidate_seed_evidence_ranking.csv',index=False)
    b2=ev[ev.method.eq('FullB2')].copy()
    b2.to_csv(ROOT/'FullB2_candidate_seed_ranking.csv',index=False)
    # Representative seed is a posthoc choice; preserve all other candidates and seeds.
    shortlist=b2.drop_duplicates('candidate_id').head(8)
    shortlist.to_csv(ROOT/'CASE_SHORTLIST.csv',index=False)
    write(ROOT/'audit/CASE_SEARCH_READY_FOR_BIOLOGICAL_REVIEW.json',dict(status='COMPLETE',seed_cases=len(ev),
        method_candidates=ev.candidate_id.nunique(),FullB2_seed_cases=len(b2),folds=FOLDS,
        selection='posthoc representative seed allowed by user; all candidates/seeds retained',
        priority='DEG and functional program support, then matched-cell UMAP advantages, then Jaccard/partition stability',
        ranking='descriptive triage, not proof of a new cell subtype; manual interpretation still required'))
    print('CASE_SEARCH_READY_FOR_BIOLOGICAL_REVIEW',shortlist[['candidate_id','seed']].to_json(orient='records'),flush=True)


def main():
    wait('DEG_COMPLETE.json');deg=pool_gene_tests()
    wait('RNA_UMAP_EVIDENCE_COMPLETE.json');wait('ALL_SEED_ANALYSIS_COMPLETE.json')
    gs=pool_pathways('fgsea_all_seeds/all_candidate_gsea_summary.csv.gz','all_fold_all_seed_GSEA_full.csv.gz')
    ex=pool_pathways('fgsea_all_seeds_expressed/all_candidate_gsea_summary.csv.gz','all_fold_all_seed_GSEA_expressed.csv.gz')
    ora=pool_pathways('fgsea_all_seeds/all_candidate_ora_summary.csv.gz','all_fold_all_seed_ORA.csv.gz')
    rank_candidates(deg,gs,ex,ora)


if __name__=='__main__':main()
