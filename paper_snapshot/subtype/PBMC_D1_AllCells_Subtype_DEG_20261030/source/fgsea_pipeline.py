"""Fast 10,000-permutation primary GSEA; existing GSEApy run remains sensitivity only."""
import subprocess,time,os
from analysis_common import *
from enrichment import read_gmt,rank_input,ora,DBS

def main():
    while not (RUN/'audit/DEG_COMPLETE.json').exists():time.sleep(30)
    while not (RUN/'R_library/fgsea/DESCRIPTION').exists():time.sleep(30)
    target=RUN/'fgsea';target.mkdir(exist_ok=True)
    comparisons=pd.read_csv(RUN/'tables/anchor_enrichment_inputs.csv')
    tasks=[];oras=[]
    databases={db:read_gmt(RUN/'input'/fn) for db,fn in DBS.items()}
    for rec in comparisons.to_dict('records'):
        cid=rec['candidate_id'];contrast=rec['contrast'];seed=int(rec['seed'])
        d=pd.read_csv(RUN/rec['path']);rank=rank_input(d)
        rankfile=target/f'{cid}__{contrast}__rank.csv';rank.to_csv(rankfile,index=False)
        resultfile=target/f'{cid}__{contrast}__results.csv'
        tasks.append(dict(candidate_id=cid,method=rec['method'],reference_subclass=rec['reference_subclass'],
                 seed=seed,contrast=contrast,rank_file=str(rankfile),result_file=str(resultfile)))
        for db,fn in DBS.items():
            meta=dict(candidate_id=cid,method=rec['method'],reference_subclass=rec['reference_subclass'],
                seed=seed,contrast=contrast,anchor_rule='lowest available fixed seed; chosen before DEG',
                database=db,gmt_sha256=sha(RUN/'input'/fn))
            oras.append(ora(d,databases[db],meta))
    # Fixed priority only affects delivery order: historic user case, stable NK, PCA control, then all remaining.
    jobs=pd.DataFrame(tasks)
    priority={'D1_ALL_FullB2_CD8_TEM_C1':0,'D1_ALL_FullB2_NK_C1':1,'D1_ALL_PCA_sourcefit_CD4_Naive_C1':2}
    jobs['priority']=jobs.candidate_id.map(priority).fillna(3).astype(int)
    order={'A_vs_B':0,'A_vs_subclass_rest':1,'B_vs_subclass_rest':2}
    jobs['contrast_priority']=jobs.contrast.map(order)
    jobs=jobs.sort_values(['priority','candidate_id','contrast_priority'],kind='stable')
    jobs.to_csv(target/'jobs.csv',index=False)
    all_ora=pd.concat(oras,ignore_index=True)
    all_ora['FDR_BH_global_all_candidate_contrast_database_direction_tests']=bh(all_ora.nominal_p.to_numpy())
    all_ora['FDR_BH_within_comparison']=np.nan
    for _,ix in all_ora.groupby(['candidate_id','contrast']).groups.items():
        all_ora.loc[ix,'FDR_BH_within_comparison']=bh(all_ora.loc[ix,'nominal_p'].to_numpy())
    all_ora.to_csv(target/'all_candidate_ora_summary.csv.gz',index=False,compression='gzip')
    print('ORA_COMPLETE',len(all_ora),flush=True)
    subprocess.run(['/home/u202522802045/envs/supermap_r/bin/Rscript',str(RUN/'source/run_fgsea.R'),str(RUN)],check=True,env=os.environ.copy())
    tables=[]
    for job in jobs.itertuples():
        d=pd.read_csv(job.result_file)
        d['raw_nominal_p']=d.pval
        d['nominal_p_floored']=d.pval.clip(lower=1/10001)
        d['FDR q-val']=d.padj
        d['Lead_genes']=d.leadingEdge.fillna('')
        d['leading_edge_genes']=d.leadingEdge.fillna('')
        d['matched_genes']=d['size']
        d['lead_gene_count']=d.leading_edge_genes.str.split(';').map(len)
        d['gmt_sha256']=d.database.map({db:sha(RUN/'input'/fn) for db,fn in DBS.items()})
        d['inference_scope']='gene-set permutation on posthoc cell-level ranks; no independent validation'
        tables.append(d)
    gs=pd.concat(tables,ignore_index=True)
    gs['FDR_BH_global_all_candidate_contrast_GO_Reactome_tests']=bh(gs.nominal_p_floored.to_numpy())
    gs['FDR_BH_within_candidate_contrast_both_databases']=np.nan
    for _,ix in gs.groupby(['candidate_id','contrast']).groups.items():
        gs.loc[ix,'FDR_BH_within_candidate_contrast_both_databases']=bh(gs.loc[ix,'nominal_p_floored'].to_numpy())
    gs.to_csv(target/'all_candidate_gsea_summary.csv.gz',index=False,compression='gzip')
    write(RUN/'audit/FGSEA_COMPLETE.json',dict(status='COMPLETE',anchor_candidate_contrasts=len(jobs),
        database_GSEA_analyses=108,permutations=10000,terms=len(gs),ORA_tests=len(all_ora),
        implementation=gs.implementation.iloc[0],ranked_genes=int(gs.n_ranked_genes.min()),
        global_BH_family='all 18 candidates x 3 contrasts x both GO/Reactome databases',
        p_floor=1/10001,private_R_library=str(RUN/'R_library'),existing_model_environment_modified=False,
        databases={db:dict(file=fn,sha256=sha(RUN/'input'/fn)) for db,fn in DBS.items()},
        gseapy_note='Earlier GSEApy 1.1.10 worker continues separately as sensitivity; primary files live under fgsea/ and cannot be overwritten by it'))
    print('PRIMARY_FGSEA_COMPLETE',flush=True)

if __name__=='__main__':main()
