"""GO/Reactome for every predefined anchor contrast, using full-gene ranks."""
import sys,time,traceback
from scipy.stats import hypergeom
from analysis_common import *
sys.path.insert(0,str(RUN/'vendor'))
import gseapy as gp
DBS={'GO_BP_2025':'GO_Biological_Process_2025.gmt','Reactome_2022':'Reactome_2022.gmt'}
PERM=10000

def read_gmt(path):
    terms={}
    for line in Path(path).read_text().splitlines():
        fields=line.split('\t')
        terms[fields[0]]=set(g for g in fields[2:] if g)
    return terms

def rank_input(d):
    # Retain ALL tested genes, including tied/zero statistics; no significant-gene truncation.
    r=d[['gene_id','gene_symbol','wilcoxon_score_signed']].copy()
    r=r.replace([np.inf,-np.inf],np.nan).dropna()
    r=r[r.gene_symbol.astype(str).str.len()>0]
    # Resolve duplicate symbols deterministically by greatest absolute statistic, then ID.
    r['abs_score']=r.wilcoxon_score_signed.abs()
    r=r.sort_values(['abs_score','gene_id'],ascending=[False,True],kind='stable').drop_duplicates('gene_symbol')
    return r.sort_values(['wilcoxon_score_signed','gene_id'],ascending=[False,True],kind='stable')

def ora(d,terms,meta):
    universe=set(d.loc[d.expressed_in_comparison,'gene_symbol'].astype(str))
    rows=[]
    for direction,sign in [('A_high',1),('B_high',-1)]:
        hits=set(d.loc[(d.FDR_BH_global_all_candidate_seed_contrast_tests<.05)&
                     (d.log2FC_meanCP10K*sign>.25)&d.expressed_in_comparison,'gene_symbol'].astype(str))
        for term,genes in terms.items():
            target=genes&universe
            if not 10<=len(target)<=500:continue
            overlap=hits&target
            p=float(hypergeom.sf(len(overlap)-1,len(universe),len(target),len(hits))) if hits else 1.
            rows.append(dict(**meta,direction=direction,Term=term,nominal_p=p,overlap=len(overlap),
                       overlap_genes=';'.join(sorted(overlap)),gene_set_matched=len(target),
                       universe_genes=len(universe),DEG_genes=len(hits),
                       odds_enrichment=(len(overlap)/len(hits))/(len(target)/len(universe)) if hits else 0.,
                       DEG_rule='global all-candidate/seed/contrast BH<0.05 & abs(log2FC)>0.25',
                       background='expressed symbols in same comparison, not whole genome'))
    return pd.DataFrame(rows)

def main():
    (RUN/'enrichment').mkdir(exist_ok=True)
    # A small true algorithm witness catches private dependency issues without touching model environment.
    genes=[f'G{i}' for i in range(120)]
    test=gp.prerank(rnk=pd.DataFrame({'gene':genes,'score':np.linspace(3,-3,120)}),
        gene_sets={'positive':genes[:15],'negative':genes[-15:],'mixed':genes[::5]},
        permutation_num=32,min_size=5,max_size=100,seed=20020130,threads=2,outdir=None,verbose=False)
    assert len(test.res2d)>0 and np.isfinite(pd.to_numeric(test.res2d.NES)).all()
    write(RUN/'audit/GSEA_CPU_WITNESS.json',dict(status='PASS',gseapy=gp.__version__,
          private_import=str(Path(gp.__file__).resolve()),synthetic_terms=len(test.res2d),
          CPU_only=True,base_environment_modified=False))
    while not (RUN/'audit/DEG_COMPLETE.json').exists():time.sleep(30)
    comparisons=pd.read_csv(RUN/'tables/anchor_enrichment_inputs.csv')
    assert len(comparisons)==18*3
    databases={db:read_gmt(RUN/'input'/filename) for db,filename in DBS.items()}
    ora_all=[];gsea_all=[];statuses=[];failures=[]
    for rec in comparisons.to_dict('records'):
        d=pd.read_csv(RUN/rec['path']);rank=rank_input(d)
        cid=rec['candidate_id'];contrast=rec['contrast'];seed=int(rec['seed'])
        rank.to_csv(RUN/'enrichment'/f'{cid}__{contrast}__rank.csv',index=False)
        meta=dict(candidate_id=cid,method=rec['method'],reference_subclass=rec['reference_subclass'],
                  seed=seed,contrast=contrast,anchor_rule='lowest available fixed seed; chosen before DEG')
        for db,filename in DBS.items():
            gmt=RUN/'input'/filename
            prefix=RUN/'enrichment'/f'{cid}__{contrast}__{db}'
            m=dict(**meta,database=db,gmt_sha256=sha(gmt))
            ort=ora(d,databases[db],m);ort.to_csv(prefix.with_suffix('.ora.csv.gz'),index=False,compression='gzip');ora_all.append(ort)
            try:
                cached=prefix.with_suffix('.gsea.csv.gz')
                if cached.exists():table=pd.read_csv(cached)
                else:
                    started=time.perf_counter()
                    print('GSEA_START',cid,contrast,db,'genes',len(rank),flush=True)
                    res=gp.prerank(rnk=rank[['gene_symbol','wilcoxon_score_signed']],gene_sets=str(gmt),
                       permutation_num=PERM,min_size=10,max_size=500,seed=20020130,threads=2,outdir=None,verbose=False)
                    table=res.res2d.copy()
                    table['raw_nominal_p']=pd.to_numeric(table['NOM p-val'],errors='coerce').fillna(1.)
                    table['nominal_p_floored']=table.raw_nominal_p.clip(lower=1/(PERM+1))
                    table['NES']=pd.to_numeric(table.NES);table['ES']=pd.to_numeric(table.ES)
                    for key,value in m.items():table[key]=value
                    universe=set(rank.gene_symbol.astype(str))
                    table['matched_genes']=table.Term.map(lambda t:len(databases[db][t]&universe))
                    table['n_ranked_genes']=len(rank);table['permutations']=PERM
                    table['rank_tie_rule']='score descending, Ensembl ID ascending; no jitter'
                    table['lead_gene_count']=table.Lead_genes.fillna('').str.split(';').map(len)
                    table.to_csv(cached,index=False,compression='gzip')
                    print('GSEA_DONE',cid,contrast,db,len(table),'seconds',round(time.perf_counter()-started,1),flush=True)
                gsea_all.append(table)
                statuses.append(dict(**m,status='COMPLETE',terms=len(table),ranked_genes=len(rank),permutations=PERM))
            except Exception:
                error=traceback.format_exc();failures.append(dict(**m,error=error))
                statuses.append(dict(**m,status='FAILED',error=error))
                write(RUN/'audit/GSEA_FAILURES.json',failures)
            pd.DataFrame(statuses).to_csv(RUN/'tables/enrichment_job_status.csv',index=False)
    allora=pd.concat(ora_all,ignore_index=True)
    allora['FDR_BH_global_all_candidate_contrast_database_direction_tests']=bh(allora.nominal_p.to_numpy())
    allora['FDR_BH_within_comparison']=np.nan
    for _,ix in allora.groupby(['candidate_id','contrast']).groups.items():
        allora.loc[ix,'FDR_BH_within_comparison']=bh(allora.loc[ix,'nominal_p'].to_numpy())
    allora.to_csv(RUN/'all_candidate_ora_summary.csv.gz',index=False,compression='gzip')
    if gsea_all:
        gs=pd.concat(gsea_all,ignore_index=True)
        gs['FDR_BH_global_all_candidate_contrast_GO_Reactome_tests']=bh(gs.nominal_p_floored.to_numpy())
        gs['FDR_BH_within_candidate_contrast_both_databases']=np.nan
        for _,ix in gs.groupby(['candidate_id','contrast']).groups.items():
            gs.loc[ix,'FDR_BH_within_candidate_contrast_both_databases']=bh(gs.loc[ix,'nominal_p_floored'].to_numpy())
        gs['leading_edge_genes']=gs.Lead_genes.fillna('')
        gs['inference_scope']='gene-set permutation on posthoc cell-level ranks; not independent validation'
        gs.to_csv(RUN/'all_candidate_gsea_summary.csv.gz',index=False,compression='gzip')
    else:gs=pd.DataFrame()
    write(RUN/'audit/ENRICHMENT_COMPLETE.json',dict(status='COMPLETE' if not failures else 'PARTIAL',
        anchor_candidate_contrasts=len(comparisons),gsea_runs=len(statuses),failed=len(failures),gsea_terms=len(gs),ora_terms=len(allora),
        gseapy=gp.__version__,permutations=PERM,p_floor=1/(PERM+1),
        rank='all tested genes, signed frozen Wilcoxon statistic; no significant-gene filtering',
        duplicate_symbol_rule='largest absolute statistic then gene ID',
        zero_p_handling='floor nominal permutation P to 1/(10000+1), never zero FDR',
        strict_ora='global-gene BH<0.05 with effect>0.25; expressed-gene universe',
        global_gsea_BH_family='all anchor candidate x 3 contrasts x GO/Reactome pathways',
        databases={db:dict(file=fn,sha256=sha(RUN/'input'/fn),terms=len(databases[db])) for db,fn in DBS.items()},
        caveats=['posthoc candidate selection','seeds are not biological replicates','no independent donor replication',
                 'anchor excludes selecting best biological/P-value seed; all-seed DEG direction remains separately available']))
    print('ENRICHMENT_COMPLETE',len(gs),'GSEA tests',len(allora),'ORA tests',flush=True)

if __name__=='__main__':main()
