"""Export fixed selected-case plotting data. No refitting or cell reselection."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
Q='FDR_BH_all_folds_all_seeds'
METHODS=['FullB2','PCA_sourcefit','PCA_targetfit_fullnorm']
SEEDS=[20020130,20020131,20020132]
CANONICAL=['FGFBP2','GZMB','GZMH','PRF1','NKG7','GNLY','IFNG-AS1','CD28','IL7R','GZMK','TCF7','CCR7',
           'TOX','ICOS','ADAM23','ABCB1','DNTT','GP1BB']

def load_terms(filename,selected):
    pieces=[]
    for d in pd.read_csv(ROOT/filename,chunksize=30000):
        hit=d[d.candidate_id.isin(selected)&d.contrast.eq('A_vs_B')]
        if len(hit):pieces.append(hit)
    return pd.concat(pieces,ignore_index=True)

def main():
    cases=json.loads((ROOT/'SELECTED_CASES.json').read_text())
    terms={name:load_terms(fn,[c['candidate_id'] for c in cases]) for name,fn in [
        ('GSEA_full','all_fold_all_seed_GSEA_full.csv.gz'),
        ('GSEA_expressed','all_fold_all_seed_GSEA_expressed.csv.gz'),
        ('ORA','all_fold_all_seed_ORA.csv.gz')]}
    ranking=pd.read_csv(ROOT/'candidate_seed_evidence_ranking.csv')
    for case in cases:
        fold=case['fold'];cid=case['candidate_id'];seed=case['seed'];run=ROOT/'analyses'/fold
        cfg=json.loads((run/'analysis_config.json').read_text());base=Path(cfg['base'])
        target=pd.read_csv(run/'input/cell_order_reference.csv').set_index('barcode')
        membership=pd.read_csv(run/'tables/candidate_membership.csv.gz')
        mem=membership[membership.candidate_id.eq(cid)&membership.seed.eq(seed)].copy()
        assert mem.barcode.is_unique and set(mem.group)=={'A','B'}
        sub=mem.reference_subclass.iloc[0];out=ROOT/'case_studies'/case['id'];out.mkdir(parents=True,exist_ok=True)
        case.update(reference_subclass=sub,n_A=int(mem.group.eq('A').sum()),n_B=int(mem.group.eq('B').sum()),
            target_cells=len(target),subclass_cells=int(target.reference_label.eq(sub).sum()),
            selection='posthoc representative fold/seed; all other results preserved; same frozen barcodes in every method')
        (out/'case.json').write_text(json.dumps(case,indent=2,ensure_ascii=False),encoding='utf-8')
        mem.to_csv(out/'fixed_membership.csv',index=False)
        z=[]
        for s in SEEDS:
            independent=membership[membership.candidate_id.eq(cid)&membership.seed.eq(s)].set_index('barcode').group
            for method in METHODS:
                d=pd.read_csv(base/'figures'/method/f'seed_{s}/plot_data.csv').set_index('barcode')
                assert set(d.index)==set(target.index)
                d=d.loc[target.index,['UMAP1','UMAP2']].join(target)
                d['fixed_group']=mem.set_index('barcode').group.reindex(d.index).fillna('Other')
                d['independent_split_group']=independent.reindex(d.index).fillna('Other')
                d['independent_split_available']=len(independent)>0
                d['method']=method;d['seed']=s;z.append(d.reset_index())
        pd.concat(z,ignore_index=True).to_csv(out/'umap_all_seeds.csv.gz',index=False,compression='gzip')
        deg=ROOT/'deg_allfold_global'/f'{cid}__{seed}__A_vs_B.csv.gz'
        d=pd.read_csv(deg);d.to_csv(out/'all_gene_DEG.csv.gz',index=False,compression='gzip')
        panel=set(np.load(base/'artifacts/gene_ids.npy').astype(str));d['gene_in_model']=d.gene_id.isin(panel)
        keep=(d[Q]<.05)&(d.log2FC_meanCP10K.abs()>.5)&((d.pct_detected_A>=.1)|(d.pct_detected_B>=.1))
        markers=[]
        for sign in [1,-1]:
            dd=d[keep&(d.log2FC_meanCP10K*sign>0)].copy();dd['effect']=dd.log2FC_meanCP10K.abs()
            markers.extend(dd.sort_values(['effect',Q],ascending=[False,True]).head(4).gene_symbol)
        # Same predeclared biology panel for all T-cell cases; data-driven strongest genes remain separate.
        canonical=[g for g in CANONICAL if g in set(d.gene_symbol)]
        if sub.startswith('CD8'):canonical=canonical[:12]
        else:canonical=canonical[12:]
        genes=list(dict.fromkeys(canonical+markers));dot=[]
        for gene in genes:
            g=d[d.gene_symbol.eq(gene)].sort_values(Q).iloc[0]
            for group in ['A','B']:
                dot.append(dict(gene=gene,group=group,mean_log1pCP10K=g['mean_log1pCP10K_'+group],
                    fraction_detected=g['pct_detected_'+group],log2FC=g.log2FC_meanCP10K,global_FDR=g[Q],
                    gene_in_model=bool(g.gene_in_model)))
        pd.DataFrame(dot).to_csv(out/'marker_dotplot.csv',index=False)
        # Missing metadata stays missing. Physical QC and biological modules are distinguished in plots.
        qq=pd.read_csv(run/'tables/cell_qc_and_module_scores.csv').set_index('barcode').loc[mem.barcode]
        qq=qq.join(mem.set_index('barcode')[['group']]);qq.to_csv(out/'cell_qc_modules.csv')
        qt=pd.read_csv(run/'tables/confounder_and_module_tests.csv')
        qt[qt.candidate_id.eq(cid)&qt.seed.eq(seed)].to_csv(out/'QC_module_statistics.csv',index=False)
        stab=pd.read_csv(run/'tables/cross_seed_candidate_stability.csv')
        stab[stab.candidate_id.eq(cid)].to_csv(out/'cross_seed_stability.csv',index=False)
        ev=ranking[ranking.candidate_id.eq(cid)]
        ev.to_csv(out/'all_seed_evidence.csv',index=False)
        recovery=pd.read_csv(run/'tables/fixed_candidate_recovery_all_methods.csv')
        recovery[recovery.candidate_id.eq(cid)&recovery.seed.eq(seed)].to_csv(out/'same_members_embedding_recovery.csv',index=False)
        atac=pd.read_csv(ROOT/'atac_evidence_summary.csv')
        ar=atac[atac.candidate_id.eq(cid)&atac.seed.eq(seed)]
        if len(ar):
            rr=ar.iloc[0]
            peaks=pd.read_csv(rr.path);peaks.to_csv(out/'all_peaks_DA.csv.gz',index=False,compression='gzip')
            ar.to_csv(out/'ATAC_case_summary.csv',index=False)
            for field,name in [('gene_peak_links_file','exact_source_template_links.csv.gz'),
                               ('overlap_gene_peak_links_file','exploratory_source_template_overlap.csv.gz')]:
                if field in ar.columns and pd.notna(rr[field]):
                    pd.read_csv(rr[field]).to_csv(out/name,index=False,compression='gzip')
        for name,t in terms.items():
            selected=t[t.candidate_id.eq(cid)&t.seed.eq(seed)]
            selected.to_csv(out/(name+'.csv'),index=False)
        # Save exact running enrichment curves for one GO and one Reactome program in EACH ranking universe.
        for scope in ['full','expressed']:
            t=terms['GSEA_'+scope];t=t[t.candidate_id.eq(cid)&t.seed.eq(seed)]
            chosen=[]
            for db,g in t.groupby('database'):
                immune=g[g.Term.str.contains('cytotox|lymph|immun|t cell|hematopo|myelo|stem cell|eryth',case=False,regex=True)]
                use=immune if len(immune) else g
                chosen.append(use.sort_values([Q,'nominal_p_floored']).iloc[0])
            rankdir=run/('fgsea_all_seeds' if scope=='full' else 'fgsea_all_seeds_expressed')
            ranks=pd.read_csv(rankdir/f'{cid}__{seed}__A_vs_B__rank.csv')
            curves=[];spec=[]
            for row in chosen:
                fn='GO_Biological_Process_2025.gmt' if row.database=='GO_BP_2025' else 'Reactome_2022.gmt'
                line=next(x for x in (run/'input'/fn).read_text().splitlines() if x.split('\t')[0]==row.Term)
                gs=set(line.split('\t')[2:]);hit=ranks.gene_symbol.isin(gs).to_numpy()
                weights=abs(ranks.wilcoxon_score_signed.to_numpy());den=weights[hit].sum()
                if den<=0:continue
                steps=np.where(hit,weights/max(den,1e-300),-1/max(1,(~hit).sum()))
                running=np.cumsum(steps);observed=running[np.argmax(np.abs(running))]
                assert abs(observed-row.ES)<1e-7,(case['id'],scope,row.Term,observed,row.ES)
                for i,(gene,score,h,es) in enumerate(zip(ranks.gene_symbol,ranks.wilcoxon_score_signed,hit,running)):
                    curves.append(dict(database=row.database,Term=row.Term,rank=i+1,gene=gene,score=score,hit=bool(h),running_ES=es))
                spec.append(dict(database=row.database,Term=row.Term,NES=row.NES,ES=row.ES,global_FDR=row[Q],
                    matched_genes=int(row.matched_genes),leading_edge_genes=row.leading_edge_genes,
                    term_selection='lowest global FDR among immune/hematopoietic terms, separately by database; posthoc illustration'))
            pd.DataFrame(curves).to_csv(out/f'GSEA_{scope}_curves.csv.gz',index=False,compression='gzip')
            pd.DataFrame(spec).to_csv(out/f'GSEA_{scope}_curves_metadata.csv',index=False)
        print('CASE_DATA_EXPORTED',case['id'],len(mem),flush=True)

if __name__=='__main__':main()
