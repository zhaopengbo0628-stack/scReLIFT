"""Configure all-fold/all-seed discovery and molecular analysis; never edit prior results."""
import json,shutil,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'PBMC_D1_AllCells_Subtype_DEG_20261030'
OLDMODEL=ROOT.parent/'PBMC_D1_AllCells_Dmode_20261030'
COUNTS={'D1':9347,'D2':6317,'D3':8105,'D4':2413}


def output(path,text):
    compile(text,str(path),'exec') if path.suffix=='.py' else None
    path.write_text(text)


def main():
    for target,n in COUNTS.items():
        run=ROOT/'analyses'/target;base=OLDMODEL if target=='D1' else ROOT/'folds'/target
        for sub in ['source','audit','logs','input','tables','partitions','deg','figures']:(run/sub).mkdir(parents=True,exist_ok=True)
        (run/'analysis_config.json').write_text(json.dumps(dict(target=target,cells=n,base=str(base),registry=str(ROOT/'input/datasets.json')),indent=2))
        for name in ['R_library','vendor']:
            p=run/name
            if not p.exists():p.symlink_to(OLD/name,target_is_directory=True)
        for fn in ['GO_Biological_Process_2025.gmt','Reactome_2022.gmt']:
            dest=run/'input'/fn
            if not dest.exists():dest.symlink_to(OLD/'input'/fn)
        for fn in ['frozen_task99_analysis.py','enrichment.py']:
            shutil.copy2(OLD/'source'/fn,run/'source'/fn)
        s=(OLD/'source/analysis_common.py').read_text()
        s=s.replace("BASE=RUN.parent/'PBMC_D1_AllCells_Dmode_20261030'",'CONFIG=json.loads((RUN/"analysis_config.json").read_text())\nBASE=Path(CONFIG["base"])\nTARGET=CONFIG["target"]\nN_TARGET=CONFIG["cells"]\nREGISTRY=json.loads(Path(CONFIG["registry"]).read_text())')
        s=s.replace("'artifacts/D1_barcodes.tsv'", "f'artifacts/{TARGET}_barcodes.tsv'")
        s=s.replace("'audit/D1_allcell_labels.csv'", "f'audit/{TARGET}_allcell_labels.csv'").replace('9347','N_TARGET')
        output(run/'source/analysis_common.py',s)
        s=(OLD/'source/inventory.py').read_text()
        s=s.replace('assert k==34','assert k==frozen.k_extra(len(set(labels)))').replace('9347','N_TARGET')
        s=s.replace("'D1'",'TARGET').replace("f'D1_ALL_", "f'{TARGET}_ALL_")
        s=s.replace("fixed_k=34",'fixed_k=k')
        output(run/'source/inventory.py',s)
        s=(OLD/'source/deg_all_candidates.py').read_text()
        start=s.index("    root=SHARED/'data/staging/PBMC/")
        end=s.index('    totals=',start)
        s=s[:start]+'''    record=REGISTRY[TARGET]
    x=sparse.load_npz(record['rna']).tocsr()
    genes=pd.read_csv(record['gene_table'],sep='\\t')
    ids=genes.iloc[:,0].astype(str).to_numpy(dtype=str)
    symbols=genes.iloc[:,1].astype(str).to_numpy(dtype=str)
    assert x.shape==(N_TARGET,36601) and np.array_equal(cells,np.asarray(Path(record['barcodes']).read_text().splitlines()))
    assert np.isfinite(x.data).all() and np.all(x.data>=0) and np.array_equal(x.data,np.round(x.data))
'''+s[end:]
        s=s.replace("sparse.load_npz(root/'atac_counts.npz')","sparse.load_npz(record['atac'])").replace('9347','N_TARGET')
        s=s.replace("    qc['ATAC_peak_read_counts']=np.asarray(atac.sum(1)).ravel()", "    if not record['atac_is_binary']: qc['ATAC_peak_read_counts']=np.asarray(atac.sum(1)).ravel()")
        s=s.replace("'D1'",'TARGET').replace("sha(root/'rna_counts.npz')","sha(Path(record['rna']))")
        output(run/'source/deg_all_candidates.py',s)
        # D1 full-library tests already exist; copy small summaries and link immutable large tables.
        if target=='D1':
            for fn in ['candidate_inventory.csv','all_candidate_deg_summary.csv']:shutil.copy2(OLD/fn,run/fn)
            for p in (OLD/'tables').iterdir():
                if p.is_file():shutil.copy2(p,run/'tables'/p.name)
            for p in (OLD/'deg').glob('*.csv.gz'):
                d=run/'deg'/p.name
                if not d.exists():d.symlink_to(p)
            for fn in ['DEG_COMPLETE.json','INVENTORY_COMPLETE.json']:shutil.copy2(OLD/'audit'/fn,run/'audit'/fn)
            p=run/'input/cell_order_reference.csv'
            if not p.exists():p.symlink_to(OLD/'input/cell_order_reference.csv')
        for expressed,outdir,rscript,flag in [(False,'fgsea_all_seeds','run_fgsea_all_seeds.R','GSEA_ALL_SEEDS_COMPLETE.json'),
                    (True,'fgsea_all_seeds_expressed','run_fgsea_all_seeds_expressed.R','GSEA_ALL_SEEDS_EXPRESSED_COMPLETE.json')]:
            s=(OLD/'source/fgsea_pipeline.py').read_text()
            s=s.replace("RUN/'fgsea'",f"RUN/'{outdir}'")
            s=s.replace("RUN/'tables/anchor_enrichment_inputs.csv'","RUN/'all_candidate_deg_summary.csv'")
            s=s.replace("f'{cid}__{contrast}__rank.csv'", "f'{cid}__{seed}__{contrast}__rank.csv'")
            s=s.replace("f'{cid}__{contrast}__results.csv'", "f'{cid}__{seed}__{contrast}__results.csv'")
            if expressed:s=s.replace('rank=rank_input(d)', 'rank=rank_input(d[d.expressed_in_comparison].copy())')
            s=s.replace("['candidate_id','contrast']","['candidate_id','seed','contrast']")
            s=s.replace("source/run_fgsea.R",'source/'+rscript)
            s=s.replace('audit/FGSEA_COMPLETE.json','audit/'+flag).replace('database_GSEA_analyses=108','database_GSEA_analyses=len(jobs)*2')
            s=s.replace('all 18 candidates x 3 contrasts x both GO/Reactome databases','all candidates x all available seeds x all 3 contrasts x GO/Reactome')
            s=s.replace('lowest available fixed seed; chosen before DEG','every recorded candidate seed; selection after complete comparison only')
            s=s.replace("FDR_BH_within_candidate_contrast_both_databases",'FDR_BH_within_candidate_seed_contrast_both_databases')
            output(run/'source'/f'{outdir}.py',s)
            s=(OLD/'source/run_fgsea.R').read_text().replace("file.path(run,'fgsea",f"file.path(run,'{outdir}")
            s=s.replace('length(scores)>30000','length(scores)>1000' if expressed else 'length(scores)>30000')
            s=s.replace('audit/FGSEA_CPU_WITNESS.txt',f'audit/{outdir}_CPU_WITNESS.txt')
            s=s.replace('lowest available fixed seed; chosen before DEG','all available seeds; no preselected best seed')
            (run/'source'/rscript).write_text(s)
    print('ALL_FOLD_ANALYSES_CONFIGURED',flush=True)


if __name__=='__main__':main()
