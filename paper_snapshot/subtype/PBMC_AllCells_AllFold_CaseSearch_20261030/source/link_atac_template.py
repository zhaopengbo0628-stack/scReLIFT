"""Exact-coordinate overlap with frozen Source-only B_D; no Target graph fitting."""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1]
Q='FDR_BH_all_B2_candidate_seed_peak_tests'
GQ='FDR_BH_all_folds_all_seeds'

def interval_matches(source_peaks,target_names):
    # All overlaps with >=50% of BOTH intervals, without pretending coordinates are identical.
    by={}
    for i,name in enumerate(target_names):
        chrom,coords=name.split(':',1);st,en=map(int,coords.split('-',1));by.setdefault(chrom,[]).append((st,en,i))
    for chrom,rows in by.items():
        a=np.asarray(sorted(rows));by[chrom]=(a,a[:,0],int((a[:,1]-a[:,0]).max()))
    rows=[]
    for name in set(source_peaks):
        chrom,coords=name.split(':',1);st,en=map(int,coords.split('-',1))
        if chrom not in by:continue
        a,starts,maxwidth=by[chrom]
        lo=np.searchsorted(starts,st-maxwidth,side='left');hi=np.searchsorted(starts,en,side='left')
        for ts,te,idx in a[lo:hi]:
            overlap=max(0,min(en,te)-max(st,ts))
            if overlap/(en-st)>=.5 and overlap/(te-ts)>=.5:
                rows.append(dict(canonical_peak=name,overlapping_target_peak=target_names[idx],
                    overlap_bp=int(overlap),source_interval_fraction=overlap/(en-st),target_interval_fraction=overlap/(te-ts)))
    return pd.DataFrame(rows)

def main():
    while not (ROOT/'audit/ALL_B2_ATAC_COMPLETE.json').exists():time.sleep(30)
    while not (ROOT/'audit/ALL_FOLD_GENE_BH_COMPLETE.json').exists():time.sleep(30)
    summary=pd.read_csv(ROOT/'atac_evidence_summary.csv');registry=json.loads((ROOT/'input/datasets.json').read_text())
    out=ROOT/'atac_posthoc/source_template_links';out.mkdir(exist_ok=True)
    for fold,rec in registry.items():
        base=ROOT.parent/'PBMC_D1_AllCells_Dmode_20261030' if fold=='D1' else ROOT/'folds'/fold
        genes=np.load(base/'artifacts/gene_ids.npy').astype(str)
        source_peaks=np.load(base/'artifacts/candidate_graph.npz')['peak_names'].astype(str)
        graph=sparse.load_npz(base/'artifacts/B_D.npz').tocoo();assert graph.shape==(len(source_peaks),len(genes))
        names=pd.read_csv(rec['gene_table'],sep='\t').iloc[:,:2];names.columns=['gene_id','gene_symbol']
        symbols=names.set_index('gene_id').gene_symbol
        for ix,r in summary[summary.fold.eq(fold)].iterrows():
            d=pd.read_csv(r.path).set_index('peak')
            gene=pd.read_csv(ROOT/'deg_allfold_global'/f'{r.candidate_id}__{r.seed}__A_vs_B.csv.gz').set_index('gene_id')
            link=pd.DataFrame({'canonical_peak':source_peaks[graph.row],'gene_id':genes[graph.col],'B_D_weight':graph.data})
            link['gene_symbol']=link.gene_id.map(symbols)
            link['exact_peak_in_target']=link.canonical_peak.isin(d.index)
            # Preserve unmatched edges explicitly rather than fabricating Target correspondence.
            for col in ['delta_accessibility',Q,'pct_accessible_A','pct_accessible_B']:
                link[col]=link.canonical_peak.map(d[col])
            link['gene_log2FC']=link.gene_id.map(gene.log2FC_meanCP10K)
            link['gene_global_FDR']=link.gene_id.map(gene[GQ])
            link['strict_gene']=link.gene_global_FDR.lt(.05)&link.gene_log2FC.abs().gt(.25)
            link['strict_peak']=link[Q].lt(.05)&link.delta_accessibility.abs().ge(.05)
            link['gene_and_peak_associated']=link.strict_gene&link.strict_peak
            link['gene_peak_direction_agrees']=np.sign(link.gene_log2FC)==np.sign(link.delta_accessibility*link.B_D_weight)
            link['scope']='exact coordinates only; source-selected edge; posthoc Target gene/peak association; not causal'
            p=out/f'{r.candidate_id}__{r.seed}.csv.gz';link.to_csv(p,index=False,compression='gzip')
            summary.loc[ix,'source_template_edges']=len(link)
            summary.loc[ix,'exact_target_template_edges']=int(link.exact_peak_in_target.sum())
            summary.loc[ix,'strict_gene_and_peak_template_edges']=int(link.gene_and_peak_associated.sum())
            summary.loc[ix,'gene_peak_links_file']=str(p)
            matches=interval_matches(link.canonical_peak.to_numpy(),d.index.to_numpy())
            mapped=link[['canonical_peak','gene_id','gene_symbol','B_D_weight','gene_log2FC','gene_global_FDR','strict_gene']].merge(matches,on='canonical_peak',how='left')
            for col in ['delta_accessibility',Q,'pct_accessible_A','pct_accessible_B']:
                mapped[col]=mapped.overlapping_target_peak.map(d[col])
            mapped['strict_peak']=mapped[Q].lt(.05)&mapped.delta_accessibility.abs().ge(.05)
            mapped['gene_and_peak_associated']=mapped.strict_gene&mapped.strict_peak
            mapped['scope']='EXPLORATORY interval overlap >=50% reciprocal; not identical peaks, not proven same regulatory element'
            pp=out/f'{r.candidate_id}__{r.seed}__reciprocal_overlap.csv.gz'
            mapped.to_csv(pp,index=False,compression='gzip')
            summary.loc[ix,'reciprocal_overlap_target_links']=int(mapped.overlapping_target_peak.notna().sum())
            summary.loc[ix,'overlap_strict_gene_and_peak_links']=int(mapped.gene_and_peak_associated.sum())
            summary.loc[ix,'overlap_gene_peak_links_file']=str(pp)
            summary.loc[ix,'motif_enrichment_status']='UNAVAILABLE: no verified motif occurrence/background matrix'
    summary.to_csv(ROOT/'atac_evidence_summary.csv',index=False)
    (ROOT/'audit/ATAC_TEMPLATE_LINKS_COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',comparisons=len(summary),
        matching='exact coordinates primary; separate exploratory >=50% reciprocal interval overlaps; unmatched remain missing',
        graph='frozen Source-only B_D from each fold; no Target-based selection',causal_validation=False),indent=2))
    print('ATAC_TEMPLATE_LINKS_COMPLETE',flush=True)

if __name__=='__main__':main()
