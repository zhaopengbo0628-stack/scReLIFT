"""Small audited all-fold counts while the complete large tables are being written."""
import time,sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analyses/D1/source'))
from analysis_common import bh

def main():
    for fold in ['D1','D2','D3','D4']:
        while not (ROOT/'analyses'/fold/'audit/ALL_SEED_ANALYSIS_COMPLETE.json').exists():time.sleep(15)
    rows=[]
    for scope,fn,pcol in [('full','fgsea_all_seeds/all_candidate_gsea_summary.csv.gz','nominal_p_floored'),
                         ('expressed','fgsea_all_seeds_expressed/all_candidate_gsea_summary.csv.gz','nominal_p_floored'),
                         ('ORA','fgsea_all_seeds/all_candidate_ora_summary.csv.gz','nominal_p')]:
        tables=[]
        cols=['candidate_id','method','reference_subclass','seed','contrast','database','Term',pcol]
        for f in ['D1','D2','D3','D4']:
            tables.append(pd.read_csv(ROOT/'analyses'/f/fn,usecols=cols).assign(fold=f))
        d=pd.concat(tables,ignore_index=True);raw=d[pcol].to_numpy();p=np.where(np.isfinite(raw),raw,1.)
        d['global_FDR']=bh(p)
        d['immune']=d.Term.str.contains('cytotox|lymph|t cell|b cell|interferon|antigen|chemokin|leukocyte|myeloid|immun|monocyt|natural killer|inflamm',case=False,regex=True)
        for keys,g in d[d.contrast.eq('A_vs_B')].groupby(['fold','candidate_id','method','reference_subclass','seed','database']):
            sig=g.global_FDR.lt(.05)
            rows.append(dict(zip(['fold','candidate_id','method','reference_subclass','seed','database'],keys),scope=scope,
                tested_terms=len(g),significant_terms=int(sig.sum()),immune_terms=int((sig&g.immune).sum()),min_FDR=float(g.global_FDR.min())))
        pd.DataFrame(rows).to_csv(ROOT/'audit/corrected_pathway_counts_preview.csv',index=False)
        print('PATHWAY_COUNTS_PREVIEW',scope,flush=True)

if __name__=='__main__':main()
