"""Frozen Task99 differential peaks, posthoc only; all recorded B2 candidates/seeds."""
import json,sys,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analyses/D1/source'))
from frozen_task99_analysis import rank_peaks,bh

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    out=ROOT/'atac_posthoc';out.mkdir(exist_ok=True);records=[];allp=[]
    registry=json.loads((ROOT/'input/datasets.json').read_text())
    for fold,rec in registry.items():
        run=ROOT/'analyses'/fold
        mem=pd.read_csv(run/'tables/candidate_membership.csv.gz');mem=mem[mem.method.eq('FullB2')]
        cells=Path(rec['barcodes']).read_text().splitlines();ix=pd.Index(cells)
        x=sparse.load_npz(rec['atac']).tocsr();assert x.shape[0]==len(ix)
        root=Path(rec['atac']).parent
        if rec['atac_is_binary']:
            peaks=np.load(root/'peak_names.npy').astype(str)
            assert np.isin(x.data,[0,1]).all()
        else:
            bed=pd.read_csv(root/'peaks.bed',sep='\t',header=None)
            peaks=(bed[0].astype(str)+':'+bed[1].astype(str)+'-'+bed[2].astype(str)).to_numpy()
        assert len(peaks)==x.shape[1] and len(set(peaks))==len(peaks)
        for (cid,seed),m in mem.groupby(['candidate_id','seed']):
            path=out/f'{cid}__{seed}__A_vs_B.csv.gz'
            if path.exists():d=pd.read_csv(path)
            else:
                aa=np.zeros(len(ix),bool);bb=aa.copy()
                aa[ix.get_indexer(m.loc[m.group.eq('A'),'barcode'])]=True
                bb[ix.get_indexer(m.loc[m.group.eq('B'),'barcode'])]=True
                assert aa.sum()+bb.sum()==len(m)
                d=rank_peaks(x,aa,bb,peaks,cid,fold,int(seed))
                # Frozen helper's legacy label is wrong for these inputs. Never call peak reads fragments.
                d=d.rename(columns={'mean_fragments_A':'mean_peak_value_A','mean_fragments_B':'mean_peak_value_B'})
                d['input_mode']='binary_accessibility' if rec['atac_is_binary'] else 'peak_read_counts'
                d.to_csv(path,index=False,compression='gzip')
            assert len(d)==len(peaks) and np.isfinite(d.p_value).all()
            records.append(dict(fold=fold,candidate_id=cid,seed=int(seed),reference_subclass=m.reference_subclass.iloc[0],
                n_A=int(m.group.eq('A').sum()),n_B=int(m.group.eq('B').sum()),tested_peaks=len(d),path=str(path),
                paired_ATAC=True,input_mode='binary_accessibility' if rec['atac_is_binary'] else 'peak_read_counts',
                raw_ATAC_sha256=rec['atac_sha256']))
            allp.append(d.p_value.to_numpy());print('ATAC_DONE',cid,seed,len(d),flush=True)
        del x
    globalq=bh(np.concatenate(allp));offset=0;tops=[]
    for rec in records:
        path=Path(rec['path']);d=pd.read_csv(path);n=len(d)
        q='FDR_BH_all_B2_candidate_seed_peak_tests';d[q]=globalq[offset:offset+n];offset+=n
        strict=(d[q]<.05)&d.delta_accessibility.abs().ge(.05)
        rec['global_strict_DA']=int(strict.sum());rec['global_FDR_lt005']=int((d[q]<.05).sum())
        rec['status']='STATISTICAL_ASSOCIATION' if strict.any() else 'NO_GLOBAL_SIGNIFICANT_EFFECT'
        rec['limitations']='posthoc correlated cells; raw peak depth not covariate-adjusted; no independent cohort'
        d.to_csv(path,index=False,compression='gzip');rec['result_sha256']=sha(path)
        top=d.assign(abs_delta=d.delta_accessibility.abs()).sort_values([q,'abs_delta'],ascending=[True,False]).head(30)
        tops.append(top)
    pd.DataFrame(records).to_csv(ROOT/'atac_evidence_summary.csv',index=False)
    pd.concat(tops).to_csv(out/'top_ranked_peaks_all_B2.csv',index=False)
    (ROOT/'audit/ALL_B2_ATAC_COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',comparisons=len(records),
        peak_tests=len(globalq),all_recorded_B2_candidates=True,target_ATAC_used_for_training=False,
        hypothesis_family='all tested peaks across every recorded Full B2 candidate seed and all four folds',
        statistic='frozen Task99 two-sided asymptotic Mann-Whitney; abs accessibility difference>=0.05',
        covariate_adjustment=False,motif_enrichment='UNAVAILABLE: motif background scan not provided'),indent=2))
    print('ALL_B2_ATAC_COMPLETE',len(records),flush=True)

if __name__=='__main__':main()
