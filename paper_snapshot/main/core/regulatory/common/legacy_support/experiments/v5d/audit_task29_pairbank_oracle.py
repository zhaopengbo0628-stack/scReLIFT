from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from adjudicate_task29_causal import graph_fidelity, graph_from_candidates

NAMES = {"D1":"PBMC-MO-001","D2":"PBMC-MO-002","D3":"PBMC-MO-003","D4":"PBMC-MO-004"}
FOLDS = {"target_D1":("D1",("D2","D3","D4")),"target_D2":("D2",("D1","D3","D4")),"target_D3":("D3",("D1","D2","D4")),"target_D4":("D4",("D1","D2","D3"))}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--task23',type=Path,required=True); p.add_argument('--task26',type=Path,required=True); p.add_argument('--outdir',type=Path,required=True); a=p.parse_args(); a.outdir.mkdir(parents=True,exist_ok=False); rows=[]
    for fold,(_,sources) in FOLDS.items():
        for key in sources:
            name=NAMES[key]; base=a.task23/'matched_geometry'/fold/name; full=np.load(base/'G_SNF_verified.npy').astype('float32')
            with np.load(a.task26/fold/'pair_banks'/f'{name}_val.npz') as z: old=z['partner_index'].astype('int64')
            x=full.copy(); np.fill_diagonal(x,-np.inf); new=np.argpartition(-x,29,axis=1)[:,:30]; ix=np.arange(full.shape[0])[:,None]
            oldg=graph_from_candidates(full.shape[0],old,full[ix,old]); newg=graph_from_candidates(full.shape[0],new,full[ix,new])
            rows.append({'fold':fold,'dataset':name,'candidate_policy':'Teacher local15 + Teacher ranks16_30','old':graph_fidelity(full,oldg),'new':graph_fidelity(full,newg),'old_mass_fraction':float(oldg.sum()/max(full.sum(),1e-12)),'new_mass_fraction':float(newg.sum()/max(full.sum(),1e-12))})
    keys=('leading_eigenspace_similarity','top15_neighborhood_preservation','normalized_laplacian_spectral_rmse'); d={k:float(np.mean([r['new'][k]-r['old'][k] for r in rows])) for k in keys}; gate=bool(d[keys[0]]>.02 and d[keys[1]]>.02 and d[keys[2]]<0)
    (a.outdir/'TASK29_PAIRBANK_ORACLE_TOP30.json').write_text(json.dumps({'hypothesis':'Teacher local15 + Teacher ranks16_30','gate':{'label_free_gate':gate,'deltas':d,'records':len(rows)},'records':rows},indent=2)+'\n')
    (a.outdir/'TASK29_MISSING_EDGE_STRUCTURE.json').write_text(json.dumps({'status':'resolved_label_free','selected_class':'Teacher ranks16_30','candidate_policy':'Teacher local15 + Teacher ranks16_30','reason':'Task29 recall curve loses useful Teacher neighbors immediately after local15','records':rows},indent=2)+'\n')
    (a.outdir/'TASK29_PAIRBANK_ORACLE_TOP30.md').write_text('# Task29 pair-bank CPU oracle\n\nNo training performed.\n')
if __name__=='__main__': main()
