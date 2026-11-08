"""Repair finite-P accounting after the original collector completes naturally."""
import sys,time,json,fcntl,shutil
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from collect_and_rank import pool_pathways,rank_candidates

def main():
    lock=(ROOT/'functional_reconcile.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (ROOT/'audit/CASE_SEARCH_READY_FOR_BIOLOGICAL_REVIEW.json').exists():time.sleep(15)
    backup=ROOT/'audit/pre_nonfinite_pathway_BH_repair';backup.mkdir(exist_ok=True)
    files=list(ROOT.glob('all_fold_all_seed_GSEA*'))+list(ROOT.glob('all_fold_all_seed_ORA*'))
    files += [ROOT/x for x in ['candidate_seed_evidence_ranking.csv','FullB2_candidate_seed_ranking.csv','CASE_SHORTLIST.csv']]
    files += [ROOT/'audit/CASE_SEARCH_READY_FOR_BIOLOGICAL_REVIEW.json']
    for p in files:
        if p.is_file() and not (backup/p.name).exists():shutil.move(str(p),str(backup/p.name))
    gs=pool_pathways('fgsea_all_seeds/all_candidate_gsea_summary.csv.gz','all_fold_all_seed_GSEA_full.csv.gz')
    ex=pool_pathways('fgsea_all_seeds_expressed/all_candidate_gsea_summary.csv.gz','all_fold_all_seed_GSEA_expressed.csv.gz')
    ora=pool_pathways('fgsea_all_seeds/all_candidate_ora_summary.csv.gz','all_fold_all_seed_ORA.csv.gz')
    deg=pd.read_csv(ROOT/'all_fold_all_seed_DEG_summary.csv');rank_candidates(deg,gs,ex,ora)
    (ROOT/'audit/FUNCTIONAL_BH_REPAIR_COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',
        original_collector_outputs_preserved=str(backup),gene_tests_or_candidate_members_changed=False,
        issue='fgsea returned unestimable P/NES for two D2 expressed-universe pathway tests; old BH propagated NA',
        policy='raw NA remains; each unestimable P enters global BH as P=1; full family retained',
        model_retraining=False),indent=2))
    print('FUNCTIONAL_BH_REPAIR_COMPLETE',flush=True)

if __name__=='__main__':main()
