"""Reconcile inherited stage flags against completed, numerically verified results."""
import json
import shutil
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    audit=json.loads((ROOT/'audit/FINAL_NUMERICAL_VERIFICATION.json').read_text())
    assert audit['status']=='PASS' and audit['RNA_contrasts']==537
    canonical=pd.read_csv(ROOT/'candidate_seed_evidence_matrix.csv')
    columns=['DEG_status','GO_Reactome_status']
    assert canonical[columns].eq('COMPLETE').all().all()
    backup=ROOT/'audit/stale_status_before_finalization'
    backup.mkdir(exist_ok=True)
    changed=[]
    for case in json.loads((ROOT/'SELECTED_CASES.json').read_text()):
        path=ROOT/'case_studies'/case['id']/'all_seed_evidence.csv'
        old=pd.read_csv(path)
        matching=canonical[canonical.candidate_id.eq(case['candidate_id'])]
        assert set(old.seed)==set(matching.seed)
        assert old[columns].notna().all().all()
        if not old[columns].eq('COMPLETE').all().all():
            snapshot=backup/(case['id']+'_all_seed_evidence.csv')
            if not snapshot.exists():shutil.copy2(path,snapshot)
            old[columns]='COMPLETE'
            old.to_csv(path,index=False)
            changed.append(str(path.relative_to(ROOT)))
    path=ROOT/'ANALYSIS_STAGE_STATUS.csv'
    snapshot=backup/path.name
    if not snapshot.exists():shutil.copy2(path,snapshot)
    pd.DataFrame([dict(fold=fold,inventory=True,DEG=True,RNA_UMAP=True,
        GSEA_expressed=True,GSEA_full=True,complete=True) for fold in ['D1','D2','D3','D4']]).to_csv(path,index=False)
    changed.append(path.name)
    (ROOT/'audit/FINAL_STATUS_METADATA_RECONCILIATION.json').write_text(json.dumps(dict(
        status='PASS',verification='FINAL_NUMERICAL_VERIFICATION.json',
        primary_seed_table_rows=len(canonical),updated=changed,
        numerical_results_and_membership_changed=False,
        historical_status_snapshots_preserved=str(backup.relative_to(ROOT))),indent=2))
    print('FINAL_STATUS_METADATA_PASS',changed,flush=True)


if __name__=='__main__':main()
