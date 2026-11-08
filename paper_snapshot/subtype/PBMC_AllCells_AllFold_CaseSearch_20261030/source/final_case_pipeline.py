"""Persistently finish audited tables, case exports, figures and Chinese report."""
import os,sys,subprocess,time,json,fcntl
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PY='/home/u202522802045/envs/graph_attention/bin/python'

def call(name,*args):
    print('FINAL_STAGE_START',name,*args,flush=True)
    subprocess.run([PY,'-u',str(ROOT/'source'/name),*args],cwd=ROOT,check=True,env=os.environ.copy())
    print('FINAL_STAGE_DONE',name,*args,flush=True)

def main():
    lock=(ROOT/'final_cases.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (ROOT/'audit/FUNCTIONAL_BH_REPAIR_COMPLETE.json').exists():time.sleep(15)
    call('finalize_evidence.py');call('build_case_data.py')
    for case in json.loads((ROOT/'SELECTED_CASES.json').read_text()):
        for script in ['plot_case_umap.py','plot_case_markers.py','plot_case_pathways.py','plot_case_qc_stability.py','plot_case_atac.py']:
            call(script,case['id'])
    call('plot_candidate_evidence_overview.py');call('write_final_report.py');call('write_status.py')
    (ROOT/'audit/FINAL_CASE_ARTIFACTS_COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',cases=3,
        chinese_report=True,formats=['600dpi PNG','vector PDF','plot CSV','reproducible code'],
        visual_review='PENDING: must inspect rendered figures before final delivery'),indent=2))
    print('FINAL_CASE_ARTIFACTS_COMPLETE',flush=True)

if __name__=='__main__':main()
