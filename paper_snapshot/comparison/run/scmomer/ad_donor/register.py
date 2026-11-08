"""Reserve isolated AD comparison task and verify Task69 evaluator regression."""
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime,timezone
ROOT=Path('/home/u202522802045/share/zpb')
sys.path.insert(0,str(ROOT/'comparison/evaluation'))
from ad_donor import evaluate,sha
import numpy as np
import pandas as pd
registry=ROOT/'result_contain/task_registry.csv'
with registry.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    with registry.open() as f:reader=csv.DictReader(f);columns=reader.fieldnames;rows=list(reader)
    numbers=[int(r['task_id']) for r in rows]
    for folder in (ROOT/'result_contain',ROOT/'slum'):
        for p in folder.iterdir():
            try:numbers.append(int(p.name.split('_')[0]))
            except ValueError:pass
    task=max(numbers)+1;name=f'{task}_scmomer_ad_donor_benchmark_20261015';out=ROOT/'result_contain'/name
    out.mkdir();(out/'source').mkdir()
    files=[ROOT/'comparison/evaluation/ad_donor.py',ROOT/'comparison/run/scmomer/ad_donor/run.py',
           ROOT/'comparison/run/scmomer/s_mode_adapter.py',ROOT/'comparison/run/neural_s_mode_intra.py',ROOT/'comparison/run/pbmc_intra_common.py']
    for p in files:shutil.copy2(p,out/'source'/p.name)
    cfg=dict(task_id=task,result=str(out),source_hashes={str(p):sha(p) for p in files},
        setting='external pretrained initialized, current 11 train donors adaptation',pretraining_overlap='UNVERIFIED',
        official_repo='https://github.com/nobody927/scMomer',official_commit='e683cee73d03e605e98394bd2b4fd073d0fac91a',
        upstream_cell_manifest='UNAVAILABLE',download_asset_match_to_official_drive='UNVERIFIED',
        seeds=list(range(20020130,20020135)),run_tier='pre_final',predecessors=[r['task_id'] for r in rows if r['status'].startswith(('RUNNING','QUEUED'))])
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    row=dict.fromkeys(columns,'');row.update(task_id=str(task),status='REGISTERED_CHECKING',experiment_name=name,
        model_version='scMomer-S AD donor benchmark',result_path=str(out),run_script_path=str(out/'source/run.py'),
        stdout_path=str(ROOT/'slum'/f'{name}.out'),stderr_path=str(ROOT/'slum'/f'{name}.err'),
        pid_path=str(ROOT/'slum'/f'{name}.pid'),exitcode_path=str(out/'exitcode'),launcher='direct_nohup_queue',scheduler='none',
        datasets='GSE214979; SEAAD read-only audit',seeds='20020130-20020134',created_at=datetime.now(timezone.utc).isoformat())
    rows.append(row);tmp=registry.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:w=csv.DictWriter(f,columns);w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
    tmp.replace(registry)
ref=ROOT/'result_contain/69_smode_ad_donor_prediction_5seed_20261014'
for seed in range(20020130,20020135):
    original=ref/'models/AD'/f'seed_{seed}'
    dest=out/'full_b2_evaluation'/f'seed_{seed}'
    evaluate(np.load(original/'cell_embeddings.npy'),ref,dest,seed,'Full B2')
    old=pd.read_csv(original/'probe_predictions.csv');new=pd.read_csv(dest/'probe_predictions.csv')
    np.testing.assert_allclose(old.ad_probability,new.ad_probability,rtol=0,atol=1e-12)
    oldm=pd.read_csv(original/'probe_metrics.csv');newm=pd.read_csv(dest/'probe_metrics.csv')
    for k in ['AUROC','Balanced_Accuracy','Macro_F1','Sensitivity','Specificity','TN','FP','FN','TP','PCA','C']:
        np.testing.assert_allclose(oldm[k],newm[k],rtol=0,atol=1e-12)
(out/'regression_verification.json').write_text(json.dumps(dict(status='PASS',seeds=5,schemes=3,
    original_probability_and_metrics_unchanged=True,task69_read_only=True),indent=2))
(out/'README.md').write_text('''# GSE214979 exploratory donor Benchmark

Full B2 Task69 remains read-only. The evaluator reuses its frozen donor split, taxonomy, pooling and fold-local Probe fitting. Five-seed regression must exactly reproduce original probabilities and metrics. AUPRC is average precision with AD positive; no new model selection.

scMomer-S: external pretrained initialized, frozen RNA encoder, train-donor-only paired reconstruction and S3 distillation using the existing verified project adapter. This is NOT an exact full S1/S2/S3 from-scratch official reproduction. Upstream label-conditioned adversarial pairs are not added. Pretraining cohort overlap and asset chain of custody remain UNVERIFIED; no clean-pretraining claim. Official architecture is preserved. Reference-gene aligned library normalization is model-specific; evaluator is identical.

Fixed 11 train donors and four test donors (1230,4443,4482,HCT17HEX), Task69 indices unchanged. Full B2 64 dimensions; scMomer 128 dimensions. No test ATAC read. Probe imputation/scaling/PCA/C choice are train-donor-only. Inner LODO evaluates Probe with fixed outer-train representation.

Task69 test results/P2 selection were already observed: exploratory, not independent confirmatory validation. Five seeds are repeated model training on a single holdout, not independent cohorts. Age/region/repository/sex/composition confounding remains.

Queue after existing tasks, shared A30 lock, 2 CPU threads; smoke first, then single seed end-to-end and remaining seeds. No SEAAD training or 15-seed expansion.
''')
print(json.dumps(dict(task_id=task,result=str(out),regression='PASS')))
