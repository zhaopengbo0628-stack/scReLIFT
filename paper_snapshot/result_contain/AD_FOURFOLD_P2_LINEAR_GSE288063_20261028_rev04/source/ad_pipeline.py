"""Durable download-gated four-fold benchmark. No decisions depend on TARGET metrics."""
from pathlib import Path
import argparse,fcntl,hashlib,importlib.metadata as md,json,os,platform,subprocess,sys,time,traceback
import ad_runtime as a
from ad_new_cohort import BUNDLE,NAME

SOURCE=Path(__file__).parent
ENV=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')

def status(stage,**kw):
    state=dict(stage=stage,pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    a.write(a.OUT/'queue/pipeline_status.json',state)
    with (a.OUT/'queue/events.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
    print(json.dumps(state),flush=True)

def run(script,*args,gpu=False,log=None):
    env=dict(ENV,CUDA_VISIBLE_DEVICES='0' if gpu else '')
    cmd=[sys.executable,str(SOURCE/script),*map(str,args)]
    def invoke():
        with (log or (a.OUT/'queue'/f'{Path(script).stem}.log')).open('a') as out:
            subprocess.run(cmd,env=env,stdout=out,stderr=subprocess.STDOUT,check=True,cwd=SOURCE)
    if not gpu:return invoke()
    # Coordinate our GPU jobs and wait for unrelated GPU users rather than preempting.
    with (a.ROOT/'.b2_gpu_training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        while True:
            memory=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            if memory<500:break
            status('WAIT_FREE_GPU',memory_used_MiB=memory,next_command=cmd);time.sleep(30)
        invoke()

def freeze():
    a.protocol()
    versions=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
        packages={n:md.version(n) for n in ['numpy','scipy','pandas','scikit-learn','torch','h5py','anndata','scanpy','joblib']},
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip())
    if (a.OUT/'software_versions.json').exists():assert a.read(a.OUT/'software_versions.json')==versions
    else:a.write(a.OUT/'software_versions.json',versions)
    names=['ad_runtime.py','ad_train.py','ad_rna.py','ad_new_cohort.py','ad_evaluate.py','ad_biology.py',
        'ad_prepare.py','ad_verify_bundle.py','ad_pipeline.py','ad_final_check.py','ad_finish.py',
        'ad_comparison_report.py','ad_export.py','test_ad_replacement.py']
    dependencies=[a.REFERENCE,a.ROOT/'result_contain/AD_Dmode_4Cohort_LODO_20261026_rev02/queue/ad_fourfold_prepare.py',
        a.ROOT/'result_contain/AD_Dmode_4Cohort_20261024/queue/source_balanced_selector.py',
        a.ROOT/'result_contain/AD_Dmode_4Cohort_20261024/source_snapshot/harmonize_frozen.py']
    dependencies += list((a.ROOT/'main/core/regulatory/common/legacy_support/v5d').glob('*.py'))
    record=dict(protocol_sha256=a.EXPECTED_PROTOCOL,source_hashes={str(SOURCE/n):a.sha(SOURCE/n) for n in names},
        dependencies={str(p):a.sha(p) for p in dependencies},scientific_methods_unchanged=True,
        adaptation='replace excluded cohort; three Probe SOURCE cohorts; author library-qualified count reader; fail-closed H5 gate')
    p=a.OUT/'IMPLEMENTATION_FREEZE.json'
    if p.exists():assert a.read(p)==record,'Implementation changed; record repair before resuming'
    else:a.write(p,record)
    a.write(a.OUT/'FREEZE_COMPLETE.json',dict(protocol_sha256=a.EXPECTED_PROTOCOL,versions_sha256=a.sha(a.OUT/'software_versions.json'),
        implementation_sha256=a.sha(p),target_metrics_read=False))

def verify_dependencies():
    rec=a.read(a.OUT/'IMPLEMENTATION_FREEZE.json')
    for p,d in {**rec['dependencies'],**rec['source_hashes']}.items():assert a.sha(p)==d,p

def main():
    (a.OUT/'queue').mkdir(parents=True,exist_ok=True);(a.OUT/'reports').mkdir(exist_ok=True)
    with (a.OUT/'queue/pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        freeze();run('test_ad_replacement.py')
        while True:
            manifest=a.read(BUNDLE/'download_manifest.json')
            if manifest.get('status')=='COMPLETE':break
            processes=subprocess.check_output(['ps','-eo','args'],text=True)
            if not any('python' in x and 'download_gse288063_min.py' in x for x in processes.splitlines()):
                raise RuntimeError('Downloader exited before COMPLETE; inspect download.log and resume axel states')
            status('WAIT_DOWNLOAD',download_status=manifest.get('status'));time.sleep(60)
        verify_dependencies();status('VERIFY_H5_MULTIOME_AND_LIBRARY_BARCODES');run('ad_verify_bundle.py')
        assert a.read(BUNDLE/'verification_report_rev04.json')['status']=='PASS'
        status('PREPROCESSING_FOUR_FOLDS');a.write(a.OUT/'queue/status.json',dict(stage='PREPROCESSING'))
        if not (a.PREP/'CPU_COMPLETE.json').exists():run('ad_prepare.py')
        assert a.read(a.PREP/'CPU_COMPLETE.json')['status']=='COMPLETE'
        verify_dependencies()
        run('ad_train.py','--fold','fold4','--seed',a.SEEDS[0],'--smoke',gpu=True)
        assert a.read(a.OUT/'GPU_SMOKE.json')['status']=='PASS'
        for fold,cfg in a.protocol()['folds'].items():
            assert len(cfg['sources'])==3 and NAME in cfg['sources']+[cfg['target']]
            for seed in a.SEEDS:
                verify_dependencies()
                completed=a.OUT/'models'/fold/f'seed_{seed}'/'TRAINING_COMPLETE.json'
                if not completed.exists():
                    status('TRAINING',fold=fold,seed=seed)
                    a.write(a.OUT/'queue/status.json',dict(stage='TRAINING',fold=fold,seed=seed))
                    run('ad_train.py','--fold',fold,'--seed',seed,gpu=True,log=a.OUT/'queue'/f'{fold}_{seed}.log')
                assert a.read(completed)['status']=='COMPLETE'
                for method in ['Full_B2','RNA_only_Encoder','PCA_RNA64']:
                    status('EVALUATION',fold=fold,seed=seed,method=method)
                    a.write(a.OUT/'queue/analysis_status.json',dict(stage='EVALUATION',fold=fold,seed=seed,method=method))
                    run('ad_evaluate.py','--fold',fold,'--seed',seed,'--method',method,gpu=method!='PCA_RNA64')
                    run('ad_biology.py','--fold',fold,'--seed',seed,'--method',method)
                run('ad_comparison_report.py');run('ad_export.py')
        a.write(a.OUT/'queue/status.json',dict(stage='ALL_TRAINING_COMPLETE'))
        a.write(a.OUT/'queue/analysis_status.json',dict(stage='ALL_ANALYSES_COMPLETE',runs=36))
        status('FINAL_DETERMINISTIC_VERIFICATION');run('ad_finish.py');verify_dependencies()
        assert a.read(a.OUT/'reports/FINAL_COMPLETE.json')['status']=='COMPLETE'
        status('COMPLETE',runs=36,folds=4,seeds=a.SEEDS)

if __name__=='__main__':
    try:main()
    except Exception:
        status('FAILED',traceback=traceback.format_exc())
        raise
