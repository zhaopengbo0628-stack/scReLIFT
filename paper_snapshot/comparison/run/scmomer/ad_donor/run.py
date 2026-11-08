"""GSE214979 scMomer-S, fixed Task69 donors, external initialization explicitly recorded."""
import argparse
import csv
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import pandas as pd
from scipy import sparse
ROOT=Path('/home/u202522802045/share/zpb')
REF=ROOT/'result_contain/69_smode_ad_donor_prediction_5seed_20261014'
sys.path[:0]=[str(ROOT/'comparison/run'),str(ROOT/'comparison/run/scmomer'),str(ROOT/'comparison/evaluation')]
from pbmc_intra_common import sha,write
from ad_donor import evaluate
import s_mode_adapter as adapter
from neural_s_mode_intra import training

def status(out,value,**extra):
    import os
    write(out/'status.json',dict(status=value,**extra))
    registry=ROOT/'result_contain/task_registry.csv'
    task=str(json.loads((out/'config.json').read_text())['task_id'])
    with registry.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with registry.open() as f: reader=csv.DictReader(f); columns=reader.fieldnames; rows=list(reader)
        for row in rows:
            if row['task_id']==task: row['status']=value
        temporary=registry.with_suffix('.tmp')
        with temporary.open('w',newline='') as f:
            writer=csv.DictWriter(f,columns);writer.writeheader();writer.writerows(rows);f.flush();os.fsync(f.fileno())
        temporary.replace(registry)

def read_rna(raw,rows,features,mapping):
    spec=importlib.util.spec_from_file_location('task69_counts',REF/'source/ad.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    x=mod.read_counts(raw,rows,features)
    keep=mapping>=0
    coo=x[:,keep].tocoo()
    x=sparse.coo_matrix((coo.data,(coo.row,mapping[keep][coo.col])),shape=(len(rows),16906)).tocsr()
    sums=np.asarray(x.sum(1)).ravel()
    if np.any(sums<=0): raise ValueError('Zero reference-space RNA library')
    x=sparse.diags(10000/sums).dot(x).tocsr();x.data=np.log1p(x.data)
    return x

def prepare(out):
    import h5py
    cfg=json.loads((REF/'source/config.json').read_text()); bench=Path(cfg['benchmark'])
    raw=bench/'GSE214979_filtered_feature_bc_matrix.h5'
    with h5py.File(raw) as h:
        bars=h['matrix/barcodes'].asstr()[:]
    cells=pd.read_csv(REF/'cells.csv',dtype={'donor':str,'barcode':str})
    np.testing.assert_array_equal(bars,cells.barcode.to_numpy())
    rna=np.load(bench/'rna_feature_index.npy');atac=np.load(bench/'atac_feature_index.npy')
    with h5py.File(ROOT/'scMomer_official/panglao_10000.h5ad') as h:
        genes=h['var'][h['var'].attrs['_index']].asstr()[:]
    names=pd.read_csv(bench/'rna_features.tsv',sep='	').gene_name.to_numpy()
    if len(set(genes))!=16906: raise ValueError('Ambiguous reference gene symbols')
    ambiguous=pd.Index(names).duplicated(keep=False)
    positions={g:i for i,g in enumerate(genes)}
    mapping=np.array([positions.get(g,-1) if not bad else -1 for g,bad in zip(names,ambiguous)])
    if len(np.unique(mapping[mapping>=0]))!=int((mapping>=0).sum()): raise ValueError('Ambiguous reference mapping')
    cache=out/'cache';cache.mkdir()
    pd.DataFrame(dict(raw_rna_column=np.flatnonzero(ambiguous),gene_name=names[ambiguous])).to_csv(cache/'ambiguous_symbols_excluded_from_model.csv',index=False)
    np.save(cache/'gene_ids.npy',genes)
    write(cache/'mapping.json',dict(mapped_genes=int((mapping>=0).sum()),reference_genes=16906,
        normalization='official reference alignment -> library 10000 -> log2(1+x); internal stored ln converted by adapter',
        cells_filtered=0,unmapped_reference='zero columns; no fitted query statistics',
        ambiguous_native_columns=int(ambiguous.sum()),ambiguous_symbol_policy='Exclude duplicated native symbols from model mapping; preserve raw counts; never merge distinct Ensembl IDs by symbol.'))
    ix={k:np.load(Path(cfg['split'])/f'{k}_index.npy') for k in ('train','val','test')}
    donors=pd.read_csv(REF/'donor_split.csv',dtype={'donor':str})
    for k,v in ix.items():
        expected=set(donors.loc[donors.split.eq('test' if k=='test' else 'train'),'donor'])
        if set(cells.iloc[v].donor)!=expected: raise ValueError('Donor leakage in split')
        np.save(cache/f'{k}_index.npy',v)
    spec=importlib.util.spec_from_file_location('task69_counts',REF/'source/ad.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    for k in ('train','val'):
        sparse.save_npz(cache/f'rna_log_{k}.npz',read_rna(raw,ix[k],rna,mapping))
        sparse.save_npz(cache/f'atac_{k}.npz',mod.read_counts(raw,ix[k],atac,binary=True))
    np.save(cache/'raw_to_reference.npy',mapping)
    write(cache/'manifest.json',dict(raw_sha256=sha(raw),split_sha256={k:sha(Path(cfg['split'])/f'{k}_index.npy') for k in ix},
        test_atac_read=False,current_training_donors=sorted(set(cells.iloc[ix['train']].donor))))
    return cfg,raw,rna,mapping,cells

def run(out):
    import torch
    for name,digest in json.loads((out/'config.json').read_text())['source_hashes'].items():
        if sha(name)!=digest:raise ValueError('Verified source changed: '+name)
    from threadpoolctl import threadpool_limits
    status(out,'QUEUED_AFTER_EXISTING_TASKS')
    while True:
        rows=list(csv.DictReader((ROOT/'result_contain/task_registry.csv').open()))
        active=[]
        for row in rows:
            if int(row['task_id'])>=int(json.loads((out/'config.json').read_text())['task_id']):continue
            if not row['status'].startswith(('RUNNING','QUEUED')):continue
            pid=row.get('pid','')
            if not pid and row.get('pid_path') and Path(row['pid_path']).exists():pid=Path(row['pid_path']).read_text().strip()
            if pid.isdigit() and Path('/proc',pid).exists():active.append(row)
            elif not pid:raise ValueError('Unresolved predecessor PID: '+row['task_id'])
        write(out/'queue_check.json',dict(active_predecessors=[r['task_id'] for r in active],checked_pid_presence=True))
        if not active:break
        time.sleep(30)
    with (ROOT/'.benchmark_a30.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        while __import__('subprocess').check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(30)
        status(out,'RUNNING_PREPROCESS_SMOKE_AND_5SEED')
        torch.set_num_threads(2)
        with threadpool_limits(2):
            cfg,raw,rna,mapping,cells=prepare(out);cache=out/'cache'
            data={}
            for k in ('train','val'):
                data[f'rna_log_{k}']=sparse.load_npz(cache/f'rna_log_{k}.npz')
                data[f'rna_{k}']=data[f'rna_log_{k}']
                data[f'atac_{k}']=sparse.load_npz(cache/f'atac_{k}.npz')
            hashes={str(p):sha(p) for p in (ROOT/'scMomer_official/panglao_10000.h5ad',ROOT/'scMomer_official/pretrained_scmomer.pth')}
            adapter.assets=lambda _: (np.arange(16906),hashes)
            original_features=adapter.rna_features
            frozen_features={}
            def shared_features(model,matrix,mapping,device,batch):
                if id(matrix) not in frozen_features:
                    frozen_features[id(matrix)]=original_features(model,matrix,mapping,device,batch)
                return frozen_features[id(matrix)]
            adapter.rna_features=shared_features
            config=json.loads((out/'config.json').read_text())['training_settings']
            def fit(seed,smoke):
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                unit=out/('smoke' if smoke else 'models')/f'seed_{seed}';unit.mkdir(parents=True)
                subset={k:v[:8 if k.endswith('train') else 4] for k,v in data.items()} if smoke else data
                settings=dict(config,batch_size=2) if smoke else config
                history=[];stages=[];torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                model,report=adapter.fit(subset,settings,unit,seed,smoke,'cuda',cache,
                    training(subset,settings,seed,smoke,'cuda',history,stages))
                model.eval().requires_grad_(False)
                print('FIT_COMPLETED',seed,'smoke',smoke,flush=True)
                sample=adapter.predict(model,subset['rna_log_val'][:2],{'mapping':np.arange(16906)},'cuda',1)
                if sample.shape!=(2,128) or not np.isfinite(sample).all(): raise ValueError('Invalid frozen embedding')
                write(unit/'training.json',dict(**report,history=history,stages=stages,seconds=time.perf_counter()-start,
                    peak_vram_bytes=torch.cuda.max_memory_allocated(),config=settings,seed=seed,
                    setting='external_pretrained_initialized_current_train_donor_adaptation',pretraining_overlap='UNVERIFIED',
                    test_used_for_training=False,gpu=torch.cuda.get_device_name(0),
                    total_parameters=sum(p.numel() for p in model.parameters())))
                if smoke:
                    del model;torch.cuda.empty_cache();return
                z=np.empty((len(cells),128),np.float32);start=time.perf_counter();digest=sha(unit/'best.pt')
                for donor in sorted(cells.donor.unique()):
                    rows=np.flatnonzero(cells.donor.eq(donor))
                    x=read_rna(raw,rows,rna,mapping)
                    z[rows]=adapter.predict(model,x,{'mapping':np.arange(16906)},'cuda',1)
                if sha(unit/'best.pt')!=digest or any(p.requires_grad for p in model.parameters()): raise ValueError('Model mutated')
                np.save(unit/'cell_embeddings.npy',z)
                evaluate(z,REF,unit,seed,'scMomer-S')
                write(unit/'manifest.json',dict(status='COMPLETE',checkpoint_sha256=digest,embedding_sha256=sha(unit/'cell_embeddings.npy'),
                    inference_seconds=time.perf_counter()-start,test_input='RNA only',seed=seed))
                del model;torch.cuda.empty_cache()
            fit(20020130,True)
            write(out/'smoke_verification.json',dict(status='PASS',frozen_inference=True))
            del frozen_features; frozen_features={}
            for seed in json.loads((out/'config.json').read_text())['seeds']:
                fit(seed,False)
                summarize(out)
                write(out/'progress.json',dict(completed_seeds=seed-20020129,expected_seeds=len(json.loads((out/'config.json').read_text())['seeds'])))

def summarize(out):
    rows=[];pred=[]
    selected=set(json.loads((out/'config.json').read_text())['seeds'])
    for method,root in [('Full B2',out/'full_b2_evaluation'),('scMomer-S',out/'models')]:
        for p in root.glob('seed_*/probe_metrics.csv'):
            if int(p.parent.name.split('_')[-1]) in selected:rows.append(pd.read_csv(p))
        for p in root.glob('seed_*/probe_predictions.csv'):
            if int(p.parent.name.split('_')[-1]) in selected:pred.append(pd.read_csv(p))
    if rows:
        frame=pd.concat(rows);frame.to_csv(out/'comparison_per_seed.csv',index=False)
        fields=['AUROC','Balanced_Accuracy','Macro_F1','Sensitivity','Specificity','AUPRC']
        s=frame.groupby(['method','scheme'])[fields].agg(['mean','std']);s.columns=['_'.join(c) for c in s.columns]
        s.to_csv(out/'comparison_summary.csv')
        lines=['# Exploratory fixed-donor comparison (pre_final)', '',
            'Task69 holdout/P2 already observed. No independent confirmatory claim.', '',
            '| Method | Scheme | Seeds | AUROC | Balanced Accuracy | Macro-F1 | Sensitivity | Specificity | AUPRC |',
            '|---|---|---|---|---|---|---|---|---|']
        for (method,scheme),row in s.iterrows():
            n=int(((frame.method==method)&(frame.scheme==scheme)).sum())
            values=[f"{row[f+'_mean']:.4f} +/- {row[f+'_std']:.4f}" for f in fields]
            lines.append('| '+method+' | '+scheme+' | '+str(n)+' | '+' | '.join(values)+' |')
        (out/'comparison_summary.md').write_text('\n'.join(lines)+'\n')
    if pred:pd.concat(pred).to_csv(out/'comparison_predictions.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if json.loads((args.out/'config.json').read_text()).get('user_paused'):
        status(args.out,'PAUSED_BY_USER');raise SystemExit(0)
    try:run(args.out);status(args.out,'COMPLETED_PRE_FINAL');(args.out/'exitcode').write_text('0')
    except Exception as e:status(args.out,'FAILED_HARD_STOP',error=repr(e));(args.out/'exitcode').write_text('1');raise
