"""Single-source adapter for frozen Task35 Full B2; 20 models, 80 evaluations."""
import argparse
import copy
import csv
import fcntl
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

ROOT = Path('/home/u202522802045/share/zpb')
sys.path[:0] = [str(ROOT), str(ROOT/'main/run'), str(ROOT/'main/run/d_mode'), str(ROOT/'main/core/regulatory/common/legacy_support')]
from data.code.s_mode.prepare import legacy, prepare, reader, splits
from main.core.regulatory.s_mode.single_source import finite, map_rna

sys.path.insert(0,str(ROOT/'comparison/run'))
import efficiency_v2 as efficiency

REFERENCE = ROOT/'main/core/run_b2_strict_lodo.py'
REFERENCE_SHA = '3de1ad3642e7af276658d9b239ccaf9457adf7f8e6078b95a237cf4605f2c7e1'
spec = importlib.util.spec_from_file_location('smode_frozen_b2', REFERENCE)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
if b.sha(REFERENCE) != REFERENCE_SHA:
    raise ValueError('Frozen Full B2 changed')
BASE_CONFIG = copy.deepcopy(b.CONFIG)
SEEDS = list(range(20020130,20020135))


def source_data(cache, key):
    state = np.load(cache/'rna_state.npz')
    mean, std = state['mean'], state['std']
    source = dict(name=key, key=key)
    for split in ('train','val'):
        raw = np.load(cache/f'rna_{split}.npy')
        field = np.load(cache/f'F_{split}.npy')
        if raw.shape != field.shape:
            raise ValueError('RNA/F dimensions differ')
        source[split] = dict(rna=((raw-mean)/(std+1e-6)).astype(np.float32),
                              raw_nonzero=raw!=0, field=field)
        for name,value in source[split].items(): finite(value,name)
    source['raw_zero_standardized'] = -mean/(std+1e-6)
    return source, mean, std


def warnings(teacher, source, device):
    rows=[]
    for split in ('train','val'):
        outputs=b.forward_numpy(teacher,source[split]['rna'],device,source[split]['field'])
        for name,value in zip(['z_R','z_F','z_T','RNA_reconstruction','F_reconstruction'],outputs):
            finite(value,name)
            if name.startswith('z_'):
                info=b.rank_info(value)
                rows.append(dict(split=split,representation=name,rank=info,
                                 warning=info['effective_rank']<BASE_CONFIG['teacher_gate']['min_rank']))
        source[split]['teacher_latent']=outputs[2]
    return dict(policy='WARNING_ONLY; never skip seed',latent_diagnostics=rows,
                F_retention_mse=float(np.mean((outputs[4]-source['val']['field'])**2)))


def evaluate(unit, source_key, target_key, student, cache, mean, std, seed, device):
    evaluation=unit/'evaluation'/f'{source_key}_to_{target_key}'
    if (evaluation/'manifest.json').exists():
        return legacy.checked(evaluation/'manifest.json')
    index,cells=splits(target_key)
    counts,ids,raw_path,provenance=reader.raw_rna(target_key)  # RNA-only loader
    genes=np.load(cache/'gene_ids.npy')
    x,hit=map_rna(counts[index['test']],ids,genes,mean,std)
    evaluation.mkdir(parents=True,exist_ok=True)
    torch.cuda.synchronize();started=time.perf_counter()
    embedding=b.forward_numpy(student,x,device)
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    finite(embedding,'frozen Student output')
    path=evaluation/'embedding.npy';np.save(path,embedding)
    digest=legacy.sha(path)
    frozen=b.now()
    # Labels are read only after frozen Student output has been saved and hashed.
    label_path=legacy.DATA[target_key]/'processed/label_harmonized.csv'
    with label_path.open() as handle: labels=list(csv.DictReader(handle))
    if len(labels)!=len(cells) or any(row['barcode']!=cell for row,cell in zip(labels,cells)):
        raise ValueError('Evaluation labels/barcodes misaligned')
    y=np.unique(np.asarray([row['cell_type'] for row in labels])[index['test']],return_inverse=True)[1]
    if legacy.sha(path)!=digest: raise ValueError('Frozen embedding changed')
    metrics=b.evaluate(embedding,y)
    scaled=b.StandardScaler().fit_transform(embedding)
    neighbors=NearestNeighbors(n_neighbors=min(16,len(y))).fit(scaled).kneighbors(scaled,return_distance=False)
    purity=[]
    for i,row in enumerate(neighbors):
        row=row[row!=i][:15]
        purity.append(np.mean(y[row]==y[i]))
    metrics['knn_purity_at_15']=float(np.mean(purity))
    np.save(evaluation/'test_index.npy',index['test'])
    np.save(evaluation/'gene_hit_mask.npy',hit)
    (evaluation/'cell_order.tsv').write_text('barcode\n'+'\n'.join(np.asarray(cells)[index['test']])+'\n')
    result=dict(status='COMPLETE',run_tier='pre_final',source=source_key,target=target_key,seed=seed,
        scope='intra' if source_key==target_key else 'zero-shot inter; posthoc clustering, not supervised label transfer',
        metrics=metrics,student_checkpoint_sha256=legacy.sha(unit/'checkpoints/student/best.pt'),
        source_cache_manifest_sha256=legacy.sha(cache/'manifest.json'),
        target_rna_source=str(raw_path),target_rna_sha256=legacy.sha(raw_path),provenance=provenance,
        gene_hit_count=int(hit.sum()),gene_missing_count=int((~hit).sum()),gene_hit_fraction=float(hit.mean()),
        gene_missing_ids=genes[~hit].tolist(),input_scaler='Source-train only',
        target_atac_used=False,target_fit=False,target_adaptation=False,embedding_frozen_at=frozen,
        label_read_after_embedding_hash=True,evaluation_transductive_scaling=True,known_k=True,
        inference_seconds=seconds,cells_per_second=len(x)/seconds,
        hashes={str(p):legacy.sha(p) for p in evaluation.iterdir() if p.is_file()})
    legacy.write(evaluation/'manifest.json',result)
    return result


def train(key,seed,out,cache,manifest,smoke=False,evaluation_targets=None,evaluate_fn=None):
    targets=('D1','D2','D3','D4') if evaluation_targets is None else evaluation_targets
    evaluator=evaluate if evaluate_fn is None else evaluate_fn
    unit=out/('_smoke' if smoke else 'models')/key/f'seed_{seed}'
    if (unit/'manifest.json').exists():
        completed=json.loads((unit/'manifest.json').read_text())
        if completed['status']=='COMPLETE':
            legacy.checked(unit/'training_manifest.json')
            for target in targets:
                legacy.checked(unit/'evaluation'/f'{key}_to_{target}'/'manifest.json')
            print('VERIFIED_EXISTING',key,seed,flush=True)
            return
    fresh=not any(unit.rglob('*.pt'))
    workflow_started=efficiency.begin('cuda:0')
    src,mean,std=source_data(cache,key)
    b.CONFIG=copy.deepcopy(BASE_CONFIG)
    b.CONFIG.update(balanced_batch_counts=[256],run_tier='smoke' if smoke else 'pre_final',
                    validation='single Source validation; original sample-weighted stage loss')
    if smoke:
        b.CONFIG['epochs']=1
        for split,n in [('train',128),('val',64)]:
            src[split]={k:v[:n] for k,v in src[split].items()}
    inputs=dict(manifest['hashes'])
    inputs.update({str(REFERENCE):REFERENCE_SHA,str(Path(__file__)):legacy.sha(Path(__file__))})
    device=torch.device('cuda:0')
    started=time.perf_counter();b.seed_all(seed)
    teacher=b.Teacher(len(mean),len(mean)).to(device)
    tick=efficiency.begin(device,reset=False)
    teacher=b.train_stage(teacher,[src],unit/'checkpoints/teacher',seed,device,True,inputs)
    teacher_efficiency=efficiency.end(tick,device)
    for parameter in teacher.parameters(): finite(parameter.detach().cpu().numpy(),'Teacher parameter')
    if not np.isfinite(b.read(unit/'checkpoints/teacher/manifest.json')['best_source_val_score']):
        raise ValueError('Teacher validation loss NaN/Inf')
    diagnostic=warnings(teacher,src,device)
    legacy.write(unit/'research_warnings.json',diagnostic)
    teacher.requires_grad_(False)
    latent=src['train']['teacher_latent']
    src.update(latent_mean=latent.mean(0),latent_std=latent.std(0))
    np.savez(unit/'latent_state.npz',mean=src['latent_mean'],std=src['latent_std'])
    inputs[str(unit/'checkpoints/teacher/best.pt')]=legacy.sha(unit/'checkpoints/teacher/best.pt')
    b.seed_all(seed)
    student=copy.deepcopy(teacher.rna_encoder).requires_grad_(True).to(device)
    tick=efficiency.begin(device,reset=False)
    student=b.train_stage(student,[src],unit/'checkpoints/student',seed,device,False,inputs)
    student_efficiency=efficiency.end(tick,device)
    for parameter in student.parameters(): finite(parameter.detach().cpu().numpy(),'Student parameter')
    if not np.isfinite(b.read(unit/'checkpoints/student/manifest.json')['best_source_val_score']):
        raise ValueError('Student validation loss NaN/Inf')
    student.requires_grad_(False)
    for split in ('train','val'):
        z=b.forward_numpy(student,src[split]['rna'],device)
        finite(z,'Source Student latent')
        np.save(unit/f'{split}_student_embedding.npy',z)
    stages={s:b.read(unit/f'checkpoints/{s}/manifest.json') for s in ('teacher','student')}
    trained=dict(status='SMOKE_COMPLETE' if smoke else 'TRAINING_COMPLETE',source=key,seed=seed,
        configuration=b.CONFIG,source_manifest_sha256=legacy.sha(cache/'manifest.json'),
        source_only=True,labels_used=False,target_used=False,student_frozen_at=b.now(),
        stages=stages,wall_seconds=time.perf_counter()-started,
        frozen_reference_sha256=REFERENCE_SHA,
        hashes={str(p):legacy.sha(p) for p in unit.rglob('*.pt')},
        environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(),gpu_vram=torch.cuda.get_device_properties(0).total_memory))
    train_resources=dict(seconds=teacher_efficiency['seconds']+student_efficiency['seconds'],
        peak_allocated_bytes=max(teacher_efficiency['peak_allocated_bytes'],student_efficiency['peak_allocated_bytes']),
        peak_reserved_bytes=max(teacher_efficiency['peak_reserved_bytes'],student_efficiency['peak_reserved_bytes']))
    fitting=efficiency.end(workflow_started,device)
    fitting['source_workflow_seconds']=fitting['seconds'] if fresh else None
    trained['efficiency_v2']=efficiency.record(train_resources,fitting,None,
        sum(s['resources']['parameters'] for s in stages.values()),sum(p.numel() for p in student.parameters()),256,device)
    trained['efficiency_v2'].update(fresh=fresh,stages=dict(teacher=teacher_efficiency,student=student_efficiency),
        checkpoint_sha256=legacy.sha(unit/'checkpoints/student/best.pt'),
        source_input_manifest_sha256=legacy.sha(cache/'manifest.json'))
    legacy.write(unit/'training_manifest.json',trained)
    if not smoke:
        for target in targets:
            evaluator(unit,key,target,student,cache,mean,std,seed,device)
        legacy.write(unit/'manifest.json',dict(status='COMPLETE',source=key,seed=seed,training=trained,
            evaluation_count=len(targets),run_tier='pre_final'))
    del teacher,student
    torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','smoke','run'],required=True)
    parser.add_argument('--source',choices=['D1','D2','D3','D4'],required=True)
    parser.add_argument('--seed',type=int,choices=SEEDS,default=SEEDS[0])
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if config.get('benchmark_identity_confirmed') is not True:
        raise ValueError('Benchmark identity must be explicitly resolved before execution')
    cache_root=Path(config['cache_root']);out=Path(config['result_root'])
    cache,manifest=prepare(args.source,cache_root,config['identities'][args.source])
    if args.stage=='prepare': return
    with (ROOT/'.benchmark_a30.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        with threadpool_limits(2):
            train(args.source,args.seed,out,cache,manifest,smoke=args.stage=='smoke')


if __name__=='__main__': main()
