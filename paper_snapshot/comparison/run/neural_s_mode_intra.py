"""Fixed-split S-intra runner for scMomer-S and ECLARE-S."""
import argparse
import importlib.util
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

from pbmc_intra_common import ROOT, SEEDS, prepare, load, read, write, sha, evaluate
import efficiency_v2 as efficiency


def adapter(method):
    spec = importlib.util.spec_from_file_location(f'{method}_adapter',ROOT/f'comparison/run/{method}/s_mode_adapter.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def training(data, config, seed, smoke, device, history, stages):
    import torch
    def train(model, loss, path, stage, extra=None):
        parameters = [p for p in model.parameters() if p.requires_grad]
        if extra is not None:
            parameters += list(extra.parameters())
        tick = efficiency.begin(device, reset=False)
        optimizer = torch.optim.AdamW(parameters,lr=config['learning_rate'],weight_decay=config['weight_decay'])
        best,stale = float('inf'),0
        for epoch in range(1,(1 if smoke else config['max_epochs'])+1):
            model.train()
            if hasattr(model,'rna_model'):
                model.rna_model.eval()
            order = np.random.permutation(data['rna_train'].shape[0])
            for i in range(0,len(order),config['batch_size']):
                optimizer.zero_grad()
                value = loss('train',order[i:i+config['batch_size']])
                if not torch.isfinite(value):
                    raise ValueError(f'Nonfinite training loss: {stage}')
                value.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                    raise ValueError(f'Nonfinite gradients: {stage}')
                optimizer.step()
            model.eval()
            count = data['rna_val'].shape[0]
            with torch.no_grad(),torch.random.fork_rng(devices=[torch.cuda.current_device()] if device=='cuda' else []):
                torch.manual_seed(seed)
                score = sum(float(loss('val',np.arange(i,min(i+config['batch_size'],count))))*min(config['batch_size'],count-i) for i in range(0,count,config['batch_size']))/count
            if not np.isfinite(score):
                raise ValueError(f'Nonfinite validation loss: {stage}')
            history.append(dict(stage=stage,epoch=epoch,validation_loss=score))
            if score < best:
                best,stale = score,0
                state = dict(state_dict=model.state_dict(),best_epoch=epoch,validation_loss=score)
                if extra is not None:
                    state['distillation_state_dict'] = extra.state_dict()
                torch.save(state,path)
            else:
                stale += 1
            if stale >= config['patience']:
                break
        state = torch.load(path,map_location=device,weights_only=False)
        model.load_state_dict(state['state_dict'])
        if extra is not None:
            extra.load_state_dict(state['distillation_state_dict'])
        stages.append(dict(stage=stage,trainable_parameters=sum(p.numel() for p in {id(p):p for p in parameters}.values()),
                           **efficiency.end(tick,device)))
        return state
    return train


def main(method):
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=['prepare','smoke','unit'])
    parser.add_argument('--dataset',required=True,choices=['D1','D2','D3','D4'])
    parser.add_argument('--seed',type=int,choices=SEEDS,default=SEEDS[0])
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--run-seeds',type=int,nargs='+',choices=SEEDS)
    args = parser.parse_args()
    if args.device=='cpu':
        os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(2)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    config = read(ROOT/f'comparison/configs/{method}/s_mode_pbmc_intra.json')
    if args.run_seeds:
        config['seeds']=args.run_seeds
        if args.seed not in args.run_seeds:
            raise ValueError('Unit seed must belong to requested run seeds')
    cache = prepare(args.dataset)
    if args.stage=='prepare':
        return
    smoke = args.stage=='smoke'
    if smoke and method=='scmomer':
        config['batch_size']=2; config['rna_batch_size']=1
    if not smoke:
        gate = ROOT/f'comparison/result/{method}/s_mode/intra/smoke/{args.dataset}/seed_{SEEDS[0]}/manifest.json'
        if not gate.exists() or read(gate)['status']!='SMOKE_PASS':
            raise RuntimeError('CPU smoke gate must pass before pre_final training')
    output_root = args.output_root or ROOT/f'comparison/result/{method}/s_mode/intra'/('smoke' if smoke else 'pre_final')
    unit = output_root/args.dataset/f'seed_{args.seed}'
    if unit.exists():
        raise RuntimeError(f'Existing unit preserved: {unit}')
    lock = None
    unit_created = False
    try:
        if args.device=='cuda':
            import fcntl
            lock = (ROOT/'.benchmark_a30.lock').open('a')
            print('Waiting for benchmark GPU lock',flush=True)
            fcntl.flock(lock,fcntl.LOCK_EX)
            torch.cuda.reset_peak_memory_stats()
        if unit.exists():
            raise RuntimeError(f'Existing unit preserved: {unit}')
        unit.mkdir(parents=True)
        unit_created = True
        write(unit/'request.json',dict(method=method,dataset=args.dataset,seed=args.seed,config=config,command=sys.argv))
        started = time.perf_counter()
        workflow_started=started
        data, provenance = load(cache,smoke)
        input_load_seconds = time.perf_counter()-started
        if smoke:
            for k in data:
                data[k] = data[k][:8 if k.endswith('_train') else 4] if method=='scmomer' else data[k]
        for split in ('train','val','test'):
            np.save(unit/f'{split}_index.npy',np.load(cache/f'{split}_index.npy')[:data[f'rna_{split}'].shape[0]])
        np.save(unit/'gene_ids.npy',np.load(cache/'gene_ids.npy'))
        history = []
        module = adapter(method)
        source_data = {k:v for k,v in data.items() if not k.endswith('_test')}
        stages=[]
        fit_started=efficiency.begin(args.device)
        model,result = module.fit(source_data,config,unit,args.seed,smoke,args.device,cache,
            training(source_data,config,args.seed,smoke,args.device,history,stages))
        fitting=efficiency.end(fit_started,args.device)
        fitting['source_workflow_seconds']=time.perf_counter()-workflow_started
        if args.device=='cuda':
            torch.cuda.synchronize()
        training_seconds = time.perf_counter()-started
        checkpoint = torch.load(unit/'best.pt',map_location='cpu',weights_only=False)
        model.eval().requires_grad_(False)
        model.zero_grad(set_to_none=True)
        inference_parameters = sum(p.numel() for p in model.parameters())
        snapshot = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        batch=config.get('rna_batch_size',config['batch_size'])
        module.predict(model,data['rna_log_test'][:batch],checkpoint,args.device,batch)
        started = efficiency.begin(args.device)
        z = module.predict(model,data['rna_log_test'],checkpoint,args.device,config.get('rna_batch_size',config['batch_size']))
        if args.device=='cuda':
            torch.cuda.synchronize()
        inference=efficiency.end(started,args.device)
        inference_seconds = inference['seconds']
        peak_allocated = max(fitting['peak_allocated_bytes'],inference['peak_allocated_bytes']) if args.device=='cuda' else None
        peak_reserved = max(fitting['peak_reserved_bytes'],inference['peak_reserved_bytes']) if args.device=='cuda' else None
        training_record=dict(seconds=sum(x['seconds'] for x in stages),
            peak_allocated_bytes=fitting['peak_allocated_bytes'],peak_reserved_bytes=fitting['peak_reserved_bytes'])
        v2=efficiency.record(training_record,fitting,inference,result['trainable_parameters'],inference_parameters,batch,args.device)
        v2.update(input_load_seconds=input_load_seconds,stages=stages,checkpoint_sha256=sha(unit/'best.pt'),
                  source_input_manifest_sha256=sha(cache/'manifest.json'))
        if not np.isfinite(z).all():
            raise ValueError('Nonfinite test embedding')
        for k,v in model.state_dict().items():
            if not torch.equal(v.detach().cpu(),snapshot[k]):
                raise RuntimeError(f'Frozen inference mutated {k}')
        del snapshot,model
        reloaded = module.make_model(checkpoint['peaks'],args.device)
        reloaded.load_state_dict(checkpoint['state_dict'],strict=True)
        reloaded.eval().requires_grad_(False)
        subset = data['rna_log_test'][:4 if method=='scmomer' else 64]
        np.testing.assert_allclose(z[:subset.shape[0]],module.predict(reloaded,subset,checkpoint,args.device,2 if method=='scmomer' else 17),rtol=1e-4,atol=1e-5)
        np.save(unit/'rna_test_embedding.npy',z)
        write(unit/'history.json',history)
        result.update(efficiency_v2=v2,status='SMOKE_PASS' if smoke else 'TRAINED',run_tier='temporary_smoke' if smoke else 'pre_final',
            method='scMomer-S' if method=='scmomer' else 'ECLARE-S',dataset=args.dataset,seed=args.seed,
            identity=provenance['identity'],family='s_mode_intra',input_dimensions=[provenance['rna_genes'],provenance['atac_peaks']],
            split_counts={s:data[f'rna_{s}'].shape[0] for s in ('train','val','test')},
            test_atac_loaded=False,labels_used_for_fitting=False,test_used_for_fitting=False,
            cache_manifest_sha256=sha(cache/'manifest.json'),checkpoint_sha256=sha(unit/'best.pt'),
            training_seconds=training_seconds,inference_seconds=inference_seconds,total_wall_seconds=training_seconds+inference_seconds,
            cells_per_second=len(z)/inference_seconds,runtime_executable=sys.executable,
            inference_parameters=inference_parameters,
            trainable_parameters_scope='sum of parameters fitted across Source teacher and student stages; frozen pretrained RNA excluded',
            configured_seeds=config['seeds'],seed_count=len(config['seeds']),
            total_time_scope='input loading + training/construction + frozen inference; excludes queue, posthoc scoring and historical cache build',
            gpu_model=torch.cuda.get_device_name() if args.device=='cuda' else None,
            peak_gpu_allocated_bytes=peak_allocated,
            peak_gpu_reserved_bytes=peak_reserved,
            finite_forward_backward=True,reload_equal=True,batch_size_invariant=True,frozen_state_unchanged=True,
            prediction_sha256=sha(unit/'rna_test_embedding.npy'),
            code_sha256={str(p):sha(p) for p in [Path(__file__),ROOT/f'comparison/run/{method}/s_mode_adapter.py',ROOT/'comparison/run/pbmc_intra_common.py']},
            official_code_sha256={str(p):sha(p) for p in sorted((ROOT/f'comparison/methods/{method}/official').rglob('*.py'))})
        write(unit/'manifest.json',result)
        if not smoke:
            evaluate(unit,cache,args.dataset)
            result['status']='COMPLETE';write(unit/'manifest.json',result)
        print(result['status'],method,args.dataset,flush=True)
    except Exception as error:
        if unit_created:
            write(unit/'failure.json',dict(status='FAILED',error=repr(error)))
        raise
    finally:
        if lock is not None:
            lock.close()
