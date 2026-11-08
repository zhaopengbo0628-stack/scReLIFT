"""Official Multigrate/Cobolt modules and explicitly adapted Palette-S."""
import argparse
import importlib.util
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import linalg, sparse
from scipy.io import mmwrite

from pbmc_intra_common import ROOT, SEEDS, prepare, load, read, write, sha, evaluate
import efficiency_v2 as efficiency


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(method, dims, latent, device):
    import torch
    official = ROOT/'comparison/methods'/method/'official'
    if method == 'multigrate':
        sys.path.insert(0,str(official/'src'))
        from multigrate.module import MultiVAETorch
        import inspect
        assert Path(inspect.getfile(MultiVAETorch)).resolve().is_relative_to(official.resolve())
        model = MultiVAETorch(modality_lengths=dims, z_dim=latent, losses=['nb','mse'],
            condition_encoders=False, condition_decoders=False, normalization='layer',
            cat_covariate_dims=[], cont_covariate_dims=[],
            cat_covs_idx=torch.empty(0,dtype=torch.long),
            cont_covs_idx=torch.empty(0,dtype=torch.long))
    else:
        spec = importlib.util.spec_from_file_location('official_cobolt',official/'cobolt/model/coboltmodel.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.CoboltModel(in_channels=dims,latent_dim=latent,n_dataset=[1,1],log=True)
        model.device = device
        model.mu2 = model.mu2.to(device)
        model.var2 = model.var2.to(device)
    return model.to(device)


def tensor(matrix, device):
    import torch
    return torch.as_tensor(matrix.toarray(),dtype=torch.float32,device=device)


def loss(model, method, rna, atac, rows, combination, weight, device):
    import torch
    r = tensor(rna[rows],device)
    a = tensor(atac[rows],device)
    if method == 'multigrate':
        from scvi import REGISTRY_KEYS
        tensors = {REGISTRY_KEYS.X_KEY:torch.cat([r,a],dim=1),
                   REGISTRY_KEYS.SIZE_FACTOR_KEY:r.sum(1,keepdim=True).clamp_min(1)}
        return model(tensors, loss_kwargs={'kl_weight':weight})[2].loss
    xs = [r if combination[0] else None,a if combination[1] else None]
    ids = [torch.zeros(len(rows),dtype=torch.long,device=device) if present else None
           for present in combination]
    kl, reconstruction = model([xs,ids],elbo_combn=[combination])
    return weight*kl+reconstruction


def predict(model, method, rna, peaks, device, batch_size):
    import torch
    output = []
    with torch.inference_mode():
        for offset in range(0,rna.shape[0],batch_size):
            r = tensor(rna[offset:offset+batch_size],device)
            if method == 'multigrate':
                a = torch.zeros((len(r),peaks),dtype=r.dtype,device=device)
                z = model.inference(torch.cat([r,a],dim=1))['mu'].cpu().numpy()
            else:
                ids = torch.zeros(len(r),dtype=torch.long,device=device)
                z = model.get_latent([[r,None],[ids,None]],elbo_bool=[True,False])
            if not np.isfinite(z).all():
                raise ValueError('Nonfinite frozen inference')
            output.append(z)
    return np.concatenate(output)


def torch_run(method, data, config, unit, seed, smoke, device, workflow_started=None):
    import torch
    started = efficiency.begin(device)
    seed_all(seed)
    torch.set_num_threads(2)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    dims = [data['rna_train'].shape[1],data['atac_train'].shape[1]]
    model = make_model(method,dims,config['latent_dim'],device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if method == 'multigrate':
        optimizer = torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=1e-3)
        combinations = [None]
        atac_key = 'atac_tfidf'
    else:
        optimizer = torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
        combinations = [[False,True],[True,False],[True,True]]
        atac_key = 'atac'
    epochs = 2 if smoke else config['max_epochs']
    batch = config['batch_size']
    history, best, stale = [], float('inf'), 0
    rng = np.random.default_rng(seed)
    train_started = efficiency.begin(device, reset=False)
    for epoch in range(1,epochs+1):
        model.train()
        warmup = config['kl_warmup_epochs']
        weight = min(1.,(epoch-1)/warmup)
        for combination in combinations:
            order = rng.permutation(data['rna_train'].shape[0])
            batches = list(np.array_split(order,max(1,int(np.ceil(len(order)/batch)))))
            if len(batches[-1]) == 1 and len(batches)>1:
                tail = batches.pop()
                batches[-1] = np.concatenate([batches[-1],tail])
            for rows in batches:
                optimizer.zero_grad()
                value = loss(model,method,data['rna_train'],data[f'{atac_key}_train'],
                             rows,combination,weight,device)
                if not torch.isfinite(value):
                    raise ValueError('Nonfinite training loss')
                value.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('Nonfinite gradient')
                optimizer.step()
        model.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=[0] if device=='cuda' else []):
            torch.manual_seed(seed)
            score, count = 0., data['rna_val'].shape[0]
            for offset in range(0,count,batch):
                rows = np.arange(offset,min(count,offset+batch))
                for combination in combinations:
                    value = loss(model,method,data['rna_val'],data[f'{atac_key}_val'],
                                 rows,combination,1.,device).item()
                    score += value*len(rows) if method=='multigrate' else value
            score /= count
        if not np.isfinite(score):
            raise ValueError('Nonfinite validation loss')
        history.append(dict(epoch=epoch,source_validation_loss=score,kl_weight=weight))
        if score < best:
            best, stale = score, 0
            torch.save(dict(state_dict=model.state_dict(),dims=dims,latent_dim=config['latent_dim'],
                best_epoch=epoch,validation_loss=best,seed=seed,method=method),unit/'best.pt')
        else:
            stale += 1
        if stale >= config['patience']:
            break
    training = efficiency.end(train_started,device)
    fit_seconds = time.perf_counter()-started
    best_state = torch.load(unit/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(best_state['state_dict'])
    model.eval().requires_grad_(False)
    fitting = efficiency.end(started,device)
    fitting['source_workflow_seconds'] = time.perf_counter()-workflow_started if workflow_started is not None else None
    best_epoch=best_state['best_epoch']
    model.zero_grad(set_to_none=True)
    del optimizer, value, best_state
    frozen_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    predict(model,method,data['rna_test'][:batch],dims[1],device,batch)
    started = efficiency.begin(device)
    z = predict(model,method,data['rna_test'],dims[1],device,batch)
    if device == 'cuda':
        torch.cuda.synchronize()
    inference = efficiency.end(started,device)
    inference_seconds = inference['seconds']
    peak_allocated = max(training['peak_allocated_bytes'],inference['peak_allocated_bytes']) if device=='cuda' else None
    peak_reserved = max(training['peak_reserved_bytes'],inference['peak_reserved_bytes']) if device=='cuda' else None
    check_rna = data['rna_test'][:64]
    reference = z[:64]
    np.testing.assert_allclose(reference,predict(model,method,check_rna,dims[1],device,17),rtol=1e-4,atol=1e-5)
    for k,v in model.state_dict().items():
        assert torch.equal(v.detach().cpu(),frozen_state[k]), f'Inference mutated {k}'
    reloaded = make_model(method,dims,config['latent_dim'],device)
    reloaded.load_state_dict(torch.load(unit/'best.pt',map_location=device,weights_only=False)['state_dict'])
    reloaded.eval().requires_grad_(False)
    np.testing.assert_allclose(reference,predict(reloaded,method,check_rna,dims[1],device,batch),rtol=1e-5,atol=1e-6)
    np.save(unit/'rna_test_embedding.npy',z)
    write(unit/'history.json',history)
    import importlib.metadata
    versions = {package:importlib.metadata.version(package) for package in
                (['torch','scvi-tools'] if method=='multigrate' else ['torch'])}
    return dict(best_epoch=best_epoch,best_validation_loss=best,
        efficiency_v2=efficiency.record(training,fitting,inference,params,sum(p.numel() for p in model.parameters()),batch,device),
        checkpoint_sha256=sha(unit/'best.pt'),trainable_parameters=params,
        inference_parameters=sum(p.numel() for p in model.parameters()),
        training_seconds=fit_seconds,inference_seconds=inference_seconds,
        cells_per_second=len(z)/max(inference_seconds,1e-12),runtime_versions=versions,
        peak_gpu_allocated_bytes=peak_allocated, peak_gpu_reserved_bytes=peak_reserved,
        gpu_model=torch.cuda.get_device_name() if device=='cuda' else None,
        latent='RNA-only PoE posterior mean',optional_cobolt_xgboost_correction=False,
        finite_forward_backward=True,reload_equal=True,batch_size_invariant=True,frozen_state_unchanged=True)


def palette_run(data,config,unit,seed,cache,workflow_started=None):
    started = time.perf_counter()
    cells = np.asarray((cache/'barcodes.txt').read_text().splitlines())
    train_ids = cells[np.load(cache/'train_index.npy')][:data['rna_train'].shape[0]]
    for modal,features in [('rna',np.load(cache/'gene_ids.npy')),('atac',np.load(cache/'peak_ids.npy'))]:
        matrix = data['rna_log_train'] if modal=='rna' else data['atac_tfidf_train']
        mmwrite(unit/f'{modal}.mtx',matrix.T)
        (unit/f'{modal}_features.txt').write_text('\n'.join(features)+'\n')
    (unit/'train_cells.txt').write_text('\n'.join(train_ids)+'\n')
    cmd=[config['rscript'],str(ROOT/'comparison/run/palette/fit_source.R'),str(unit),
         str(seed),str(config['latent_dim']),str(ROOT/'comparison/methods/palette/official/R/PaletteSC.R')]
    with (unit/'palette.log').open('w') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    source=np.loadtxt(unit/'source_embedding.tsv',delimiter='\t',ndmin=2)
    assert source.shape == (len(train_ids),config['latent_dim']) and np.isfinite(source).all()
    # Source-only minimum-norm linear projection, explicitly a protocol adaptation.
    weights,_,rank,_=linalg.lstsq(data['rna_log_train'].toarray(),source,lapack_driver='gelsd')
    np.savez_compressed(unit/'frozen_map.npz',weights=weights,genes=np.load(cache/'gene_ids.npy'),seed=seed)
    fitting=efficiency.end(started,'cpu')
    fitting['source_workflow_seconds']=time.perf_counter()-workflow_started if workflow_started is not None else None
    training_seconds=fitting['seconds']
    data['rna_log_test'][:config['batch_size']]@weights
    started=time.perf_counter()
    z=data['rna_log_test']@weights
    inference=efficiency.end(started,'cpu')
    inference_seconds=inference['seconds']
    assert np.isfinite(z).all()
    frozen_hash=sha(unit/'frozen_map.npz')
    reloaded=np.load(unit/'frozen_map.npz')['weights']
    np.testing.assert_allclose(z,data['rna_log_test']@reloaded,rtol=1e-7,atol=1e-8)
    pieces=np.vstack([data['rna_log_test'][i:i+17]@reloaded for i in range(0,len(z),17)])
    np.testing.assert_allclose(z,pieces,rtol=1e-7,atol=1e-8)
    assert frozen_hash==sha(unit/'frozen_map.npz')
    np.save(unit/'rna_test_embedding.npy',z.astype(np.float32))
    return dict(efficiency_v2=efficiency.record(fitting,fitting,inference,None,None,config['batch_size'],'cpu'),
        best_epoch=None,best_validation_loss=None,
        validation_criterion='Not applicable: non-iterative Source-only construction, no test tuning',
        checkpoint_sha256=sha(unit/'source_palette.rds'),frozen_map_sha256=frozen_hash,
        trainable_parameters=None,map_rank=int(rank),training_seconds=training_seconds,
        inference_seconds=inference_seconds,cells_per_second=len(z)/max(inference_seconds,1e-12),
        gpu_model=None,peak_gpu_allocated_bytes=None,peak_gpu_reserved_bytes=None,
        latent='Palette-S: official Source SVD + Run.Palette + train-only minimum-norm RNA map',
        runtime_versions=read(unit/'r_runtime.json'),reload_equal=True,
        batch_size_invariant=True,frozen_state_unchanged=True)


def main(method=None):
    parser=argparse.ArgumentParser()
    if method is None:
        parser.add_argument('--method',required=True,choices=['multigrate','cobolt','palette'])
    parser.add_argument('--stage',required=True,choices=['prepare','smoke','unit'])
    parser.add_argument('--dataset',required=True,choices=['D1','D2','D3','D4'])
    parser.add_argument('--seed',type=int,choices=SEEDS,default=SEEDS[0])
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--run-seeds',type=int,nargs='+',choices=SEEDS)
    args=parser.parse_args()
    if args.device=='cpu':
        os.environ['CUDA_VISIBLE_DEVICES']=''
    method=method or args.method
    config=read(ROOT/f'comparison/configs/{method}/s_mode_pbmc_intra.json')
    if args.run_seeds:
        config['seeds']=args.run_seeds
        if args.seed not in args.run_seeds:
            raise ValueError('Unit seed must belong to requested run seeds')
    cache=prepare(args.dataset)
    if args.stage=='prepare':
        return
    smoke=args.stage=='smoke'
    if not smoke:
        gate=ROOT/f'comparison/result/{method}/s_mode/intra/smoke'/args.dataset/f'seed_{SEEDS[0]}'/'manifest.json'
        if not gate.exists() or read(gate)['status']!='SMOKE_PASS':
            raise RuntimeError('Pass the dataset CPU smoke gate before a pre_final unit')
    output_root=args.output_root or ROOT/f'comparison/result/{method}/s_mode/intra'/('smoke' if smoke else 'pre_final')
    unit=output_root/args.dataset/f'seed_{args.seed}'
    if unit.exists():
        raise RuntimeError(f'Existing unit preserved: {unit}')
    lock=None
    unit_created=False
    try:
        if args.device=='cuda' and method!='palette':
            import fcntl
            lock=(ROOT/'.benchmark_a30.lock').open('a')
            print('Waiting for benchmark GPU lock',flush=True)
            fcntl.flock(lock,fcntl.LOCK_EX)
        if unit.exists():
            raise RuntimeError(f'Existing unit preserved: {unit}')
        unit.mkdir(parents=True)
        unit_created=True
        started=time.perf_counter()
        workflow_started=started
        data,manifest=load(cache,smoke)
        input_load_seconds=time.perf_counter()-started
        np.save(unit/'gene_ids.npy',np.load(cache/'gene_ids.npy'))
        for split in ('train','val','test'):
            np.save(unit/f'{split}_index.npy',np.load(cache/f'{split}_index.npy')[:data[f'rna_{split}'].shape[0]])
        write(unit/'request.json',dict(method=method,config=config,dataset=args.dataset,seed=args.seed,
            smoke=smoke,device=args.device,command=sys.argv,cache_manifest_sha256=sha(cache/'manifest.json')))
        result=palette_run(data,config,unit,args.seed,cache,workflow_started) if method=='palette' else torch_run(method,data,config,unit,args.seed,smoke,args.device,workflow_started)
        result.update(status='SMOKE_PASS' if smoke else 'TRAINED',run_tier='temporary_smoke' if smoke else 'pre_final',
            method='Palette-S' if method=='palette' else method,dataset=args.dataset,seed=args.seed,
            identity=manifest['identity'],family='s_mode_intra',
            split_counts={s:data[f'rna_{s}'].shape[0] for s in ('train','val','test')},
            input_dimensions=[manifest['rna_genes'],manifest['atac_peaks']],
            test_atac_loaded=False,labels_used_for_fitting=False,test_used_for_fitting=False,
            preprocessing_seconds=0.,input_load_seconds=input_load_seconds,
            historical_cache_build_seconds=manifest['preprocessing_seconds'],
            configured_seeds=config['seeds'],seed_count=len(config['seeds']),
            code_sha256={str(p):sha(p) for p in [Path(__file__),ROOT/'comparison/run/pbmc_intra_common.py']},
            official_source=read(ROOT/'comparison/docs/OFFICIAL_CODE_DOWNLOADS.json')['current'][method],
            official_code_sha256={str(p):sha(p) for p in sorted((ROOT/'comparison/methods'/method/'official').rglob('*.R' if method=='palette' else '*.py'))},
            runtime_executable=sys.executable,
            prediction_sha256=sha(unit/'rna_test_embedding.npy'))
        result['efficiency_v2'].update(input_load_seconds=input_load_seconds,
            source_input_manifest_sha256=sha(cache/'manifest.json'),checkpoint_sha256=result.get('frozen_map_sha256',result['checkpoint_sha256']),
            fitting_scope='Source model initialization + fitting + best checkpoint reload; add input_load_seconds for Source workflow')
        result['total_wall_seconds']=input_load_seconds+result['training_seconds']+result['inference_seconds']
        result['total_time_scope']='input loading + training/construction + frozen inference; excludes queue, posthoc scoring and historical cache build'
        if method=='palette':
            result['code_sha256'][str(ROOT/'comparison/run/palette/fit_source.R')]=sha(ROOT/'comparison/run/palette/fit_source.R')
        write(unit/'manifest.json',result)
        if not smoke:
            evaluate(unit,cache,args.dataset)
            result['status']='COMPLETE'
            write(unit/'manifest.json',result)
        print(result['status'],method,args.dataset,args.seed,flush=True)
    except Exception as error:
        if unit_created:
            write(unit/'failure.json',dict(status='FAILED',error=repr(error)))
        raise
    finally:
        if lock is not None:
            lock.close()


if __name__=='__main__':
    main()
