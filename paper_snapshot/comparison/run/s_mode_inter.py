"""Frozen Source checkpoint inference on processed PBMC Target RNA only."""
import argparse
import csv
import fcntl
import importlib.util
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, adjusted_mutual_info_score, adjusted_rand_score,
    balanced_accuracy_score, f1_score, normalized_mutual_info_score, silhouette_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from pbmc_intra_common import (ROOT, CACHE_ROOT, EVALUATOR, read, write, sha, prepare,
                               cache_for, benchmark_for, expected_hash, relocated)
import efficiency_v2 as efficiency

KEYS = ['D1', 'D2', 'D3', 'D4']
BASE = CACHE_ROOT
INPUTS = CACHE_ROOT
BENCH = ROOT/'data/benchmark/PBMC'
PROTOCOL = ROOT/'comparison/docs/PBMC_SMODE_SCORING_TIMING_V1.json'
NAMES = dict(multigrate='Multigrate', cobolt='Cobolt', palette='Palette-S',
             eclare='ECLARE-S', scmomer='scMomer-S', b2='Full B2-S')
SCORES = {'ARI':'ari', 'NMI':'nmi', 'ACC':'cluster_acc', 'Macro-F1':'macro_f1',
          'Balanced ACC':'balanced_accuracy', 'Purity@15':'knn_purity_at_15', 'Silhouette':'silhouette'}


def prepare_inputs():
    """Select frozen Source genes after Target full-library normalization; no raw fallback."""
    sources = {s: prepare(s) for s in KEYS}
    for target in KEYS:
        folder = benchmark_for(target)
        target_cache = cache_for(target, 'pbmc_corrected_s10x20_v1_20261012')
        index = np.load(target_cache/'test_index.npy')
        files = [folder/'barcodes.txt', target_cache/'test_index.npy']
        assert files[0].read_text() == (target_cache/'barcodes.txt').read_text()
        reconstruction = None
        if target == 'D1':
            full = ROOT/'data/cache/d_mode/PBMC/ccrs_paused_history/40_full_b2_hvg_ccrs_2x2_20261011/shared/D1'
            provenance = read(full/'manifest.json')
            files += [full/'rna_full_log1p.npz', full/'gene_ids.npy', full/'manifest.json',
                      folder/'rna_counts.npz', folder/'gene_ids.npy']
            for path in files[:1]+files[2:4]:
                assert sha(path) == expected_hash(provenance['hashes'],path)
            genes = np.load(full/'gene_ids.npy')
            log = sparse.load_npz(full/'rna_full_log1p.npz')[index].tocsr()
            lookup = {g:i for i,g in enumerate(genes)}
            columns = [lookup[g] for g in np.load(folder/'gene_ids.npy')]
            known = sparse.load_npz(folder/'rna_counts.npz')[index].tocsr()
            selected = log[:,columns].tocsr()
            known.sort_indices(); selected.sort_indices()
            np.testing.assert_array_equal(known.indices, selected.indices)
            np.testing.assert_array_equal(known.indptr, selected.indptr)
            assert np.all(np.diff(known.indptr)>0)
            ratios = known.data.astype(float)*1e4/np.expm1(selected.data.astype(float))
            totals = np.array([np.median(ratios[a:b]) for a,b in zip(known.indptr[:-1],known.indptr[1:])])
            total_error = float(np.max(np.abs(totals-np.rint(totals))))
            assert total_error < .01
            totals = np.rint(totals)
            values = np.expm1(log.data.astype(float))*np.repeat(totals/1e4,np.diff(log.indptr))
            integer_error = float(np.max(np.abs(values-np.rint(values))))
            assert integer_error < .01
            counts = log.copy(); counts.data = np.rint(values).astype(np.float32)
            np.testing.assert_array_equal(np.asarray(counts.sum(1)).ravel(), totals)
            assert (counts[:,columns] != known).nnz == 0
            # Independently validate this full cache against the currently frozen D1 panel.
            own = [lookup[g] for g in np.load(sources[target]/'gene_ids.npy')]
            np.testing.assert_allclose(log[:,own].toarray(),
                sparse.load_npz(sources[target]/'rna_log_test.npz').toarray(),rtol=0,atol=0)
            reconstruction = dict(method='Invert processed full-library CP10K log1p using library totals anchored by saved raw-count panel',
                maximum_library_rounding_error=total_error, maximum_count_rounding_error=integer_error,
                all_library_sums_exact=True, all_saved_counts_exact=True, current_cache_log_panel_exact=True)
        else:
            path = folder/('rna_counts.npz' if target=='D2' else 'rna.npz')
            files.append(path)
            if target == 'D2':
                counts = sparse.load_npz(path)[index].tocsr().astype(np.float32)
                files.append(target_cache/'all_gene_ids.npy')
                genes = np.load(files[-1])
            else:
                with np.load(path) as archive:
                    counts = sparse.csr_matrix((archive['data'],archive['indices'],archive['indptr']),
                        shape=tuple(archive['shape']))[index].astype(np.float32)
                    genes = archive['gene_ids'].copy()
            assert np.all(counts.data>=0) and np.all(counts.data==np.rint(counts.data))
            totals = np.asarray(counts.sum(1)).ravel()
            assert np.all(totals>0)
            log = counts.multiply((1e4/totals)[:,None]).tocsr()
            log.data = np.log1p(log.data)
        genes = np.asarray([str(g).split('.')[0] for g in genes])
        assert len(set(genes)) == len(genes) == counts.shape[1]
        lookup = {g:i for i,g in enumerate(genes)}
        for source, cache in sources.items():
            panel = np.load(cache/'gene_ids.npy')
            columns = np.array([lookup.get(str(g).split('.')[0],-1) for g in panel])
            hit = columns>=0
            projection = sparse.csr_matrix((np.ones(hit.sum(),np.float32),(columns[hit],np.flatnonzero(hit))),
                shape=(len(genes),len(panel)))
            out = cache_for(source, 'pbmc_comparison_inter_v1')/target
            evidence = dict(source=source,target=target,target_fit=False,target_atac_used=False,
                normalization='full Target library CP10K natural-log1p before Source gene selection',
                files={str(relocated(p).resolve()):sha(p) for p in files},source_genes_sha256=sha(cache/'gene_ids.npy'),
                count_reconstruction=reconstruction,observed_source_genes=int(hit.sum()),source_genes=len(panel))
            if (out/'manifest.json').exists():
                old = read(out/'manifest.json')
                old_evidence = old['input_evidence']
                old_evidence['files'] = {str(relocated(path).resolve()):digest
                                         for path,digest in old_evidence['files'].items()}
                assert old_evidence==evidence
                for name,digest in old['hashes'].items(): assert sha(out/name)==digest
                continue
            out.mkdir(parents=True,exist_ok=True)
            for name,x in [('rna_counts',counts),('rna_log',log)]:
                sparse.save_npz(out/f'{name}.npz',(x@projection).tocsr().astype(np.float32))
            np.save(out/'gene_hit_mask.npy',hit)
            np.save(out/'gene_ids.npy',panel)
            np.save(out/'test_index.npy',index)
            cells = np.asarray(target_cache.joinpath('barcodes.txt').read_text().splitlines())[index]
            (out/'barcodes.txt').write_text('\n'.join(cells)+'\n')
            if source==target:
                for name,original in [('rna_counts','rna_test'),('rna_log','rna_log_test')]:
                    np.testing.assert_allclose(sparse.load_npz(out/f'{name}.npz').toarray(),
                        sparse.load_npz(cache/f'{original}.npz').toarray(),rtol=1e-5,atol=1e-6)
            write(out/'manifest.json',dict(input_evidence=evidence,
                hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
            print('INPUT_READY',source,target,flush=True)


def score(unit,target):
    prediction = unit/'embedding.npy'
    digest = sha(prediction)
    protocol = read(PROTOCOL)
    path = benchmark_for(target)/'label_harmonized.csv'
    assert sha(path)==next(r['sha256'] for r in protocol['labels'] if r['dataset']==target)
    with path.open() as handle: rows = list(csv.DictReader(handle))
    assert [r['barcode'] for r in rows] == cache_for(target, 'pbmc_corrected_s10x20_v1_20261012').joinpath('barcodes.txt').read_text().splitlines()
    index = np.load(unit/'test_index.npy')
    y = np.unique(np.asarray([r['cell_type'] for r in rows])[index],return_inverse=True)[1]
    z = StandardScaler().fit_transform(np.load(prediction))
    assert len(z)==len(y) and np.isfinite(z).all() and EVALUATOR==protocol['scoring']['kmeans']
    k = len(set(y)); clusters = KMeans(n_clusters=k,**EVALUATOR).fit_predict(z)
    contingency = np.zeros((k,k),np.int64)
    np.add.at(contingency,(y,clusters),1)
    a,b = linear_sum_assignment(-contingency)
    mapping = dict(zip(b,a)); matched = np.array([mapping[v] for v in clusters])
    neighbors = NearestNeighbors(n_neighbors=16).fit(z).kneighbors(z,return_distance=False)
    neighbors = np.array([row[row!=i][:15] for i,row in enumerate(neighbors)])
    acc = float(accuracy_score(y,matched))
    result = dict(ari=float(adjusted_rand_score(y,clusters)),nmi=float(normalized_mutual_info_score(y,clusters)),
        ami=float(adjusted_mutual_info_score(y,clusters)),cluster_acc=acc,accuracy=acc,
        macro_f1=float(f1_score(y,matched,average='macro',zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y,matched)),
        weighted_f1=float(f1_score(y,matched,average='weighted',zero_division=0)),
        knn_purity_at_15=float(np.mean(y[neighbors]==y[:,None])),silhouette=float(silhouette_score(z,y)),
        geometry_pearson=None,geometry_spearman=None,label_sha256=sha(path),
        labels_loaded_after_prediction_frozen=True,annotation_scope='posthoc Hungarian clustering; not label transfer')
    assert sha(prediction)==digest
    write(unit/'metrics.json',result)
    return result


def run(args):
    method,source,seed = args.method,args.source,args.seed
    trained = read(args.unit/'manifest.json')
    workflow_started=efficiency.begin('cpu' if method=='palette' else 'cuda')
    target_workflows=[]
    path = args.unit/('frozen_map.npz' if method=='palette' else 'student_best.pt' if method=='b2' else 'best.pt')
    if args.efficiency_v2 and method=='b2': path=args.unit/'checkpoints/student/best.pt'
    digest = sha(path)
    if method=='b2':
        training = read(args.unit/'training_manifest.json')
        stages = [r['resources'] for r in training['stages'].values()]
        training_peak = max(r['peak_allocated_bytes'] for r in stages)
        parameters = sum(r['parameters'] for r in stages)
        old = ROOT/'main/result/s_mode/intra/62_smode_full_b2_pbmc_scoring_timing_5seed_20261014'
        expected = expected_hash(training['hashes'],path) if args.efficiency_v2 else read(old/'evaluations'/source/f'seed_{seed}'/KEYS[0]/'manifest.json')['checkpoint_sha256']
    else:
        expected = trained['frozen_map_sha256' if method=='palette' else 'checkpoint_sha256']
        training_peak,parameters = trained['peak_gpu_allocated_bytes'],trained['trainable_parameters']
    assert digest==expected
    device = 'cpu' if method=='palette' else 'cuda'
    batch = 1 if method=='scmomer' else 256
    if device=='cuda': torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    if method=='palette':
        weights = np.load(path)['weights']
        predict = lambda x: np.concatenate([np.asarray(x[i:i+batch]@weights) for i in range(0,x.shape[0],batch)])
        model = None
    elif method=='b2':
        spec = importlib.util.spec_from_file_location('inter_b2',ROOT/'main/core/run_b2_strict_lodo.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        model = module.Encoder(2000)
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
        model.eval().requires_grad_(False).cuda()
        state = np.load(cache_for(source, 'pbmc_corrected_s10x20_v1_20261012')/'rna_state.npz')
        def predict(x):
            with torch.inference_mode():
                return np.concatenate([model(torch.from_numpy(x[i:i+batch]).cuda()).cpu().numpy() for i in range(0,len(x),batch)])
    else:
        checkpoint = torch.load(path,map_location='cpu',weights_only=False)
        if method in ['multigrate','cobolt']:
            from s_mode_intra import make_model, predict as forward
            model = make_model(method,checkpoint['dims'],checkpoint['latent_dim'],device)
            predict = lambda x: forward(model,method,x,checkpoint['dims'][1],device,batch)
        else:
            from neural_s_mode_intra import adapter
            module = adapter(method)
            if method=='scmomer':
                for asset,expected_hash in checkpoint['asset_sha256'].items(): assert sha(asset)==expected_hash
            model = module.make_model(checkpoint['peaks'],device)
            predict = lambda x: module.predict(model,x,checkpoint,device,batch)
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        model.eval().requires_grad_(False)
    # Checkpoint reload must reproduce existing Source intra predictions before any inter unit.
    if method!='b2':
        native = 'rna_test' if method in ['multigrate','cobolt'] else 'rna_log_test'
        x = sparse.load_npz(CACHE_ROOT/source/f'{native}.npz')[:4]
        expected_z = np.load(args.unit/'rna_test_embedding.npy')[:4]
        np.testing.assert_allclose(predict(x),expected_z,rtol=1e-4,atol=1e-5)
    if args.smoke:
        print('SMOKE_PASS',method,source,seed,flush=True)
        return
    if args.efficiency_v2 and method=='b2':
        check=np.load(cache_for(source, 'pbmc_corrected_s10x20_v1_20261012')/'rna_train.npy')[:4]
        reference=np.load(args.unit/'train_student_embedding.npy')[:4]
        np.testing.assert_allclose(predict(((check-state['mean'])/(state['std']+1e-6)).astype(np.float32)),reference,rtol=1e-4,atol=1e-5)
    load_seconds=time.perf_counter()-workflow_started
    snapshot = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()} if model is not None else None
    for target in KEYS:
        if target==source: continue
        out = args.output/'evaluations'/source/f'seed_{seed}'/target
        if (out/'manifest.json').exists():
            record = read(out/'manifest.json')
            assert record['status']=='COMPLETE' and record['checkpoint_sha256']==digest
            assert record['protocol_sha256']==sha(PROTOCOL)
            assert sha(out/'embedding.npy')==record['prediction_sha256']
            continue
        out.mkdir(parents=True,exist_ok=True)
        started = time.perf_counter()
        inputs = cache_for(source, 'pbmc_comparison_inter_v1')/target
        input_manifest = read(inputs/'manifest.json')
        for name,expected_hash in input_manifest['hashes'].items(): assert sha(inputs/name)==expected_hash
        genes=cache_for(source, 'pbmc_corrected_s10x20_v1_20261012')/'gene_ids.npy' if method=='b2' and args.efficiency_v2 else args.unit/'gene_ids.npy'
        np.testing.assert_array_equal(np.load(genes),np.load(inputs/'gene_ids.npy'))
        x = sparse.load_npz(inputs/('rna_counts.npz' if method in ['multigrate','cobolt'] else 'rna_log.npz'))
        if method=='b2': x = ((x.toarray()-state['mean'])/(state['std']+1e-6)).astype(np.float32)
        predict(x[:batch])
        if device=='cuda': torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter(); z = predict(x)
        if device=='cuda': torch.cuda.synchronize()
        seconds = time.perf_counter()-tick
        peak = torch.cuda.max_memory_allocated() if device=='cuda' else None
        reserved = torch.cuda.max_memory_reserved() if device=='cuda' else None
        assert np.isfinite(z).all() and len(z)==x.shape[0]
        prediction_path=out/'embedding.npy'
        if args.efficiency_v2:
            np.save(prediction_path,z.astype(np.float32))
            target_workflows.append(dict(target=target,seconds=time.perf_counter()-started,
                inference_seconds=seconds,peak_allocated_bytes=peak,peak_reserved_bytes=reserved))
        if snapshot is not None:
            for k,v in model.state_dict().items(): assert torch.equal(v.detach().cpu(),snapshot[k]),k
        assert sha(path)==digest
        if method=='b2' and not args.efficiency_v2:
            old_unit = old/'evaluations'/source/f'seed_{seed}'/target
            expected_z = np.load(old_unit/'embedding.npy')
            np.testing.assert_allclose(z,expected_z,rtol=1e-4,atol=1e-5)
        np.save(out/'embedding.npy',z.astype(np.float32))
        for name in ['test_index.npy','barcodes.txt','gene_hit_mask.npy']: shutil.copy2(inputs/name,out/name)
        metrics = read(old_unit/'metrics.json') if method=='b2' and not args.efficiency_v2 else score(out,target)
        if method=='b2': write(out/'metrics.json',metrics)
        combined = max(training_peak or 0,peak or 0) if device=='cuda' else None
        display = dict(Method=NAMES[method],**{c:metrics[k] for c,k in SCORES.items()},
            **{'Total time (s)':None,'Trainable params':parameters,'Peak VRAM (GiB)':combined/(2**30) if combined else None,
               'Inference time (s)':seconds,'Cells/sec':len(z)/seconds,'GPU':torch.cuda.get_device_name() if device=='cuda' else 'CPU'})
        write(out/'manifest.json',dict(status='COMPLETE',run_tier='pre_final',family='inter',source=source,target=target,seed=seed,
            protocol_id=read(PROTOCOL)['protocol_id'],protocol_sha256=sha(PROTOCOL),display=display,metrics=metrics,
            checkpoint_path=str(path),checkpoint_sha256=digest,checkpoint_reused=True,prediction_sha256=sha(out/'embedding.npy'),
            input_manifest_sha256=sha(inputs/'manifest.json'),input_evidence=input_manifest['input_evidence'],
            inference_seconds=seconds,inference_peak_allocated_bytes=peak,inference_peak_reserved_bytes=reserved,
            source_training_peak_allocated_bytes=training_peak,peak_allocated_bytes=combined,batch_size=batch,warmup_batches=1,
            batch_exception='upstream Performer requires batch1 for query-independent inference' if method=='scmomer' else None,
            inference_parameters=sum(p.numel() for p in model.parameters()) if model is not None else None,
            total_time_seconds=None,total_time_reason='Checkpoint reuse; historical full workflow timer unavailable',
            evaluation_wall_seconds=time.perf_counter()-started,target_fit=False,target_atac_used=False,labels_used_for_fitting=False,
            frozen_state_unchanged=True,checkpoint_reload_validated=True,
            reload_reference='existing inter prediction' if method=='b2' else 'existing intra prediction',
            environment=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda),
            source_sha256={str(p):sha(p) for p in [Path(__file__),PROTOCOL]}))
        print('COMPLETE',method,source,seed,target,seconds,flush=True)
    if args.efficiency_v2:
        write(args.output/'workflows'/source/f'seed_{seed}'/'inter_efficiency.json',
            dict(protocol_id=read(PROTOCOL)['protocol_id'],protocol_sha256=sha(PROTOCOL),
                 checkpoint_sha256=digest,model_load_and_validation_seconds=load_seconds,targets=target_workflows,
                 active_wall_seconds=load_seconds+sum(t['seconds'] for t in target_workflows),
                 complete=len(target_workflows)==3,checkpoint_reused=True))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--method',choices=list(NAMES))
    parser.add_argument('--source',choices=KEYS)
    parser.add_argument('--seed',type=int,choices=range(20020130,20020135))
    parser.add_argument('--unit',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--efficiency-v2',action='store_true')
    args = parser.parse_args()
    if args.efficiency_v2: PROTOCOL=efficiency.PROTOCOL
    torch.set_num_threads(2)
    with threadpool_limits(2):
        if args.prepare: prepare_inputs()
        else:
            if not all([args.method,args.source,args.seed,args.unit,args.output]): parser.error('inference arguments required')
            with (ROOT/'.benchmark_a30.lock').open('a') as lock:
                if args.method!='palette': fcntl.flock(lock,fcntl.LOCK_EX)
                run(args)
