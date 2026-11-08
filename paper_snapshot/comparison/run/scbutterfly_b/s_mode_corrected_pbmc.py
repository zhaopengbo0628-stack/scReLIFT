"""Single-Source adapter around the frozen, audited scButterfly training runner."""
import argparse
import csv
import fcntl
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, adjusted_mutual_info_score, adjusted_rand_score,
                             balanced_accuracy_score, f1_score, normalized_mutual_info_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path('/home/u202522802045/share/zpb')
BASE = ROOT / 'data/cache/s_mode/pbmc_corrected_s10x20_v1_20261012'
OLD = ROOT / 'result_contain/34_strict_d_scbutterfly_goal2_20261002/source/run_goal2_strict_scbutterfly.py'
sys.path[:0] = [str(ROOT), str(ROOT/'main/run'), str(ROOT/'dmode_dev'),
                str(ROOT/'dmode_dev/experiments/v5d'), str(ROOT/'goal2_dependencies')]
from data.code.s_mode.prepare import legacy, reader, splits, paired_atac
from main.core.regulatory.s_mode.single_source import canonical_genes, normalize_counts

spec = importlib.util.spec_from_file_location('frozen_scb_training', OLD)
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)
SEEDS = [20020130, 20020131, 20020132]
KEYS = ['D1', 'D2', 'D3', 'D4']
EVALUATOR = dict(init='k-means++', n_init=100, random_state=20020130,
                 max_iter=1000, tol=1e-4, algorithm='lloyd')
METRICS = ['ari', 'nmi', 'ami', 'cluster_acc', 'accuracy', 'balanced_accuracy',
           'macro_f1', 'weighted_f1', 'knn_purity_at_15']
sc.CONFIG.update(strict_protocol_version='corrected_pbmc_single_source_v1',
                 run_tier='pre_final_user_declared_3seeds', mode='S',
                 feature_rule='Task45 train-only 2000 HVGs; Source-only ATAC prevalence and TF-IDF',
                 evaluator=EVALUATOR)
sc.CONFIG['default_deviations'][0] = 'strict single-Source train-only feature protocol'
sha, write, read = sc.strict.sha, sc.strict.write, sc.branches.read


def mapped_test(target, genes):
    index, cells = splits(target)
    counts, ids, path, provenance = reader.raw_rna(target)
    assert counts.shape[0] == len(cells)
    counts, ids, _ = canonical_genes(counts[index['test']], ids)
    log = normalize_counts(counts)
    lookup = {gene: i for i, gene in enumerate(ids)}
    positions = np.asarray([lookup.get(gene, -1) for gene in genes])
    hit = positions >= 0
    x = np.zeros((len(index['test']), len(genes)), np.float32)
    x[:, hit] = log[:, positions[hit]].toarray()
    return x, dict(raw_rna_path=str(path), raw_rna_sha256=sha(path),
                   test_index=index['test'].tolist(), mapped_genes=int(hit.sum()),
                   raw_rna_provenance=provenance, target_atac_used=False)


def prepare(out):
    for source in KEYS:
        cache = out/'scbutterfly_cache'/source
        if (cache/'manifest.json').exists():
            sc.strict.verify_cache(cache)
            continue
        started = time.perf_counter()
        original = legacy.checked(BASE/source/'manifest.json')
        identity = read(legacy.DATA[source]/'DATASET_IDENTITY.json')
        assert original['identity'] == identity['canonical_identity']
        index, cells = splits(source)
        for name in index:
            np.testing.assert_array_equal(index[name], np.load(BASE/source/f'{name}_index.npy'))
        assert cells == (BASE/source/'barcodes.txt').read_text().splitlines()
        genes = np.load(BASE/source/'gene_ids.npy').astype(str)
        assert len(genes) == len(set(genes)) == 2000
        rna = sparse.vstack([sparse.csr_matrix(np.load(BASE/source/f'rna_{s}.npy'))
                             for s in ['train', 'val']], format='csr')
        ntrain = len(index['train'])
        atac, peaks, dependencies = paired_atac(source, np.concatenate([index['train'], index['val']]), cells)
        keep = np.asarray(atac[:ntrain].sum(0)).ravel() >= np.ceil(.005*ntrain)
        selected = peaks[keep]
        chromosomes = np.asarray([p.split(':')[0] for p in selected])
        order = np.argsort(chromosomes, kind='stable')
        selected, chromosomes = selected[order], chromosomes[order]
        atac = atac[:, keep][:, order].tocsr()
        idf = np.log1p(ntrain/np.maximum(np.asarray(atac[:ntrain].sum(0)).ravel(), 1)).astype(np.float32)
        values = atac.multiply(1/np.maximum(np.asarray(atac.sum(1)).ravel(), 1)[:, None]).multiply(idf).tocsr()
        scale = max(float(values[:ntrain].max()), 1e-8)
        values.data = np.clip(values.data/scale, 0, 1)
        boundaries = np.r_[0, np.flatnonzero(chromosomes[1:] != chromosomes[:-1])+1, len(selected)]
        chrom_list = np.diff(boundaries).tolist()
        assert len(chrom_list) == len(set(chromosomes)) and min(chrom_list) > 0
        cache.mkdir(parents=True, exist_ok=False)
        sparse.save_npz(cache/'source_rna.npz', rna)
        sparse.save_npz(cache/'source_atac.npz', values.astype(np.float32))
        np.save(cache/'train_index.npy', np.arange(ntrain))
        np.save(cache/'val_index.npy', np.arange(ntrain, rna.shape[0]))
        np.save(cache/'peak_names.npy', selected)
        np.save(cache/'gene_ids.npy', genes)
        np.save(cache/'train_idf.npy', idf)
        query = out/'cache'/source
        query.mkdir(parents=True, exist_ok=False)
        query_info = {}
        for target in KEYS:
            x, info = mapped_test(target, genes)
            np.save(query/f'{target}_rna_test.npy', x)
            info['mapped_rna_sha256'] = sha(query/f'{target}_rna_test.npy')
            query_info[target] = info
        (query/'target_rna_test.npy').symlink_to(query/f'{source}_rna_test.npy')
        write(query/'manifest.json', query_info)
        dependencies += [BASE/source/'manifest.json', BASE/source/'gene_ids.npy',
                         BASE/source/'rna_train.npy', BASE/source/'rna_val.npy',
                         legacy.DATA[source]/'DATASET_IDENTITY.json']
        write(cache/'manifest.json', dict(source=source, identity=original['identity'],
              source_train_cells=ntrain, source_val_cells=len(index['val']), rna_genes=2000,
              atac_peaks=len(selected), chrom_list=chrom_list, source_train_max_tfidf=scale,
              preprocessing_seconds=time.perf_counter()-started, fitting_cells=index['train'].tolist(),
              test_atac_loaded=False, target_used_for_fitting=False, labels_used=False,
              RNA_rule='full-library 1e4/log1p; task45 Source train HVGs; no zscore; missing genes zero',
              ATAC_rule='Source train prevalence >=ceil(.005*N); binary TF selected-peak depth; train IDF and train max; val clip[0,1]',
              source_files_sha256={str(p): sha(p) for p in dependencies},
              cache_sha256={p.name:sha(p) for p in cache.iterdir() if p.is_file()}))
        print('CACHE_READY', source, rna.shape, values.shape, flush=True)


def load_model(cache, unit, seed):
    import anndata as ad
    from scButterfly.butterfly import Butterfly
    cm = sc.strict.verify_cache(cache)
    butterfly = Butterfly()
    butterfly.load_data(ad.AnnData(sparse.csr_matrix((1, cm['rna_genes']), dtype=np.float32)),
                        ad.AnnData(sparse.csr_matrix((1, cm['atac_peaks']), dtype=np.float32)), [], [], [])
    butterfly.RNA_data_p, butterfly.ATAC_data_p = butterfly.RNA_data, butterfly.ATAC_data
    butterfly.is_processed = True
    sc.branches.seed_all(seed)
    butterfly.construct_model(cm['chrom_list'], R_encoder_dim_list=[256,128], A_encoder_dim_list=[32,128],
                             R_decoder_dim_list=[128,256], A_decoder_dim_list=[128,32])
    best = torch.load(unit/'best_all.pt', map_location='cpu', weights_only=False)
    for name in sc.MODULES:
        getattr(butterfly.model, name).load_state_dict(best['modules'][name])
    butterfly.model.set_eval()
    return butterfly.model


def predict(model, x, path, width):
    started = time.perf_counter()
    y = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(x),width))
    with torch.no_grad():
        for offset in range(0,len(x),64):
            batch = torch.from_numpy(np.array(x[offset:offset+64],dtype=np.float32)).cuda()
            _, latent, _, _ = model.translator.test_model(model.RNA_encoder(batch), 'RNA')
            values = model.ATAC_decoder(latent)
            assert torch.isfinite(values).all()
            values[values<1e-4] = 0
            y[offset:offset+len(batch)] = values.cpu().numpy()
    y.flush()
    torch.cuda.synchronize()
    return time.perf_counter()-started


def evaluate(unit, source, target, seed, prediction, seconds, out):
    destination = unit/'evaluations'/f'{target}.json'
    if destination.exists():
        assert read(destination)['prediction_sha256'] == sha(prediction)
        return
    frozen = sha(prediction)
    label_path = legacy.DATA[target]/'processed/label.csv'
    with label_path.open(newline='') as handle:
        raw = np.asarray([row['cell_type'] for row in csv.DictReader(handle)])
    test = np.asarray(read(out/'cache'/source/'manifest.json')[target]['test_index'])
    _, labels = np.unique(raw[test], return_inverse=True)
    scaled = StandardScaler().fit_transform(np.load(prediction,mmap_mode='r'))
    clusters = KMeans(n_clusters=len(np.unique(labels)),**EVALUATOR).fit_predict(scaled)
    table = np.zeros((len(np.unique(labels)),len(np.unique(labels))),np.int64)
    np.add.at(table,(labels,clusters),1)
    a,b = linear_sum_assignment(-table)
    mapping = dict(zip(b,a))
    matched = np.asarray([mapping[c] for c in clusters])
    neighbors = NearestNeighbors(n_neighbors=16).fit(scaled).kneighbors(scaled,return_distance=False)
    neighbors = np.asarray([row[row!=i][:15] for i,row in enumerate(neighbors)])
    accuracy = float(accuracy_score(labels,matched))
    metrics = dict(source=source,target=target,seed=seed,family='intra' if source==target else 'inter',
        ari=float(adjusted_rand_score(labels,clusters)),nmi=float(normalized_mutual_info_score(labels,clusters)),
        ami=float(adjusted_mutual_info_score(labels,clusters)),cluster_acc=accuracy,accuracy=accuracy,
        balanced_accuracy=float(balanced_accuracy_score(labels,matched)),
        macro_f1=float(f1_score(labels,matched,average='macro',zero_division=0)),
        weighted_f1=float(f1_score(labels,matched,average='weighted',zero_division=0)),
        knn_purity_at_15=float(np.mean(labels[neighbors]==labels[:,None])),
        geometry_spearman=None,geometry_pearson=None,target_atac_used=False,
        annotation_scope='posthoc test Hungarian cluster scores; NOT supervised label transfer',
        evaluator=EVALUATOR,labels_loaded_after_prediction_frozen=True,prediction_sha256=frozen,
        label_file_sha256=sha(label_path),test_label_sha256=__import__('hashlib').sha256(raw[test].tobytes()).hexdigest(),
        best_checkpoint_sha256=sha(unit/'best_all.pt'),inference_seconds=seconds,cells_per_second=len(labels)/seconds)
    write(destination,metrics)
    print('METRICS',source,target,seed,metrics['ari'],metrics['nmi'],flush=True)


def unit_run(out,source,seed,smoke):
    directory = out/'smoke' if smoke else out/'models'
    sc.run_unit(directory,source,seed,smoke)
    unit = directory/source/f'seed_{seed}'
    cm = read(out/'scbutterfly_cache'/source/'manifest.json')
    model = load_model(out/'scbutterfly_cache'/source,unit,seed)
    own = np.load(out/'cache'/source/'target_rna_test.npy',mmap_mode='r')
    if smoke:
        own = own[:64]
    check = unit/'reload_check.npy'
    predict(model,own,check,cm['atac_peaks'])
    np.testing.assert_allclose(np.load(check),np.load(unit/'predicted_atac.npy'),rtol=1e-5,atol=1e-6)
    check.unlink()
    if smoke:
        write(unit/'SMOKE_GATE.json',dict(status='PASS',reload_predictions_equal=True,
              finite_forward_backward=True,labels_used=False,test_atac_used=False))
        return
    for target in KEYS:
        path = unit/f'{target}_predicted_atac.npy'
        frozen_path = unit/'evaluations'/f'{target}_frozen.json'
        if frozen_path.exists():
            frozen = read(frozen_path)
            assert frozen['prediction_sha256'] == sha(path)
        else:
            x = np.load(out/'cache'/source/f'{target}_rna_test.npy',mmap_mode='r')
            seconds = predict(model,x,path,cm['atac_peaks'])
            frozen = dict(prediction_sha256=sha(path),inference_seconds=seconds,
                          input_sha256=sha(out/'cache'/source/f'{target}_rna_test.npy'),
                          best_checkpoint_sha256=sha(unit/'best_all.pt'))
            write(frozen_path,frozen)
        evaluate(unit,source,target,seed,path,frozen['inference_seconds'],out)
    summarize(out)


def summarize(out):
    rows = [read(p) for p in sorted((out/'models').glob('*/seed_*/evaluations/D[1-4].json'))]
    summary = []
    for source in KEYS:
        for target in KEYS:
            selected = [r for r in rows if r['source']==source and r['target']==target]
            if not selected:
                continue
            row = dict(source=source,target=target,family=selected[0]['family'],n_seeds=len(selected))
            for metric in METRICS:
                values = [r[metric] for r in selected]
                row[metric+'_mean'] = float(np.mean(values))
                row[metric+'_std'] = float(np.std(values,ddof=1)) if len(values)>1 else None
            summary.append(row)
    for family in ['intra','inter']:
        folder = out/'metrics'/family
        folder.mkdir(parents=True,exist_ok=True)
        per_seed = [r for r in rows if r['family']==family]
        aggregate = [r for r in summary if r['family']==family]
        for name,records in [('per_seed',per_seed),('mean_std',aggregate)]:
            write(folder/f'{name}.json',records)
            if records:
                with (folder/f'{name}.csv').open('w',newline='') as handle:
                    writer = csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        lines = ['3-seed pre_final; posthoc Hungarian scores; RNA-only frozen inference.', '',
                 '| Source | Target | Seeds | '+' | '.join(METRICS)+' |',
                 '|---|---|---|'+'---|'*len(METRICS)]
        for r in aggregate:
            values = [f"{r[m+'_mean']:.6f} ± {r[m+'_std']:.6f}" if r[m+'_std'] is not None else f"{r[m+'_mean']:.6f}" for m in METRICS]
            lines.append(f"| {r['source']} | {r['target']} | {r['n_seeds']} | "+' | '.join(values)+' |')
        (folder/'summary.md').write_text('\n'.join(lines)+'\n')
    efficiency = []
    for p in sorted((out/'models').glob('*/seed_*/manifest.json')):
        m = read(p); r = dict(m['resources'])
        assert sha(p.parent/'best_all.pt') == m['best_checkpoint_sha256']
        history = [x for x in read(p.parent/'history.json') if x['stage']=='all']
        chosen = next(x for x in history if x['epoch']==m['best_epoch'])
        assert chosen['source_validation_reconstruction'] == min(x['source_validation_reconstruction'] for x in history)
        evaluations = [x for x in rows if x['source']==m['fold'] and x['seed']==m['seed']]
        cm = read(out/'scbutterfly_cache'/m['fold']/'manifest.json')
        r.update(all_query_inference_seconds=sum(x['inference_seconds'] for x in evaluations),
                 preprocessing_seconds=cm['preprocessing_seconds'],
                 best_source_validation_reconstruction=chosen['source_validation_reconstruction'])
        r['total_wall_seconds'] = r['training_seconds']+r['all_query_inference_seconds']+r['preprocessing_seconds']
        efficiency.append(dict(source=m['fold'],seed=m['seed'],best_epoch=m['best_epoch'],
                          best_checkpoint_sha256=m['best_checkpoint_sha256'],**r))
    write(out/'metrics/efficiency.json',efficiency)
    if efficiency:
        with (out/'metrics/efficiency.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(efficiency[0]));writer.writeheader();writer.writerows(efficiency)
        (out/'metrics/efficiency.md').write_text('| '+' | '.join(efficiency[0])+' |\n|'+ '---|'*len(efficiency[0])+'\n'+
                    '\n'.join('| '+' | '.join(str(v) for v in r.values())+' |' for r in efficiency)+'\n')
    expected = {(source,target,seed) for source in KEYS for target in KEYS for seed in SEEDS}
    observed = {(r['source'],r['target'],r['seed']) for r in rows}
    complete = observed==expected and len(rows)==48 and len(efficiency)==12
    if complete:
        for r in rows:
            prediction = out/'models'/r['source']/f"seed_{r['seed']}"/f"{r['target']}_predicted_atac.npy"
            assert sha(prediction)==r['prediction_sha256']
            assert all(np.isfinite(r[metric]) for metric in METRICS)
    write(out/'metrics/status.json',dict(status='COMPLETE' if complete else 'PARTIAL',
          trained_models=len(efficiency),evaluations=len(rows),expected_models=12,expected_evaluations=48,
          run_tier='pre_final',seeds=SEEDS))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','smoke','unit','summary'],required=True)
    parser.add_argument('--source',choices=KEYS)
    parser.add_argument('--seed',type=int,choices=SEEDS,default=SEEDS[0])
    args=parser.parse_args()
    torch.set_num_threads(2)
    np.Inf,np.NaN=np.inf,np.nan
    sc.PREP,sc.GOAL1=args.out,args.out
    if args.stage=='prepare':
        prepare(args.out)
    elif args.stage=='summary':
        summarize(args.out)
    else:
        assert args.source
        if args.stage=='unit':
            assert read(args.out/'smoke'/args.source/f'seed_{SEEDS[0]}'/'SMOKE_GATE.json')['status']=='PASS'
        with (ROOT/'.benchmark_a30.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            with sc.branches.gpu_lock():
                unit_run(args.out,args.source,args.seed,args.stage=='smoke')
