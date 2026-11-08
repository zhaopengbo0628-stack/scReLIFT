"""Corrected source-specific Teacher control using frozen Full B2 stages."""
import argparse
import copy
import csv
import fcntl
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, '/home/u202522802045/share/zpb')
import run_updated_b2_validated as shared
import numpy as np
import torch
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from multi_teacher import select_best_single_teacher, validate_teacher_matrix

b2, prep = shared.b2, shared.prep
ROOT = prep.ROOT / 'b2_multi_teacher_ablation'
OUT = ROOT / 'results'
FOLDS = [f'target_D{i}' for i in range(1, 5)]
BASE_BATCHES = b2.batches


def batches(sources, rng):
    if len(sources) == 3:
        yield from BASE_BATCHES(sources, rng)
    else:
        assert len(sources) == 1
        source = sources[0]
        for _ in range(math.ceil(len(source['train']['rna']) / b2.CONFIG['batch_size'])):
            yield [(source, rng.integers(len(source['train']['rna']), size=b2.CONFIG['batch_size']))]


b2.batches = batches


def verify(unit):
    value = b2.read(unit / 'manifest.json')
    assert value['status'] == 'COMPLETE'
    for name, digest in value['artifact_hashes'].items():
        assert prep.sha(unit / name) == digest, name
    return value


def validate_teacher(model, sources, scales, device):
    rows = {}
    with torch.no_grad():
        for source in sources:
            values, counts = [], []
            for start in range(0, len(source['val']['rna']), 512):
                r = torch.from_numpy(source['val']['rna'][start:start+512]).to(device)
                f = torch.from_numpy(source['val']['field'][start:start+512]).to(device)
                lr, lf, lp = b2.teacher_losses(model, r, f)
                values.append([lr.item(), lf.item(), lp.item(), ((lr+lf)/scales[0]+lp/scales[1]).item()])
                counts.append(len(r))
            rows[source['name']] = dict(zip(['RNA_reconstruction', 'F_reconstruction', 'alignment', 'total'],
                                            np.average(values, axis=0, weights=counts).tolist()))
    return rows


def save_stage_accounting(directory, source_cells, steps_per_epoch):
    checkpoint = torch.load(directory / 'latest.pt', map_location='cpu', weights_only=False)
    b2.write(directory / 'training_accounting.json', dict(training_cells=source_cells,
             epochs=checkpoint['epoch'], optimizer_steps=checkpoint['epoch']*steps_per_epoch,
             batch_size=256, initial_scales=checkpoint['initial_scales'],
             best_sha256=prep.sha(directory/'best.pt'), final_sha256=prep.sha(directory/'latest.pt'),
             optimizer_and_rng_saved=True, scheduler='NONE'))


def run(fold, seed):
    assert Path('/proc/self/fd/9').resolve() == prep.ROOT / '.benchmark_a30.lock'
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
    unit = OUT / fold / f'seed_{seed}'
    if (unit/'manifest.json').exists():
        verify(unit)
        return
    sources, mean, std, sm = shared.load_sources(fold)
    inputs = dict(sm['hashes'])
    inputs.update({str(p): prep.sha(p) for p in [shared.REFERENCE, Path(__file__), ROOT/'multi_teacher.py',
                                              Path(shared.__file__), Path(prep.__file__)]})
    gate = b2.read(ROOT/'PREFLIGHT.json')
    assert gate['status'] == 'PASS' and gate['runner_sha256'] == prep.sha(__file__)
    names = tuple(s['name'] for s in sources)
    device = torch.device('cuda')
    unit.mkdir(parents=True, exist_ok=True)
    b2.write(unit/'config.json', dict(**b2.CONFIG, fold=fold, seed=seed,
             teacher_source_order=dict(zip('ABC', names)), teacher_single_source_batch_size=256,
             run_scope='pre_final_explicit_3seeds', teacher_gate_policy='diagnostic_only_for_explicit_single_source_ablation',
             reference_gate_failure_not_bypassed_for_shared_control=True))
    b2.write(unit/'data_manifest.json', sm)
    teachers, matrix, details = {}, {}, []
    for letter, source in zip('ABC', sources):
        directory = unit / f'T_{letter}'
        b2.seed_all(seed)
        teacher = b2.Teacher(len(mean), len(mean)).to(device)
        teacher = b2.train_stage(teacher, [source], directory, seed, device, True, inputs)
        save_stage_accounting(directory, len(source['train']['rna']), math.ceil(len(source['train']['rna'])/256))
        best = torch.load(directory/'best.pt', map_location='cpu', weights_only=False)
        decomposed = validate_teacher(teacher, sources, best['initial_scales'], device)
        matrix[source['name']] = {n: decomposed[n]['total'] for n in names}
        b2.write(directory/'cross_source_validation.json', decomposed)
        # This intervention explicitly trains out-of-domain Teachers; the shared
        # consolidation acceptance gate is recorded, not imposed on candidates.
        b2.write(directory/'teacher_diagnostic.json', b2.teacher_audit(teacher, sources, device))
        teachers[letter] = directory
        details.extend(dict(teacher=source['name'], source=n, **decomposed[n]) for n in names)
        del teacher
        torch.cuda.empty_cache()
    validate_teacher_matrix(matrix, names)
    selection = select_best_single_teacher(matrix)
    selected_letter = 'ABC'[names.index(selection.selected_source)]
    b2.write(unit/'teacher_selection.json', dict(selected_source=selection.selected_source,
             selected_student=f'S_{selected_letter}', scores=selection.equal_weight_validation_loss,
             criterion='equal-source mean exact normalized Teacher validation objective',
             scale_scope='each candidate frozen initial training scales; raw decompositions also saved',
             target_used=False, fourth_student_trained=False))
    with (unit/'teacher_cross_source_validation.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader(); writer.writerows(details)
    metrics = {}
    for letter in 'ABC':
        directory = unit/f'S_{letter}'
        directory.mkdir(exist_ok=True)
        emb = directory/'embeddings'
        emb.mkdir(exist_ok=True)
        teacher = b2.Teacher(len(mean), len(mean)).to(device)
        teacher.load_state_dict(torch.load(teachers[letter]/'best.pt', map_location=device, weights_only=False)['model'])
        teacher.eval().requires_grad_(False)
        for split in ['train', 'val']:
            for source in sources:
                zr, zf, zt, _, _ = b2.forward_numpy(teacher, source[split]['rna'], device, source[split]['field'])
                source[split]['teacher_latent'] = zt
                for name, z in [('z_R',zr), ('z_F',zf), ('z_T',zt)]:
                    np.save(emb/f"teacher_{source['key']}_{split}_{name}.npy", z)
        mu = np.mean([s['train']['teacher_latent'].mean(0) for s in sources], axis=0)
        sigma = np.sqrt(np.mean([s['train']['teacher_latent'].var(0)+(s['train']['teacher_latent'].mean(0)-mu)**2
                                for s in sources], axis=0)).astype(np.float32)
        for source in sources:
            source.update(latent_mean=mu, latent_std=sigma, raw_zero_standardized=-mean/(std+1e-6))
        np.savez_compressed(directory/'preprocessing.npz', RNA_mean=mean, RNA_std=std, KD_mean=mu, KD_std=sigma)
        student_inputs = dict(inputs)
        student_inputs[str(teachers[letter]/'best.pt')] = prep.sha(teachers[letter]/'best.pt')
        student_inputs[str(directory/'preprocessing.npz')] = prep.sha(directory/'preprocessing.npz')
        b2.seed_all(seed)
        student = copy.deepcopy(teacher.rna_encoder).requires_grad_(True).to(device)
        student = b2.train_stage(student, sources, directory, seed, device, False, student_inputs)
        save_stage_accounting(directory, sum(len(s['train']['rna']) for s in sources),
                              math.ceil(sum(len(s['train']['rna']) for s in sources)/256))
        for split in ['train','val']:
            np.save(emb/f'source_{split}_embedding_raw.npy',
                    np.concatenate([b2.forward_numpy(student,s[split]['rna'],device) for s in sources]))
            shared.cell_order(sources,split,emb/f'source_{split}_cell_order.tsv')
        cache = shared.target_cache(fold)
        cache_dir = shared.OUT/'validated_cache'/fold
        torch.cuda.synchronize(); started=time.perf_counter()
        raw = np.load(cache_dir/'target_rna_test.npy')
        z = b2.forward_numpy(student,((raw-mean)/(std+1e-6)).astype(np.float32),device)
        torch.cuda.synchronize(); inference=time.perf_counter()-started
        path=emb/'target_embedding_raw.npy'
        np.save(path,z); frozen=prep.sha(path)
        b2.write(directory/'embedding_manifest.json',dict(sha256=frozen,frozen_before_labels=True,cache=cache))
        (emb/'target_cell_order.tsv').write_bytes((cache_dir/'cell_order.tsv').read_bytes())
        # Target labels are read only after this Student's embedding is saved/hashed.
        with (prep.DATA[fold.split('_')[1]]/'processed/label_harmonized.csv').open() as stream:
            labels=np.asarray([r['cell_type'] for r in csv.DictReader(stream)])
        labels=np.unique(labels[np.load(cache_dir/'target_test_index.npy')],return_inverse=True)[1]
        assert prep.sha(path)==frozen
        scaled=StandardScaler().fit_transform(z)
        np.save(emb/'target_embedding_standardized.npy',scaled)
        score=b2.evaluate(z,labels)
        neighbors=NearestNeighbors(n_neighbors=15).fit(scaled).kneighbors(return_distance=False)
        score.update(knn_purity_at_15=float((labels[neighbors]==labels[:,None]).mean()),
                     cell_type_silhouette=float(silhouette_score(scaled,labels)))
        b2.write(directory/'metrics.json',score)
        tm=b2.read(teachers[letter]/'manifest.json')['resources']; st=b2.read(directory/'manifest.json')['resources']
        b2.write(directory/'efficiency.json',dict(teacher=tm,student=st,TOTAL_PARAMS=tm['parameters']+st['parameters'],
                 target_inference_seconds=inference, deployment_path_training_seconds=tm['training_seconds']+st['training_seconds'],
                 peak_GPU_MB=max(tm['peak_allocated_bytes'],st['peak_allocated_bytes'])/2**20))
        b2.write(directory/'leakage_audit.json',dict(target_atac_read=False,target_labels_after_embedding=True,
                 target_fitted_preprocessing=False,target_selection=False,source_only_teacher_selection=True))
        metrics[letter]=score
        del teacher,student
        torch.cuda.empty_cache()
    b2.write(unit/'metrics.json',dict(candidates=metrics,best_single=metrics[selected_letter],selected=selected_letter))
    b2.write(unit/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,
             GPU=torch.cuda.get_device_name(),VRAM_bytes=torch.cuda.get_device_properties(0).total_memory,
             precision='float32; TF32 disabled'))
    b2.write(unit/'manifest.json',dict(status='COMPLETE',fold=fold,seed=seed,
             artifact_hashes={str(p.relative_to(unit)):prep.sha(p) for p in unit.rglob('*')
                             if p.is_file() and p.name!='run.log'}))
    print('UNIT_COMPLETE',fold,seed,selected_letter,flush=True)


def preflight():
    audit=b2.read(prep.ROOT/'UPDATED_FOUR_DATASET_INDEPENDENCE_AUDIT.json')
    assert audit['FOUR_DATASET_INDEPENDENCE']=='PASS'
    for d in audit['datasets'].values():assert prep.sha(d['rna_path'])==d['rna_sha256']
    rows=[]
    for fold in FOLDS:
        sources,mean,std,sm=shared.load_sources(fold)
        shared.target_cache(fold)
        b2.seed_all(20020130)
        model=b2.Teacher(len(mean),len(mean))
        x=torch.from_numpy(sources[0]['train']['rna'][:8]);f=torch.from_numpy(sources[0]['train']['field'][:8])
        loss=sum(b2.teacher_losses(model,x,f));loss.backward()
        assert torch.isfinite(loss)
        student=copy.deepcopy(model.rna_encoder).eval()
        assert torch.equal(student(x),model.rna_encoder.eval()(x))
        for count in [1,3]:
            batch=next(batches(sources[:count],np.random.default_rng(0)))
            assert sum(len(i) for _,i in batch)==256
        rows.append(dict(fold=fold,sources=sm['sources'],genes=len(mean)))
    b2.write(ROOT/'PREFLIGHT.json',dict(status='PASS',rows=rows,runner_sha256=prep.sha(__file__),
             frozen_reference_sha256=prep.sha(shared.REFERENCE),source_only=True,optimizer_step=False,
             shared_reference_status='SCIENTIFIC_BLOCKER; not required for this independent intervention',
             run_tier='pre_final_explicit_3seeds'))
    print('PREFLIGHT_PASS',flush=True)


def summarize():
    rows=[];selection=[]
    keys=['ari','nmi','cluster_acc','macro_f1','balanced_accuracy','knn_purity_at_15','cell_type_silhouette']
    for fold in FOLDS:
        for seed in shared.SEEDS:
            unit=OUT/fold/f'seed_{seed}';verify(unit)
            value=b2.read(unit/'metrics.json');selection.append(dict(fold=fold,seed=seed,**b2.read(unit/'teacher_selection.json')))
            for name in ['A','B','C','BEST']:
                score=value['best_single'] if name=='BEST' else value['candidates'][name]
                rows.append(dict(fold=fold,seed=seed,strategy=name,**{k:score[k] for k in keys}))
    overall={}
    for name in ['A','B','C','BEST']:
        means=[{k:float(np.mean([r[k] for r in rows if r['seed']==s and r['strategy']==name])) for k in keys} for s in shared.SEEDS]
        overall[name]={k:dict(mean=float(np.mean([m[k] for m in means])),std=float(np.std([m[k] for m in means],ddof=1))) for k in keys}
    result=dict(status='COMPLETE',rows=rows,overall=overall,selection=selection,
                selection_counts={c:sum(v['selected_student']==f'S_{c}' for v in selection) for c in 'ABC'},
                shared_full_b2='UNAVAILABLE_AS_COMPLETE_MATCHED_CONTROL',delta_vs_shared=None,
                std_scope='three seed-level four-fold means; sample std',run_tier='pre_final_explicit_3seeds',
                checkpoints=len(list(OUT.rglob('*.pt'))),target_embeddings=len(list(OUT.rglob('target_embedding_raw.npy'))),
                UMAP_READY=True,interpretation='source-specific versus multi-source consolidation; not coordinate alignment alone')
    b2.write(ROOT/'MULTI_TEACHER_3SEED_REPORT.json',result)
    with (ROOT/'MULTI_TEACHER_PER_RUN.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Multi-Teacher Ablation','', 'Explicit three-seed pre_final; not final 15-seed evidence.', '',
           '| Strategy | ARI mean +/- std | NMI mean +/- std |','| --- | --- | --- |']
    for name,v in overall.items():lines.append(f"| {name} | {v['ari']['mean']:.6f} +/- {v['ari']['std']:.6f} | {v['nmi']['mean']:.6f} +/- {v['nmi']['std']:.6f} |")
    lines += ['','Shared Full B2 matched complete control unavailable; no old-D2 replacement and no comparative claim.',
              'Single-source Teachers see fewer cells/steps. This does not isolate latent-coordinate alignment.']
    (ROOT/'MULTI_TEACHER_3SEED_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('ABLATION_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true');p.add_argument('--summarize',action='store_true')
    p.add_argument('--fold',choices=FOLDS);p.add_argument('--seed',type=int,choices=shared.SEEDS);a=p.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if a.dry_run:preflight()
        elif a.summarize:summarize()
        else:
            assert a.fold and a.seed
            run(a.fold,a.seed)
