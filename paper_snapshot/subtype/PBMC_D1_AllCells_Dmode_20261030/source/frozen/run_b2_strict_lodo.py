"""Gated source-only shared regulatory Teacher and RNA-only dense KD."""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, adjusted_mutual_info_score, adjusted_rand_score,
                             balanced_accuracy_score, f1_score, normalized_mutual_info_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn import functional as F

ROOT = Path('/home/u202522802045/share/zpb')
GOAL1 = ROOT / 'result_contain/strict_lodo_two_step_20261002'
FOLDS = [f'target_D{i}' for i in range(1, 5)]
SEEDS = [20020130, 20020131, 20020132]
EVALUATOR = dict(init='k-means++', n_init=100, random_state=20020130,
                 max_iter=1000, tol=1e-4, algorithm='lloyd')
CONFIG = dict(method='B2', run_tier='pre_final_user_declared_gated_seeds', epochs=100,
              patience=15, batch_size=256, clip=1.0, teacher_lr=1e-3, student_lr=1e-4,
              weight_decay=1e-4, mask_rate=.15, eps=1e-6, evaluator=EVALUATOR,
              architecture='input-512-LN-GELU-dropout0.1-64-LN; mean fusion; 64-512-GELU-input decoders',
              teacher_objectives=['RNA+F reconstruction', 'same-cell MSE'],
              initial_scale_batches=8, student_objective='source-train standardized clean Teacher MSE',
              student_encoder_includes_final_layernorm=True, balanced_batch_counts=[86, 85, 85],
              validation='equal dataset means', strong_mean_ari=.55, promising_delta_ari=.03,
              nmi_mean_regression_limit=.01, nmi_fold_regression_limit=.03,
              teacher_gate=dict(min_rank=8., min_active_variance_fraction=.8, variance_threshold=1e-4,
                                max_same_shuffled_ratio=.9, min_probe_mean_r2=0.,
                                min_F_R_norm_ratio=.1, min_fused_R_difference=.05),
              source_gate_scope='every source validation dataset in all 4 outer folds',
              cohort_caveat='inherited paired/QC cohort and label-stratified splits',
              target_atac_used=False, target_adaptation=False,
              EVALUATION_TIME_UNSUPERVISED_SCALING=True, KNOWN_K_EVALUATION=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False


def save_array(path, array):
    array = np.asarray(array, dtype=np.float32)
    assert np.isfinite(array).all()
    np.save(path, array)
    return sha(path)


def save_checkpoint(path, checkpoint):
    temporary = path.with_name(path.name+'.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def registry_update(out, status, seeds):
    path = ROOT / 'result_contain/task_registry.csv'
    with path.open('r+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        reader = csv.DictReader(handle)
        columns, rows = reader.fieldnames, list(reader)
        matches = [r for r in rows if r['task_id'] == '35']
        if not matches:
            used = {int(r['task_id']) for r in rows}
            for directory in [ROOT / 'slum', ROOT / 'result_contain']:
                for p in directory.iterdir():
                    m = re.match(r'(?:task)?(\d+)(?:_|\b)', p.name, re.I)
                    if m and p.resolve() != out.resolve():
                        used.add(int(m.group(1)))
            assert 35 not in used and max(used) == 34, sorted(used)
            row = dict.fromkeys(columns, '')
            row.update(task_id='35', experiment_name=out.name, model_version='B2-shared-regulatory-Teacher-dense-KD',
                       result_path=str(out), created_at=now(), datasets='D1,D2,D3,D4',
                       git_commit=read(out / 'git_info.json')['commit'])
            rows.append(row)
        else:
            assert len(matches) == 1 and matches[0]['result_path'] == str(out)
            row = matches[0]
        row.update(status=status, updated_at=now(), seeds=','.join(map(str, seeds)), pid=str(os.getpid()))
        if status.startswith('COMPLETED'):
            row['completed_at'] = now()
        handle.seek(0)
        handle.truncate()
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def prepare(out):
    if (out / 'manifest.json').exists():
        assert read(out / 'config.json') == CONFIG
        return
    completion = read(GOAL1 / 'COMPLETION_MANIFEST.json')
    assert completion['status'] == 'COMPLETE'
    for p, digest in completion['source_files_sha256'].items():
        assert sha(p) == digest, p
    out.mkdir(parents=True, exist_ok=True)
    write(out / 'config.json', CONFIG)
    write(out / 'data_manifest.json', dict(source_files_sha256=completion['source_files_sha256'],
          inherited_strict_cache=str(GOAL1 / 'cache'), B_D_F_rebuilt=False, target_atac_used=False))
    write(out / 'manifest.json', dict(status='PREPARED', task_id=35, created_at=now(),
          config_sha256=sha(out / 'config.json'), runner_sha256=sha(__file__)))
    environment = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
          numpy=np.__version__, gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name',
          '--format=csv,noheader'], text=True).strip())
    (out / 'environment.txt').write_text(json.dumps(environment, indent=2) + '\n')
    registry_update(out, 'PREPARED', [SEEDS[0]])


def labels_after_frozen(fold):
    cm = read(GOAL1 / 'cache' / fold / 'manifest.json')
    raw = Path(cm['raw_target_source'])
    processed = raw.parent if raw.suffix == '.npz' else raw.parent.parent / 'processed'
    index = np.load(GOAL1 / 'cache' / fold / 'target_test_index.npy')
    with (processed / 'label.csv').open() as handle:
        labels = np.asarray([r['cell_type'] for r in csv.DictReader(handle)])[index]
    return np.unique(labels, return_inverse=True)[1]


def evaluate(x, labels):
    scaled = StandardScaler().fit_transform(x)
    km = KMeans(n_clusters=len(np.unique(labels)), **EVALUATOR).fit(scaled)
    clusters = km.labels_
    table = np.zeros((len(np.unique(labels)), len(np.unique(labels))), np.int64)
    np.add.at(table, (labels, clusters), 1)
    a, b = linear_sum_assignment(-table)
    mapping = dict(zip(b, a))
    matched = np.asarray([mapping[c] for c in clusters])
    return dict(ari=float(adjusted_rand_score(labels, clusters)), nmi=float(normalized_mutual_info_score(labels, clusters)),
          ami=float(adjusted_mutual_info_score(labels, clusters)), cluster_acc=float(accuracy_score(labels, matched)),
          accuracy=float(accuracy_score(labels, matched)), balanced_accuracy=float(balanced_accuracy_score(labels, matched)),
          macro_f1=float(f1_score(labels, matched, average='macro', zero_division=0)),
          weighted_f1=float(f1_score(labels, matched, average='weighted', zero_division=0)),
          annotation_scope='posthoc Hungarian-mapped clusters, NOT supervised label transfer',
          inertia=float(km.inertia_), iterations=int(km.n_iter_), evaluator=EVALUATOR,
          labels_loaded_after_embedding_frozen=True)


def canonical(out):
    rows = []
    conditions = [('RNA', GOAL1 / 'rna', 'z_target_test.npy', SEEDS),
          ('scButterfly', ROOT / 'result_contain/34_strict_d_scbutterfly_goal2_20261002', 'predicted_atac.npy', SEEDS),
          ('D_parent', GOAL1 / 'parent', 'z_target_test.npy', SEEDS[:1]),
          ('capacity', ROOT / 'result_contain/33_strict_d_capacity_goal2_20261002', 'z_target_test.npy', SEEDS[:1])]
    for method, base, filename, seeds in conditions:
        for fold in FOLDS:
            for seed in seeds:
                unit = base / fold / f'seed_{seed}'
                destination = out / 'canonical' / method / fold / f'seed_{seed}.json'
                manifest, digest = read(unit / 'manifest.json'), sha(unit / filename)
                assert digest == manifest.get('embedding_sha256', manifest.get('prediction_sha256'))
                if destination.exists():
                    row = read(destination)
                    assert row['input_sha256'] == digest and row['evaluator'] == EVALUATOR
                else:
                    started = time.perf_counter()
                    x = np.load(unit / filename)
                    assert np.isfinite(x).all()
                    y = labels_after_frozen(fold)
                    row = dict(method=method, fold=fold, training_seed=seed, input=str(unit / filename),
                               input_sha256=digest, **evaluate(x, y), seconds=time.perf_counter() - started)
                    assert sha(unit / filename) == digest
                    write(destination, row)
                rows.append(row)
                print('CANONICAL', method, fold, seed, row['ari'], row['nmi'], flush=True)
    summary = {}
    for method in ['RNA', 'scButterfly', 'D_parent', 'capacity']:
        selected = [r for r in rows if r['method'] == method]
        per_seed = []
        for seed in sorted({r['training_seed'] for r in selected}):
            units = [r for r in selected if r['training_seed'] == seed]
            per_seed.append(dict(seed=seed, **{m: float(np.mean([r[m] for r in units])) for m in ['ari','nmi']}))
        summary[method] = dict(per_seed=per_seed, mean={m:float(np.mean([r[m] for r in per_seed])) for m in ['ari','nmi']},
          std={m:float(np.std([r[m] for r in per_seed], ddof=1)) if len(per_seed)>1 else None for m in ['ari','nmi']})
    write(out / 'CANONICAL_BASELINES.json', dict(status='COMPLETE', rows=rows, summary=summary,
          std_scope='sample std across seed-specific four-fold means', evaluator=EVALUATOR))
    lines = ['# Canonical Frozen Baselines', '', '| Method | Seeds | Mean ARI | Std ARI | Mean NMI | Std NMI |',
             '| --- | --- | --- | --- | --- | --- |']
    for m, v in summary.items():
        lines.append(f"| {m} | {len(v['per_seed'])} | {v['mean']['ari']:.6f} | {v['std']['ari']} | {v['mean']['nmi']:.6f} | {v['std']['nmi']} |")
    (out / 'CANONICAL_BASELINES.md').write_text('\n'.join(lines) + '\n')


def load_sources(fold, out):
    cm = read(GOAL1 / 'cache' / fold / 'manifest.json')
    hroot = Path(cm['source_harmonized_root'])
    froot = hroot.parent
    p = GOAL1 / 'rna' / fold / 'seed_20020130/shared_rna_best.pt'
    saved = torch.load(p, map_location='cpu', weights_only=False)
    mean, std = saved['rna_mean'], saved['rna_std']
    expected = read(out / 'data_manifest.json')['source_files_sha256']
    sources, inputs = [], {str(p):sha(p)}
    for name in cm['sources']:
        directory = hroot / name
        with np.load(directory / 'metadata.npz', allow_pickle=True) as metadata:
            train, val = metadata['train_index'], metadata['val_index']
        assert not np.intersect1d(train, val).size
        rna = np.load(directory / 'rna.npy', mmap_mode='r')
        source = dict(name=name)
        for split, index in [('train', train), ('val', val)]:
            p = froot / f'{name}_F_{split}.npy'
            assert sha(p) == expected[str(p)]
            field = np.load(p).astype(np.float32)
            assert len(field) == len(index) and np.isfinite(field).all()
            raw = np.asarray(rna[index], np.float32)
            source[split] = dict(rna=((raw-mean)/(std+CONFIG['eps'])).astype(np.float32),
                                 raw_nonzero=raw != 0, field=field)
            inputs[str(p)] = expected[str(p)]
        for p in [directory / 'rna.npy', directory / 'metadata.npz']:
            inputs[str(p)] = sha(p)
            if str(p) in expected:
                assert inputs[str(p)] == expected[str(p)]
        sources.append(source)
    assert cm['target'] not in [s['name'] for s in sources]
    for p in [froot / 'B_D.npz', froot / 'fold_manifest.json', froot / 'gene_ids.npy']:
        assert sha(p) == expected[str(p)]
        inputs[str(p)] = expected[str(p)]
    return sources, mean, std, inputs


class Encoder(nn.Sequential):
    def __init__(self, width):
        super().__init__(nn.Linear(width, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(.1),
                         nn.Linear(512, 64), nn.LayerNorm(64))


class Teacher(nn.Module):
    def __init__(self, rdim, fdim):
        super().__init__()
        self.rna_encoder, self.field_encoder = Encoder(rdim), Encoder(fdim)
        self.rna_decoder = nn.Sequential(nn.Linear(64, 512), nn.GELU(), nn.Linear(512, rdim))
        self.field_decoder = nn.Sequential(nn.Linear(64, 512), nn.GELU(), nn.Linear(512, fdim))

    def forward(self, rna, field):
        zr, zf = self.rna_encoder(rna), self.field_encoder(field)
        zt = .5 * (zr + zf)
        return zr, zf, zt, self.rna_decoder(zt), self.field_decoder(zt)


def batches(sources, rng):
    steps = math.ceil(sum(len(s['train']['rna']) for s in sources) / CONFIG['batch_size'])
    for _ in range(steps):
        yield [(s, rng.integers(len(s['train']['rna']), size=count))
               for s, count in zip(sources, CONFIG['balanced_batch_counts'])]


def batch_tensor(batch, key, device):
    return torch.from_numpy(np.concatenate([s['train'][key][i] for s, i in batch])).to(device)


def teacher_losses(model, rna, field):
    zr, zf, _, rr, ff = model(rna, field)
    return F.mse_loss(rr,rna), F.mse_loss(ff,field), F.mse_loss(zr,zf)


@torch.no_grad()
def forward_numpy(model, values, device, field=None):
    model.eval()
    chunks = []
    for start in range(0, len(values), 512):
        x = torch.from_numpy(values[start:start+512]).to(device)
        output = model(x) if field is None else model(x, torch.from_numpy(field[start:start+512]).to(device))
        chunks.append(output.cpu().numpy() if field is None else [v.cpu().numpy() for v in output])
    return np.concatenate(chunks) if field is None else [np.concatenate([c[i] for c in chunks]) for i in range(5)]


def rank_info(z):
    z = np.asarray(z,np.float64)
    var = z.var(0)
    eig = np.maximum(np.linalg.eigvalsh(np.cov(z,rowvar=False,ddof=0)),0)
    p = eig / max(eig.sum(),1e-15)
    nz = p[p>0]
    return dict(effective_rank=float(np.exp(-(nz*np.log(nz)).sum())),variance=var.tolist(),
                active_variance_fraction=float(np.mean(var>CONFIG['teacher_gate']['variance_threshold'])))


def probe(train_x, train_y, val_x, val_y):
    scaler = StandardScaler().fit(train_x)
    model = Ridge(alpha=1.).fit(scaler.transform(train_x),train_y)
    pred = model.predict(scaler.transform(val_x))
    err, variance = np.mean((pred-val_y)**2,0),np.var(val_y,0)
    active = variance>1e-8
    r2 = 1-err[active]/variance[active]
    return dict(mean_r2=float(r2.mean()),median_r2=float(np.median(r2)),mse=float(err.mean()),
                dimensions=int(active.sum()),positive_r2_fraction=float(np.mean(r2>0)))


def teacher_audit(model, sources, device):
    rng = np.random.default_rng(20020130)
    rows,tr_all,va_all,domain_tr,domain_va = [],[],[],[],[]
    for domain,s in enumerate(sources):
        tr = forward_numpy(model,s['train']['rna'],device,s['train']['field'])
        va = forward_numpy(model,s['val']['rna'],device,s['val']['field'])
        zr,zf,zt,rr,ff = va
        # A cyclic offset gives a reproducible derangement within each source.
        permutation = (np.arange(len(zf))+rng.integers(1,len(zf))) % len(zf)
        same = np.mean((zr-zf)**2)
        shuffled = np.mean((zr-zf[permutation])**2)
        distances = (np.sum(zr*zr,1)[:,None]+np.sum(zf*zf,1)[None,:]-2*zr@zf.T)/zr.shape[1]
        nearest = np.argmin(distances,axis=1)
        rank = 1+(distances<np.diag(distances)[:,None]).sum(1)
        shuffled_rank = 1+(distances<distances[np.arange(len(zr)),permutation][:,None]).sum(1)
        row = dict(source=s['name'],z_R=rank_info(zr),z_F=rank_info(zf),z_T=rank_info(zt),
              same_cell_mse=float(same),shuffled_cell_mse=float(shuffled),same_shuffled_ratio=float(same/max(shuffled,1e-12)),
              retrieval_at_1=float(np.mean(nearest==np.arange(len(zr)))),shuffled_retrieval_at_1=float(np.mean(nearest==permutation)),
              retrieval_mrr=float(np.mean(1/rank)),shuffled_retrieval_mrr=float(np.mean(1/shuffled_rank)),
              RNA_reconstruction_mse=float(np.mean((rr-s['val']['rna'])**2)),F_reconstruction_mse=float(np.mean((ff-s['val']['field'])**2)),
              F_R_norm_ratio=float(np.mean(np.linalg.norm(zf,axis=1))/max(np.mean(np.linalg.norm(zr,axis=1)),1e-12)),
              fused_R_relative_difference=float(np.mean(np.linalg.norm(zt-zr,axis=1))/max(np.mean(np.linalg.norm(zr,axis=1)),1e-12)),probes={})
        for name,a,b,y in [('zT_to_RNA',tr[2],zt,'rna'),('zT_to_F',tr[2],zt,'field'),
                           ('zR_to_F',tr[0],zr,'field'),('zF_to_RNA',tr[1],zf,'rna')]:
            row['probes'][name] = probe(a,s['train'][y],b,s['val'][y])
        row['fused_F_probe_gain_vs_RNA_branch'] = row['probes']['zT_to_F']['mean_r2']-row['probes']['zR_to_F']['mean_r2']
        gate = CONFIG['teacher_gate']
        checks = dict(noncollapsed=all(row[k]['effective_rank']>=gate['min_rank'] and
                      row[k]['active_variance_fraction']>=gate['min_active_variance_fraction'] for k in ['z_R','z_F','z_T']),
                      paired_alignment=row['same_shuffled_ratio']<gate['max_same_shuffled_ratio'] and row['retrieval_mrr']>row['shuffled_retrieval_mrr'],
                      RNA_preserved=row['probes']['zT_to_RNA']['mean_r2']>gate['min_probe_mean_r2'],
                      F_preserved=row['probes']['zT_to_F']['mean_r2']>gate['min_probe_mean_r2'],
                      F_not_negligible=row['F_R_norm_ratio']>=gate['min_F_R_norm_ratio'] and row['fused_R_relative_difference']>=gate['min_fused_R_difference'])
        row.update(checks=checks,passed=all(checks.values()))
        rows.append(row);tr_all.append(tr[2]);va_all.append(zt)
        domain_tr.extend([domain]*len(tr[2]));domain_va.extend([domain]*len(zt))
    scaler = StandardScaler().fit(np.concatenate(tr_all))
    classifier = LogisticRegression(max_iter=1000,class_weight='balanced',random_state=20020130)
    classifier.fit(scaler.transform(np.concatenate(tr_all)),domain_tr)
    prediction = classifier.predict(scaler.transform(np.concatenate(va_all)))
    return dict(TEACHER_SHARED_ALIGNMENT_VALID='YES' if all(r['passed'] for r in rows) else 'NO',source_rows=rows,
                source_domain_accuracy=float(accuracy_score(domain_va,prediction)),
                source_domain_balanced_accuracy=float(balanced_accuracy_score(domain_va,prediction)),source_domain_chance=1/3,
                biological_labels_used=False,target_read=False,acceptance_thresholds=CONFIG['teacher_gate'])


def rng_state(rng):
    return dict(python=random.getstate(),numpy=np.random.get_state(),batch_rng=rng.bit_generator.state,
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def restore_rng(state,rng):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    rng.bit_generator.state=state['batch_rng']
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def train_stage(model,sources,unit,seed,device,teacher,inputs):
    unit.mkdir(parents=True,exist_ok=True)
    stage = 'teacher' if teacher else 'student'
    write(unit/'config.json',dict(**CONFIG,stage=stage,seed=seed))
    write(unit/'data_manifest.json',dict(input_hashes=inputs,target_read=False))
    optimizer = torch.optim.AdamW(model.parameters(),lr=CONFIG[f'{stage}_lr'],weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best,bad,start_epoch,history,scales = float('inf'),0,0,[],None
    if (unit/'latest.pt').exists():
        checkpoint = torch.load(unit/'latest.pt',map_location=device,weights_only=False)
        assert checkpoint['config_hash']==sha(unit/'config.json') and checkpoint['input_hashes']==inputs
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
        restore_rng(checkpoint['rng'],rng)
        best,bad,start_epoch=checkpoint['best_score'],checkpoint['bad_epochs'],checkpoint['epoch']
        history,scales=checkpoint['history'],checkpoint['initial_scales']
    elif teacher:
        model.eval();samples=[]
        with torch.no_grad():
            for i,batch in enumerate(batches(sources,np.random.default_rng(seed))):
                lr,lf,lp = teacher_losses(model,batch_tensor(batch,'rna',device),batch_tensor(batch,'field',device))
                samples.append([(lr+lf).item(),lp.item()])
                if i==7:break
        scales=np.maximum(np.mean(samples,axis=0),1e-8).tolist()
    if not teacher:
        mu=torch.as_tensor(sources[0]['latent_mean'],device=device)
        std=torch.as_tensor(sources[0]['latent_std'],device=device)
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    if not (unit/'manifest.json').exists():
        for epoch in range(start_epoch,CONFIG['epochs']):
            if bad>=CONFIG['patience']:break
            model.train();totals=[]
            for batch in batches(sources,rng):
                x=batch_tensor(batch,'rna',device)
                if teacher:
                    lr,lf,lp=teacher_losses(model,x,batch_tensor(batch,'field',device))
                    total=(lr+lf)/scales[0]+lp/scales[1]
                    values=[lr.item(),lf.item(),lp.item(),(lr+lf).item()/scales[0],lp.item()/scales[1],total.item()]
                else:
                    nonzero=batch_tensor(batch,'raw_nonzero',device)
                    mask=(torch.rand_like(x)<CONFIG['mask_rate'])&nonzero
                    # Mask raw nonzero counts before the frozen SOURCE standardization.
                    zero=torch.as_tensor(sources[0]['raw_zero_standardized'],device=device)
                    x=torch.where(mask,zero,x)
                    target=batch_tensor(batch,'teacher_latent',device)
                    pred=model(x)
                    total=F.mse_loss((pred-mu)/(std+1e-6),(target-mu)/(std+1e-6))
                    values=[total.item()]
                assert torch.isfinite(total)
                optimizer.zero_grad(set_to_none=True);total.backward()
                norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
                assert torch.isfinite(norm)
                optimizer.step();totals.append(values)
            model.eval();validation=[];student_diagnostics=[]
            with torch.no_grad():
                for s in sources:
                    acc,count,predictions,clean_targets=[],[],[],[]
                    for k in range(0,len(s['val']['rna']),512):
                        x=torch.from_numpy(s['val']['rna'][k:k+512]).to(device)
                        if teacher:
                            field=torch.from_numpy(s['val']['field'][k:k+512]).to(device)
                            lr,lf,lp=teacher_losses(model,x,field)
                            acc.append([lr.item(),lf.item(),lp.item(),(lr+lf).item()/scales[0],lp.item()/scales[1],((lr+lf)/scales[0]+lp/scales[1]).item()])
                        else:
                            target=torch.from_numpy(s['val']['teacher_latent'][k:k+512]).to(device)
                            pred=model(x)
                            acc.append([F.mse_loss((pred-mu)/(std+1e-6),(target-mu)/(std+1e-6)).item()])
                            predictions.append(pred.cpu().numpy());clean_targets.append(target.cpu().numpy())
                        count.append(len(x))
                    validation.append(np.average(acc,axis=0,weights=count).tolist())
                    if not teacher:
                        a,b=np.concatenate(predictions),np.concatenate(clean_targets)
                        student_diagnostics.append(dict(source=s['name'],rank=rank_info(a),
                              raw_mse=float(np.mean((a-b)**2)),cosine=float(np.mean(np.sum(a*b,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-8)))))
            val=np.mean(validation,axis=0).tolist();score=val[-1]
            record=dict(epoch=epoch+1,train=np.mean(totals,axis=0).tolist(),val=val,val_by_source=validation,
                        initial_scales=scales,component_order=['RNA_recon','F_recon','pair_MSE','normalized_info','normalized_pair','normalized_total'] if teacher else ['KD_MSE'],
                        elapsed_seconds=time.perf_counter()-started)
            if not teacher:record['source_student_diagnostics']=student_diagnostics
            history.append(record);improved=score<best
            if improved:best,bad=score,0
            else:bad+=1
            checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,best_score=best,
                            bad_epochs=bad,rng=rng_state(rng),history=history,initial_scales=scales,input_hashes=inputs,config_hash=sha(unit/'config.json'))
            save_checkpoint(unit/'latest.pt',checkpoint)
            if improved:save_checkpoint(unit/'best.pt',checkpoint)
            write(unit/'history.json',history)
            print('EPOCH',unit.parent.name,stage,seed,epoch+1,score,'best',best,flush=True)
        checkpoint=torch.load(unit/'best.pt',map_location=device,weights_only=False)
        model.load_state_dict(checkpoint['model'])
        write(unit/'manifest.json',dict(status='TRAINING_COMPLETE',best_epoch=checkpoint['epoch'],best_source_val_score=checkpoint['best_score'],seed=seed,
              best_checkpoint_sha256=sha(unit/'best.pt'),latest_checkpoint_sha256=sha(unit/'latest.pt'),input_hashes=inputs,config_hash=sha(unit/'config.json'),biological_labels_used=False,
              resources=dict(training_seconds=time.perf_counter()-started,parameters=sum(p.numel() for p in model.parameters()),
                             peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu=torch.cuda.get_device_name())))
    else:
        assert sha(unit/'best.pt')==read(unit/'manifest.json')['best_checkpoint_sha256']
        model.load_state_dict(torch.load(unit/'best.pt',map_location=device,weights_only=False)['model'])
    return model.eval()


def teacher_step(out,device):
    audits=[]
    for fold in FOLDS:
        sources,_,_,inputs=load_sources(fold,out)
        seed_all(20020130)
        model=Teacher(sources[0]['train']['rna'].shape[1],sources[0]['train']['field'].shape[1]).to(device)
        unit=out/'teacher'/fold/'seed_20020130'
        model=train_stage(model,sources,unit,20020130,device,True,inputs)
        audit=teacher_audit(model,sources,device)
        write(unit/'metrics.json',audit)
        audits.append(dict(fold=fold,**audit))
        print('TEACHER_GATE',fold,audit['TEACHER_SHARED_ALIGNMENT_VALID'],flush=True)
        del model
        torch.cuda.empty_cache()
    decision=dict(TEACHER_SHARED_ALIGNMENT_VALID='YES' if all(a['TEACHER_SHARED_ALIGNMENT_VALID']=='YES' for a in audits) else 'NO',
                  folds=audits,target_used=False)
    write(out/'TEACHER_GATE.json',decision)
    return decision


def geometry(a,b):
    mse=float(np.mean((a-b)**2))
    cos=float(np.mean(np.sum(a*b,1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-8)))
    idx=np.arange(min(len(a),1024));da,db=pdist(a[idx]),pdist(b[idx])
    count=min(16,len(a))
    na=NearestNeighbors(n_neighbors=count).fit(a).kneighbors(a,return_distance=False)
    nb=NearestNeighbors(n_neighbors=count).fit(b).kneighbors(b,return_distance=False)
    overlap=np.mean([len((set(x)-{i})&(set(y)-{i}))/(count-1) for i,(x,y) in enumerate(zip(na,nb))])
    valid=da.std()>0 and db.std()>0
    return dict(mse=mse,cosine=cos,pairwise_distance_pearson=float(np.corrcoef(da,db)[0,1]) if valid else 0.,
                pairwise_distance_spearman=float(spearmanr(da,db).statistic) if valid else 0.,knn_overlap=float(overlap))


def student_seed(out,seed,device):
    assert read(out/'TEACHER_GATE.json')['TEACHER_SHARED_ALIGNMENT_VALID']=='YES'
    rows=[]
    for fold in FOLDS:
        completed = out/'student'/fold/f'seed_{seed}'
        if (completed/'metrics.json').exists():
            result = read(completed/'metrics.json')
            assert sha(completed/'z_target_test.npy') == result['embedding_sha256']
            rows.append(result)
            continue
        sources,mean,std,inputs=load_sources(fold,out)
        teacher_unit=out/'teacher'/fold/'seed_20020130'
        teacher=Teacher(sources[0]['train']['rna'].shape[1],sources[0]['train']['field'].shape[1]).to(device)
        teacher.load_state_dict(torch.load(teacher_unit/'best.pt',map_location=device,weights_only=False)['model'])
        teacher.eval().requires_grad_(False)
        cache=out/'teacher_cache'/fold;cache.mkdir(parents=True,exist_ok=True)
        cache_hashes={}
        prior=read(cache/'manifest.json') if (cache/'manifest.json').exists() else None
        if prior:
            assert prior['teacher_checkpoint_sha256']==sha(teacher_unit/'best.pt')
        for s in sources:
            for split in ['train','val']:
                p=cache/f"{s['name']}_{split}_zT.npy"
                if prior:
                    assert sha(p)==prior['hashes'][str(p)]
                    z=np.load(p)
                else:
                    z=forward_numpy(teacher,s[split]['rna'],device,s[split]['field'])[2]
                    save_array(p,z)
                s[split]['teacher_latent']=z;cache_hashes[str(p)]=sha(p)
        mu=np.mean([s['train']['teacher_latent'].mean(0) for s in sources],axis=0)
        variance=np.mean([s['train']['teacher_latent'].var(0)+(s['train']['teacher_latent'].mean(0)-mu)**2 for s in sources],axis=0)
        sigma=np.sqrt(variance).astype(np.float32)
        for s in sources:s.update(latent_mean=mu,latent_std=sigma,raw_zero_standardized=-mean/(std+1e-6))
        write(cache/'manifest.json',dict(hashes=cache_hashes,teacher_checkpoint_sha256=sha(teacher_unit/'best.pt'),
              latent_mean=mu.tolist(),latent_std=sigma.tolist(),fit='SOURCE train dataset-balanced'))
        inputs.update(cache_hashes);inputs[str(teacher_unit/'best.pt')]=sha(teacher_unit/'best.pt')
        seed_all(seed)
        student=copy.deepcopy(teacher.rna_encoder).requires_grad_(True).to(device)
        unit=out/'student'/fold/f'seed_{seed}'
        student=train_stage(student,sources,unit,seed,device,False,inputs)
        mechanisms=[]
        for s in sources:
            zr,zf,zt,_,_=forward_numpy(teacher,s['val']['rna'],device,s['val']['field'])
            zs=forward_numpy(student,s['val']['rna'],device)
            baseline,recovered=geometry(zr,zt),geometry(zs,zt)
            mechanisms.append(dict(source=s['name'],student_rank=rank_info(zs),KD_MSE=float(np.mean(((zs-zt)/(sigma+1e-6))**2)),
              KD_cosine=recovered['cosine'],Student_vs_fused=recovered,Teacher_RNA_vs_fused=baseline,
              Student_vs_Teacher_RNA=geometry(zs,zr),MSE_recovery=1-recovered['mse']/max(baseline['mse'],1e-12),
              cosine_recovery=(recovered['cosine']-baseline['cosine'])/max(1-baseline['cosine'],1e-12)))
        write(unit/'source_mechanism.json',mechanisms)
        cm=read(GOAL1/'cache'/fold/'manifest.json');target_path=GOAL1/'cache'/fold/'target_rna_test.npy'
        assert sha(target_path)==cm['cache_sha256']['target_rna_test.npy']
        target=np.load(target_path);target=((target-mean)/(std+1e-6)).astype(np.float32)
        t=time.perf_counter();z=forward_numpy(student,target,device);elapsed=time.perf_counter()-t
        p=unit/'z_target_test.npy';digest=save_array(p,z)
        freeze=dict(status='EMBEDDING_FROZEN',embedding_sha256=digest,target_rna_sha256=sha(target_path),
              target_atac_used=False,target_statistics_fitted=False,target_graph_used=False,
              inference_seconds=elapsed,cells_per_second=len(z)/max(elapsed,1e-9),frozen_at=now())
        write(unit/'frozen_embedding_manifest.json',freeze)
        result=dict(fold=fold,seed=seed,**evaluate(z,labels_after_frozen(fold)),**freeze)
        assert sha(p)==digest
        write(unit/'metrics.json',result);rows.append(result)
        print('STUDENT_TARGET',fold,seed,result['ari'],result['nmi'],flush=True)
        del teacher,student
        torch.cuda.empty_cache()
    baselines=read(out/'CANONICAL_BASELINES.json')['rows'];deltas=[]
    for row in rows:
        base=next(r for r in baselines if r['method']=='RNA' and r['fold']==row['fold'] and r['training_seed']==seed)
        deltas.append(dict(fold=row['fold'],RNA_ari=base['ari'],RNA_nmi=base['nmi'],B2_ari=row['ari'],B2_nmi=row['nmi'],
                           delta_ari=row['ari']-base['ari'],delta_nmi=row['nmi']-base['nmi']))
    gain=float(np.mean([d['delta_ari'] for d in deltas]));nmi_gain=float(np.mean([d['delta_nmi'] for d in deltas]))
    nondegrading=sum(d['delta_ari']>=-1e-6 for d in deltas);mean_ari=float(np.mean([r['ari'] for r in rows]))
    if gain>=CONFIG['promising_delta_ari'] and nondegrading>=3 and nmi_gain>=-CONFIG['nmi_mean_regression_limit'] and min(d['delta_nmi'] for d in deltas)>=-CONFIG['nmi_fold_regression_limit']:
        category='STRONG' if mean_ari>=CONFIG['strong_mean_ari'] else 'PROMISING'
    else:category='NEGATIVE' if gain<=-.03 and nondegrading<=1 else 'NEUTRAL'
    summary=dict(seed=seed,rows=rows,deltas=deltas,mean_ari=mean_ari,mean_nmi=float(np.mean([r['nmi'] for r in rows])),
                 mean_delta_ari=gain,mean_delta_nmi=nmi_gain,nondegrading_folds=nondegrading,B2_TARGET_RESULT=category,
                 replication_authorized=category in ['STRONG','PROMISING'])
    write(out/f'B2_STUDENT_SEED{seed-20020100}.json',summary)
    return summary


def finalize(out,gate,results):
    status='COMPLETED' if gate['TEACHER_SHARED_ALIGNMENT_VALID']=='YES' and results and results[0]['B2_TARGET_RESULT'] in ['PROMISING','STRONG'] else 'COMPLETED_REJECT'
    final=dict(status=status,TEACHER_SHARED_ALIGNMENT_VALID=gate['TEACHER_SHARED_ALIGNMENT_VALID'],
          canonical_baselines=read(out/'CANONICAL_BASELINES.json')['summary'],teacher=gate,student_results=results,
          replication_seeds_run=[r['seed'] for r in results],C_CONTROL_READY=(out/'c_control/CONTROL_MANIFEST.json').exists(),
          completed_at=now(),scope='pre_final; not final 15-seed evidence',next_action='STOP after B2 adjudication')
    write(out/'metrics.json',final)
    manifest=read(out/'manifest.json');manifest.update(status=status,completed_at=now(),metrics_sha256=sha(out/'metrics.json'))
    write(out/'manifest.json',manifest);registry_update(out,status,[r['seed'] for r in results] or [20020130])
    state=read(ROOT/'CURRENT_RESEARCH_STATE.json')
    state['B2_Task35']=dict(status=status,result_path=str(out),teacher_gate=gate['TEACHER_SHARED_ALIGNMENT_VALID'],
          student_seeds=[r['seed'] for r in results],results=[{k:v for k,v in r.items() if k not in ['rows','deltas']} for r in results])
    state['current_max_task_id']=max(35,state.get('current_max_task_id',0));state['next_action']='STOP. Task35 B2 adjudicated; no other architecture launched.'
    write(ROOT/'CURRENT_RESEARCH_STATE.json',state)
    with (ROOT/'RESEARCH_DECISION_LEDGER.md').open('a') as handle:
        handle.write(f"\n\n## Task35 B2 adjudication {now()}\n\n- Canonical evaluator: n_init=100, clustering random_state=20020130 independent of training seed.\n- Teacher gate: {gate['TEACHER_SHARED_ALIGNMENT_VALID']}; status {status}; no target in Teacher training/selection.\n- Student summaries: {json.dumps([{k:v for k,v in r.items() if k not in ['rows','deltas']} for r in results])}.\n- Source B_D/F unchanged; inherited fixed cohort caveat retained. STOP; no other architecture launched.\n")
    lines=['# Task35 B2 Adjudication','',f'- Status: {status}',f"- Teacher gate: {gate['TEACHER_SHARED_ALIGNMENT_VALID']}",'',
           '| Seed | Fold | RNA ARI | B2 ARI | Delta ARI | RNA NMI | B2 NMI | Delta NMI |',
           '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in results:
        for d in r['deltas']:
            lines.append(f"| {r['seed']} | {d['fold']} | {d['RNA_ari']:.6f} | {d['B2_ari']:.6f} | {d['delta_ari']:.6f} | {d['RNA_nmi']:.6f} | {d['B2_nmi']:.6f} | {d['delta_nmi']:.6f} |")
    (out/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','canonical','smoke','run'],required=True)
    args=parser.parse_args();torch.set_num_threads(2);out=args.out
    if args.stage=='run' and read(out/'manifest.json')['status'].startswith('COMPLETED'):
        print('ALREADY_ADJUDICATED',flush=True)
        return
    with threadpool_limits(limits=2):
        if args.stage=='prepare':prepare(out)
        elif args.stage=='canonical':canonical(out)
        else:
            assert read(out/'CANONICAL_BASELINES.json')['status']=='COMPLETE'
            with (ROOT/'.b2_gpu_training.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                assert not active,active
                device=torch.device('cuda')
                if args.stage=='smoke':
                    sources,_,_,_=load_sources(FOLDS[0],out);seed_all(20020130)
                    model=Teacher(sources[0]['train']['rna'].shape[1],sources[0]['train']['field'].shape[1]).to(device)
                    batch=next(batches(sources,np.random.default_rng(20020130)))
                    r,f=batch_tensor(batch,'rna',device),batch_tensor(batch,'field',device)
                    lr,lf,lp=teacher_losses(model,r,f);total=lr+lf+lp;total.backward()
                    assert torch.isfinite(total) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                    student=copy.deepcopy(model.rna_encoder)
                    assert torch.equal(student.eval()(r),model.rna_encoder.eval()(r))
                    loss=F.mse_loss(student(r),model(r,f)[2].detach());loss.backward();assert torch.isfinite(loss)
                    write(out/'SMOKE_GATE.json',dict(status='PASS',source_only=True,teacher_student_forward_backward_finite=True,
                          teacher_parameters=sum(p.numel() for p in model.parameters()),student_parameters=sum(p.numel() for p in student.parameters())))
                else:
                    assert read(out/'SMOKE_GATE.json')['status']=='PASS'
                    registry_update(out,'RUNNING',[20020130]);gate=teacher_step(out,device);results=[]
                    if gate['TEACHER_SHARED_ALIGNMENT_VALID']=='YES':
                        for seed in SEEDS:
                            result=student_seed(out,seed,device);results.append(result)
                            if not result['replication_authorized']:break
                    finalize(out,gate,results)
                    print('FINAL',read(out/'manifest.json')['status'],flush=True)


if __name__=='__main__':
    main()
