"""Derived frozen computational core; see provenance/CORE_EXTRACTION.json."""

from __future__ import annotations

import argparse

import copy

import csv

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
