"""One SOURCE-selected Task31 geometry control; frozen Teacher and original local objective."""
import argparse
import copy
import math
import sys
import time
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

sys.path.insert(0, '/home/u202522802045/share/zpb/result_contain/36_b2_three_causal_ablations_20261003/source/experiments/v5d')
import run_b2_strict_lodo as b
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_strict_lodo_two_step as strict
sys.path.insert(0, '/home/u202522802045/share/zpb/V5/Teacher-S10x20_Student-64D-KL/source')
from v5s.denoise import DenoisingAutoencoder

p = strict.parent
ORIGINAL_BATCH, ORIGINAL_VAL = p._batch_losses, p._validation_losses


def batch_losses(model, source, bank, target, anchors, mean, std, config, device):
    local, _ = ORIGINAL_BATCH(model, source, bank, target, anchors, mean, std, config, device)
    left, _, _ = p._bank_batch(bank, anchors)
    unique, inverse = np.unique(np.r_[left, target.landmarks], return_inverse=True)
    raw = p._encode_rows(model, source, unique, 'train', mean, std, device)
    anchor = raw[torch.from_numpy(inverse[:len(left)]).to(device)]
    landmark = raw[torch.from_numpy(inverse[len(left):]).to(device)]
    distances = (torch.cdist(anchor, landmark) - target.mean) / target.std
    return local, torch.nn.functional.mse_loss(distances, torch.from_numpy(target.values[left]).to(device))


@torch.no_grad()
def validation_losses(model, source, bank, target, mean, std, config, device):
    local, _ = ORIGINAL_VAL(model, source, bank, target, mean, std, config, device)
    raw = model.forward_raw(torch.from_numpy(p.normalized_rows(source, source.val, mean, std)).to(device))
    distances = (torch.cdist(raw, raw[torch.from_numpy(target.landmarks).to(device)]) - target.mean) / target.std
    return local, float(torch.nn.functional.mse_loss(distances, torch.from_numpy(target.values).to(device)))


def metric(z, target):
    distances = p._landmark_distances(z, target.landmarks)
    prediction = (distances - target.mean) / target.std
    return dict(normalized_distance_mse=float(np.mean((prediction-target.values)**2)),
        normalized_distance_pearson=float(np.corrcoef(prediction.ravel(), target.values.ravel())[0,1]),
        normalized_distance_spearman=float(spearmanr(prediction.ravel(), target.values.ravel()).statistic))


def setup(fold):
    args = SimpleNamespace(out=b.GOAL1, root=b.ROOT)
    sources = strict.sources_for(args, fold)
    unit = b.GOAL1 / 'parent' / fold / 'seed_20020130'
    old_config = b.read(unit / 'config.json')
    config = p.V5DStudentConfig(**{f.name: old_config[f.name] for f in fields(p.V5DStudentConfig) if f.name in old_config})
    path = b.GOAL1 / 'rna' / fold / 'seed_20020130/shared_rna_best.pt'
    saved = torch.load(path, map_location='cpu', weights_only=False)
    dae = DenoisingAutoencoder(saved['feature_count'], 512, 64)
    dae.load_state_dict(saved['model_state_dict'])
    train_banks, val_banks, train_targets, val_targets = {}, {}, {}, {}
    inputs = {str(path):b.sha(path)}
    for s in sources:
        banks = b.ROOT / 'result_contain/26_dmode_rna_scaffold_bounded_residual_20260929' / fold / 'pair_banks'
        tp, vp = banks / f'{s.name}_train.npz', banks / f'{s.name}_val.npz'
        train_banks[s.name], val_banks[s.name] = p._project_topology_embeddings(p.load_pair_bank(tp), p.load_pair_bank(vp), 64)
        inputs.update({str(tp):b.sha(tp),str(vp):b.sha(vp)})
        for split, store in [('train',train_targets),('val',val_targets)]:
            path = b.ROOT / 'result_contain/31_dmode_spectral_landmark_global_20260930' / fold / 'global_targets' / f'{s.name}_{split}.npz'
            store[s.name] = strict.load_global(path)
            inputs[str(path)] = b.sha(path)
    wp = b.ROOT / 'result_contain' / strict.SOURCE_ROOT / fold / 'dataset_weights.npz'
    weights = dict(zip([s.name for s in sources], map(float, np.load(wp)['omega'])))
    inputs[str(wp)] = b.sha(wp)
    for path, digest in b.read(unit / 'manifest.json')['source_artifact_hashes'].items():
        assert b.sha(path) == digest
    return sources, config, dae.encoder, saved['rna_mean'], saved['rna_std'], train_banks, val_banks, train_targets, val_targets, weights, inputs


def audit(out):
    records = []
    for fold in p.FOLDS:
        sources, config, backbone, mean, std, tb, vb, tt, vt, weights, inputs = setup(fold)
        torch.manual_seed(20020130)
        model = p.RNAScaffoldBoundedResidualEncoder(backbone,64,128,.10).eval()
        for s in sources:
            path = b.ROOT / 'result_contain/23_dmode_snf_correction_localization_audit_20260927/matched_geometry' / fold / s.name / 'G_SNF_verified.npy'
            _, coords = p._spectral_coordinates(np.load(path))
            target = vt[s.name]
            original = p._landmark_distances(coords,target.landmarks)
            np.testing.assert_allclose(original, target.values*target.std+target.mean, atol=3e-5,rtol=3e-5)
            normalized = coords / np.maximum(np.linalg.norm(coords,axis=1,keepdims=True),1e-8)
            fixed = p._landmark_distances(normalized,target.landmarks)
            distances = np.linalg.norm(coords[:,None,:]-coords[None,:,:],axis=2)
            normalized_distances = np.linalg.norm(normalized[:,None,:]-normalized[None,:,:],axis=2)
            np.fill_diagonal(distances,np.inf);np.fill_diagonal(normalized_distances,np.inf)
            nearest = np.argsort(distances,axis=1)[:,:15]
            fix_nearest = np.argsort(normalized_distances,axis=1)[:,:15]
            overlap = np.mean([len(set(x)&set(y))/15 for x,y in zip(nearest,fix_nearest)])
            with torch.no_grad():
                raw = model.forward_raw(torch.from_numpy(p.normalized_rows(s,s.val,mean,std))).numpy()
            original_local, _ = ORIGINAL_VAL(model,s,vb[s.name],target,mean,std,config,torch.device('cpu'))
            fixed_local, _ = validation_losses(model,s,vb[s.name],target,mean,std,config,torch.device('cpu'))
            assert original_local == fixed_local
            records.append(dict(fold=fold,source=s.name,graph_sha256=b.sha(path),target_hash=inputs[str(b.ROOT/'result_contain/31_dmode_spectral_landmark_global_20260930'/fold/'global_targets'/f'{s.name}_val.npz')],
                FIX_A=dict(teacher_top15_preservation=float(overlap), teacher_landmark_ranking_spearman=float(spearmanr(original.ravel(),fixed.ravel()).statistic),
                           changes_teacher=True,geometry_consistent=True),
                FIX_B=dict(teacher_top15_preservation=1.,teacher_landmark_ranking_spearman=1.,changes_teacher=False,geometry_consistent=True),
                initial_student_norm_mean=float(np.linalg.norm(raw,axis=1).mean()), teacher_norm_mean=float(np.linalg.norm(coords,axis=1).mean()),
                initial_raw_global_mse=metric(raw,target)['normalized_distance_mse'],local_loss_identical=True))
    b.write(out/'C_GEOMETRY_SOURCE_AUDIT.json',dict(records=records,selected='FIX-B',target_used=False,labels_used=False,
        decision='FIX-B preserves exact original Teacher rankings/neighbors and targets; FIX-A changes them. Raw/raw distance definition selected, no Teacher change.',
        limitation='row-normalization mismatch removed; raw Student/Teacher magnitude mismatch may remain because residual capacity is fixed'))
    print('C_FIX_SELECTED FIX-B',flush=True)


def train(fold,out,smoke=False):
    unit = out / ('C_SMOKE' if smoke else 'C1') / fold / 'seed_20020130'
    if (unit/'metrics.json').exists():
        return b.read(unit/'metrics.json')
    sources, config, backbone, mean, std, tb, vb, tt, vt, weights, inputs = setup(fold)
    unit.mkdir(parents=True,exist_ok=True)
    epochs = 1 if smoke else config.max_epochs
    b.write(unit/'config.json',dict(**config.to_dict(),geometry_fix='FIX-B',seed=20020130,smoke=smoke,
                                 local_definition_unchanged=True,epochs=epochs))
    b.write(unit/'data_manifest.json',dict(input_hashes=inputs,target_used_for_training=False))
    device = torch.device('cuda')
    b.seed_all(20020130)
    model = p.RNAScaffoldBoundedResidualEncoder(backbone,64,128,.10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    rng = np.random.default_rng(20020130)
    with torch.no_grad():
        initial = [batch_losses(model,s,tb[s.name],tt[s.name],np.arange(min(config.batch_anchor,tb[s.name].anchors)),mean,std,config,device) for s in sources]
    scales = [max(float(np.average([float(v[i]) for v in initial],weights=[weights[s.name] for s in sources])),1e-8) for i in [0,1]]
    history,best_topology,best_total,start = [],-float('inf'),float('inf'),0
    steps = max(math.ceil(s.train.size/config.batch_anchor) for s in sources)
    latest = unit/'latest.pt'
    if latest.exists():
        saved = torch.load(latest,map_location=device,weights_only=False)
        assert saved['input_hashes']==inputs and saved['config_hash']==b.sha(unit/'config.json')
        model.load_state_dict(saved['model_state_dict']);optimizer.load_state_dict(saved['optimizer'])
        b.restore_rng(saved['rng'],rng)
        history,start,best_topology,best_total,scales = saved['history'],saved['epoch'],saved['best_topology'],saved['best_total'],saved['scales']
    else:
        b.save_checkpoint(unit/'initial.pt',dict(model_state_dict=model.state_dict(),optimizer=optimizer.state_dict(),epoch=0,rng=b.rng_state(rng),input_hashes=inputs,scales=scales))
    started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start+1,epochs+1):
        model.train();train_local=[];train_global=[]
        for _ in range(steps):
            losses=[]
            for s in sources:
                anchors=rng.integers(0,tb[s.name].anchors,size=min(config.batch_anchor,tb[s.name].anchors))
                local,glob=batch_losses(model,s,tb[s.name],tt[s.name],anchors,mean,std,config,device)
                losses.append((local/scales[0]+glob/scales[1])*weights[s.name])
                train_local.append(float(local.detach()));train_global.append(float(glob.detach()))
            loss=torch.stack(losses).sum()/sum(weights.values())
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),config.grad_clip);optimizer.step()
        topology=[];local_val=[];global_val=[]
        for s in sources:
            z=p._encode_rows(model,s,np.arange(len(s.val)),'val',mean,std,device).detach().cpu().numpy()
            lv,gv=validation_losses(model,s,vb[s.name],vt[s.name],mean,std,config,device)
            local_val.append(lv);global_val.append(gv)
            topology.append(p._topology_score(z,vb[s.name])['teacher_recall_at_10'])
        ws=[weights[s.name] for s in sources]
        topo=float(np.average(topology,weights=ws));lv=float(np.average(local_val,weights=ws));gv=float(np.average(global_val,weights=ws))
        total=lv/scales[0]+gv/scales[1]
        record=dict(epoch=epoch,train_local_kl=float(np.mean(train_local)),train_global_landmark_mse=float(np.mean(train_global)),
                    val_local_kl=lv,val_global_landmark_mse=gv,val_total_normalized=total,val_topology_score_teacher_recall_at_10=topo)
        history.append(record)
        improve_topo,improve_total=topo>best_topology,total<best_total
        best_topology=max(topo,best_topology);best_total=min(total,best_total)
        ck=dict(model_state_dict=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,rng=b.rng_state(rng),history=history,
                best_topology=best_topology,best_total=best_total,scales=scales,config_hash=b.sha(unit/'config.json'),input_hashes=inputs,
                rna_mean=mean,rna_std=std)
        b.save_checkpoint(latest,ck)
        if improve_topo:
            b.save_checkpoint(unit/'best_by_topology.pt',ck)
            b.save_checkpoint(unit/f'new_best_epoch_{epoch:03d}.pt',ck)
        if improve_total:
            b.save_checkpoint(unit/'best_by_total_normalized_loss.pt',ck)
        if epoch%10==0:
            b.save_checkpoint(unit/f'epoch_{epoch:03d}.pt',ck)
        b.write(unit/'training.json',dict(history=history,local_initial_scale=scales[0],global_initial_scale=scales[1]))
        if epoch%10==0 or smoke:print('C_EPOCH',fold,record,flush=True)
    saved=torch.load(unit/'best_by_topology.pt',map_location=device,weights_only=False)
    model.load_state_dict(saved['model_state_dict']);model.eval()
    source=[]
    for s in sources:
        z=strict.infer(model,s.rna[s.val],mean,std,raw=True)
        source.append(dict(source=s.name,topology=p._topology_score(z,vb[s.name]),global_teacher_fidelity=metric(z,vt[s.name])))
        b.save_array(unit/f'{s.name}_z_val.npy',z)
    b.write(unit/'source_validation_metrics.json',source)
    p._batch_losses=batch_losses
    b.write(unit/'diagnostics.json',strict.diagnose(model,sources,mean,std,config,tb,tt,
             dict(local_initial_scale=scales[0],global_initial_scale=scales[1])))
    p._batch_losses=ORIGINAL_BATCH
    b.write(unit/'resources.json',dict(training_seconds=time.perf_counter()-started,peak_allocated=torch.cuda.max_memory_allocated(),
                                    peak_reserved=torch.cuda.max_memory_reserved(),parameters=sum(x.numel() for x in model.parameters()),
                                    trainable_parameters=sum(x.numel() for x in model.parameters() if x.requires_grad)))
    if smoke:
        b.write(out/'C_SMOKE_GATE.json',dict(status='PASS',finite_loss=True,source_only=True,checkpoint_resume_fields=True))
        return
    target=np.load(b.GOAL1/'cache'/fold/'target_rna_test.npy')
    z=strict.infer(model,target,mean,std,raw=True)
    digest=b.save_array(unit/'z_target_test.npy',z)
    b.write(unit/'frozen_embedding_manifest.json',dict(embedding_sha256=digest,target_atac_used=False,target_population_fitting=False))
    result=dict(fold=fold,seed=20020130,embedding_sha256=digest,**b.evaluate(z,b.labels_after_frozen(fold)))
    b.write(unit/'metrics.json',result)
    b.write(unit/'checkpoint_index.json',{str(path):b.sha(path) for path in unit.glob('*.pt')})
    print('C_RESULT',fold,result['ari'],result['nmi'],flush=True)
    torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--stage',choices=['audit','smoke','run'],required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.stage=='audit':audit(args.out)
        elif args.stage=='smoke':train('target_D4',args.out,True)
        else:
            assert b.read(args.out/'C_GEOMETRY_SOURCE_AUDIT.json')['selected']=='FIX-B'
            assert b.read(args.out/'C_SMOKE_GATE.json')['status']=='PASS'
            rows=[train(fold,args.out) for fold in p.FOLDS]
            b.write(args.out/'C1_PER_UNIT.json',rows)


if __name__=='__main__':main()
