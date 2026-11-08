"""Use the frozen B2-S training engine with direct ATAC in its second input slot."""
import argparse
import copy
import fcntl
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path('/home/u202522802045/share/zpb')
sys.path[:0]=[str(ROOT),str(ROOT/'main/run'),str(ROOT/'main/run/d_mode'),str(ROOT/'main/core/regulatory/common/legacy_support')]
from data.code.s_mode.prepare_no_bf import prepare

REFERENCE=ROOT/'main/run/s_mode/train.py'
spec=importlib.util.spec_from_file_location('no_bf_smode_engine',REFERENCE)
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
b=s.b
BASE_TEACHER=b.Teacher
METRICS=['ari','nmi','ami','cluster_acc','accuracy','balanced_accuracy','macro_f1','weighted_f1','knn_purity_at_15']


def source_data(cache):
    state=np.load(cache/'rna_state.npz');mean,std=state['mean'],state['std']
    source=dict(name=cache.name,key=cache.name,raw_zero_standardized=-mean/(std+1e-6))
    for split in ['train','val']:
        raw=np.load(cache/f'rna_{split}.npy')
        # The generic engine's "field" slot carries direct ATAC; no F is loaded.
        source[split]=dict(rna=((raw-mean)/(std+1e-6)).astype(np.float32),raw_nonzero=raw!=0,
                           field=np.load(cache/f'atac_{split}.npy',mmap_mode='r'))
    width=source['train']['field'].shape[1]
    b.Teacher=lambda rdim,fdim: BASE_TEACHER(rdim,width)
    return source,mean,std


def diagnostics(teacher,source,device):
    rows=[]
    for split in ['train','val']:
        outputs=b.forward_numpy(teacher,source[split]['rna'],device,source[split]['field'])
        source[split]['teacher_latent']=outputs[2]
        for name,value in zip(['z_R','z_A','z_T'],outputs[:3]):
            s.finite(value,name);rows.append(dict(split=split,representation=name,rank=b.rank_info(value)))
    return dict(policy='WARNING_ONLY; never select by Target',latent_diagnostics=rows,
                B_used=False,F_used=False,input_slot_field='direct Source-preprocessed ATAC')


def cpu_preflight(cache,out,key):
    source,_,_=source_data(cache)
    random.seed(20020130);np.random.seed(20020130);torch.manual_seed(20020130)
    model=b.Teacher(2000,source['train']['field'].shape[1])
    optimizer=torch.optim.AdamW(model.parameters(),lr=s.BASE_CONFIG['teacher_lr'],weight_decay=1e-4)
    x=torch.from_numpy(source['train']['rna'][:8].copy())
    a=torch.from_numpy(source['train']['field'][:8].copy())
    model.train();lr,la,lp=b.teacher_losses(model,x,a);loss=lr+la+lp
    assert torch.isfinite(loss)
    loss.backward();assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
    optimizer.step();model.eval()
    vx=torch.from_numpy(source['val']['rna'][:8].copy());va=torch.from_numpy(source['val']['field'][:8].copy())
    with torch.no_grad():
        outputs=model(vx,va)
        for value in outputs:assert torch.isfinite(value).all()
        torch.testing.assert_close(outputs[2],.5*(outputs[0]+outputs[1]))
    path=out/'preflight'/key;path.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=model.state_dict(),temporary_cpu_smoke=True),path/'teacher.pt')
    model.load_state_dict(torch.load(path/'teacher.pt',map_location='cpu',weights_only=False)['model'])
    with torch.no_grad():torch.testing.assert_close(model(vx,va)[2],outputs[2])
    student=copy.deepcopy(model.rna_encoder);student.train()
    kd=torch.nn.functional.mse_loss(student(vx),outputs[2].detach());kd.backward()
    assert torch.isfinite(kd) and all(torch.isfinite(p.grad).all() for p in student.parameters())
    b.write(path/'GATE.json',dict(status='PASS',temporary_validation='8 Source train and 8 Source val cells; CPU',
        encoder_linear_layers=2,atac_width=a.shape[1],latent_width=64,mean_fusion_exact=True,
        teacher_forward_backward_finite=True,student_KD_backward_finite=True,checkpoint_reload_equal=True,
        teacher_parameters=sum(p.numel() for p in model.parameters()),B_used=False,F_used=False,test_cells_used=False))
    print('NO_BF_CPU_GATE_PASS',key,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','preflight','run'],required=True)
    parser.add_argument('--source',choices=['D1','D2','D3','D4'],required=True)
    parser.add_argument('--seed',type=int,choices=s.SEEDS,default=s.SEEDS[0]);args=parser.parse_args()
    config=json.loads(args.config.read_text());out=Path(config['result_root'])
    for path,digest in config['execution_source_hashes'].items():
        assert b.sha(path)==digest,path
    torch.set_num_threads(2)
    cache,manifest=prepare(args.source,config['cache_root'],config['identities'][args.source])
    if args.stage=='preflight':
        cpu_preflight(cache,out,args.source)
    elif args.stage=='run':
        assert b.read(out/'preflight'/args.source/'GATE.json')['status']=='PASS'
        s.source_data=source_data;s.warnings=diagnostics
        manifest=copy.deepcopy(manifest)
        manifest['hashes'].update(config['execution_source_hashes'])
        manifest['hashes'][str(args.config)]=b.sha(args.config)
        s.BASE_CONFIG.update(condition='S-no-BF-direct-ATAC-mean',B_used=False,F_used=False,
            teacher_objectives=['RNA+ATAC reconstruction','same-cell MSE'],
            reconstruction_component_note='generic F_recon/field names mean direct ATAC in this adapter')
        with (ROOT/'.benchmark_a30.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            active=b.subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
            assert not active.strip(),'GPU occupied outside the benchmark lock'
            with s.threadpool_limits(2):
                s.train(args.source,args.seed,out,cache,manifest)
        unit=out/'models'/args.source/f'seed_{args.seed}'
        for path in unit.glob('evaluation/*/manifest.json'):
            record=b.read(path);label= s.legacy.DATA[record['target']]/'processed/label_harmonized.csv'
            record.update(condition=config['condition'],B_used=False,F_used=False,label_file_sha256=b.sha(label),
                          geometry_pearson=None,geometry_spearman=None)
            record['hashes'][str(label)]=record['label_file_sha256']
            b.write(path,record)
