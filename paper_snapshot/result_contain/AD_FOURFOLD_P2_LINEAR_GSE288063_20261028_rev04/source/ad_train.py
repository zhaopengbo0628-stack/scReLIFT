"""Run source-only frozen Full B2 and matched RNA AE; never inspect TARGET scores."""
import argparse, copy, inspect, math, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import ad_runtime as a

def module():
    b=a.frozen_b2()
    src=inspect.getsource(b.train_stage)
    needle="for s in sources:\n                    acc,count,predictions,clean_targets=[],[],[],[]"
    repl="for s in sources:\n                    if len(s['val']['rna']) == 0:\n                        continue\n                    acc,count,predictions,clean_targets=[],[],[],[]"
    assert src.count(needle)==1
    exec(compile(src.replace(needle,repl),__file__,'exec'),b.__dict__)
    b.CONFIG=copy.deepcopy(b.CONFIG)
    # Remove legacy TARGET-ARI gates: only this function's SOURCE-val score is used.
    for k in ['strong_mean_ari','promising_delta_ari','nmi_mean_regression_limit',
              'nmi_fold_regression_limit','EVALUATION_TIME_UNSUPERVISED_SCALING','KNOWN_K_EVALUATION']:
        b.CONFIG.pop(k,None)
    b.CONFIG.update(method='Full B2 AD four-fold',run_tier='formal_fixed_three_seeds',
                    validation='equal nonempty SOURCE validation means; donor-disjoint',target_adaptation=False)
    return b

def load_sources(fold):
    fd=a.CACHE[fold];cm=a.read(fd/'COMPLETE.json');cfg=a.protocol()['folds'][fold]
    assert cm['summary']['target']==cfg['target']
    assert cm['summary']['sources'].split('+')==cfg['sources']
    # Check every consumed cache against the CPU completion manifest, including B provenance.
    inputs={str(fd/'COMPLETE.json'):a.sha(fd/'COMPLETE.json'),str(a.OUT/'frozen_protocol.json'):a.EXPECTED_PROTOCOL,
            str(a.REFERENCE):a.REFERENCE_SHA}
    for p in [fd/'gene_ids.npy',fd/'B_D.npz']+[fd/n/f for n in cfg['sources'] for f in ['rna.npy','field.npy','metadata.npz']]:
        assert a.sha(p)==cm['hashes'][str(p)],p
        inputs[str(p)]=cm['hashes'][str(p)]
    for p in [a.OUT/'source/ad_runtime.py',a.OUT/'source/ad_train.py']:
        inputs[str(p)]=a.sha(p)
    sources=[]
    for name in cfg['sources']:
        p=fd/name;r=np.load(p/'rna.npy',mmap_mode='r');f=np.load(p/'field.npy',mmap_mode='r');meta=np.load(p/'metadata.npz')
        s={'name':name}
        for split,key in [('train','train_index'),('val','val_index')]:
            rows=meta[key];s[split]=dict(rna=np.asarray(r[rows]),field=np.asarray(f[rows]))
        assert not np.intersect1d(meta['train_index'],meta['val_index']).size
        sources.append(s)
    mu=np.mean([s['train']['rna'].mean(0,dtype=np.float64) for s in sources],0).astype(np.float32)
    var=np.mean([s['train']['rna'].var(0,dtype=np.float64)+(s['train']['rna'].mean(0)-mu)**2 for s in sources],0)
    sigma=np.sqrt(var).astype(np.float32)
    for s in sources:
        for part in ['train','val']:
            x=s[part]['rna'];s[part]['raw_nonzero']=x!=0
            s[part]['rna']=((x-mu)/(sigma+1e-6)).astype(np.float32)
            assert np.isfinite(s[part]['rna']).all() and np.isfinite(s[part]['field']).all()
    return sources,mu,sigma,inputs

class RNA_AE(nn.Module):
    def __init__(self,b,width):
        super().__init__();self.encoder=b.Encoder(width)
        self.decoder=nn.Sequential(nn.Linear(64,512),nn.GELU(),nn.Linear(512,width))
    def forward(self,x):return self.decoder(self.encoder(x))

def train_ae(b,sources,unit,seed,inputs,device):
    unit.mkdir(parents=True,exist_ok=True);b.seed_all(seed)
    model=RNA_AE(b,sources[0]['train']['rna'].shape[1]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    rng=np.random.default_rng(seed);best=float('inf');bad=0;history=[];start=0
    config=dict(epochs=100,patience=15,lr=1e-3,weight_decay=1e-4,clip=1.,
                batch_counts=[86,85,85],architecture=b.CONFIG['architecture'],seed=seed,
                objective='unmasked RNA reconstruction MSE',validation='equal nonempty SOURCE means')
    a.write(unit/'config.json',config);config_hash=a.sha(unit/'config.json')
    if (unit/'latest.pt').exists():
        ck=torch.load(unit/'latest.pt',map_location=device,weights_only=False)
        assert ck['input_hashes']==inputs and ck['config_hash']==config_hash
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);b.restore_rng(ck['rng'],rng)
        best,bad,start,history=ck['best_score'],ck['bad_epochs'],ck['epoch'],ck['history']
    started=time.time()
    if not (unit/'manifest.json').exists():
        for epoch in range(start,100):
            if bad>=15:break
            model.train();losses=[]
            for batch in b.batches(sources,rng):
                x=b.batch_tensor(batch,'rna',device);loss=F.mse_loss(model(x),x)
                assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward()
                norm=nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
                opt.step();losses.append(loss.item())
            model.eval();vals={}
            with torch.inference_mode():
                for s in sources:
                    v=s['val']['rna']
                    if not len(v):continue
                    total=0.
                    for i in range(0,len(v),512):
                        x=torch.from_numpy(v[i:i+512]).to(device)
                        total+=F.mse_loss(model(x),x,reduction='sum').item()
                    vals[s['name']]=total/v.size
            score=float(np.mean(list(vals.values())));assert math.isfinite(score)
            history.append(dict(epoch=epoch+1,train=float(np.mean(losses)),val=score,val_by_source=vals))
            improved=score<best
            if improved:best,bad=score,0
            else:bad+=1
            ck=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,best_score=best,
                    bad_epochs=bad,rng=b.rng_state(rng),history=history,input_hashes=inputs,config_hash=config_hash)
            b.save_checkpoint(unit/'latest.pt',ck)
            if improved:b.save_checkpoint(unit/'best.pt',ck)
            a.write(unit/'history.json',history)
            print('EPOCH',unit.parent.name,'RNA_AE',seed,epoch+1,score,'best',best,flush=True)
        a.write(unit/'manifest.json',dict(status='TRAINING_COMPLETE',best_checkpoint_sha256=a.sha(unit/'best.pt'),
                biological_labels_used=False,input_hashes=inputs,seconds=time.time()-started))
    assert a.sha(unit/'best.pt')==a.read(unit/'manifest.json')['best_checkpoint_sha256']
    model.load_state_dict(torch.load(unit/'best.pt',map_location=device,weights_only=False)['model'])
    return model.eval()

def main(fold,seed,smoke=False):
    assert seed in a.SEEDS and torch.cuda.is_available();torch.set_num_threads(2)
    b=module();device=torch.device('cuda');sources,mu,sigma,inputs=load_sources(fold)
    if smoke:
        b.seed_all(seed);teacher=b.Teacher(len(mu),len(mu)).to(device)
        batch=next(b.batches(sources,np.random.default_rng(seed)))
        x=b.batch_tensor(batch,'rna',device);field=b.batch_tensor(batch,'field',device)
        loss=sum(b.teacher_losses(teacher,x,field));loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in teacher.parameters())
        a.write(a.OUT/'GPU_SMOKE.json',dict(status='PASS',temporary=True,fold=fold,seed=seed,
                shape=list(x.shape),loss=float(loss.item()),gpu=torch.cuda.get_device_name(),inputs=inputs))
        print('KERNEL_WITNESS',x.shape,loss.item(),torch.cuda.get_device_name(),flush=True);return
    unit=a.OUT/'models'/fold/f'seed_{seed}';unit.mkdir(parents=True,exist_ok=True)
    np.savez(unit/'rna_normalization.npz',mean=mu,std=sigma)
    inputs[str(unit/'rna_normalization.npz')]=a.sha(unit/'rna_normalization.npz')
    b.seed_all(seed);teacher=b.Teacher(len(mu),len(mu)).to(device)
    teacher=b.train_stage(teacher,sources,unit/'teacher',seed,device,True,inputs)
    teacher.eval().requires_grad_(False)
    for s in sources:
        for part in ['train','val']:
            x=s[part]['rna'];f=s[part]['field'];p=unit/(s['name']+'_'+part+'_teacher_latent.npy')
            z=b.forward_numpy(teacher,x,device,f)[2] if len(x) else np.empty((0,64),np.float32)
            np.save(p,z);s[part]['teacher_latent']=z;inputs[str(p)]=a.sha(p)
    lm=np.mean([s['train']['teacher_latent'].mean(0) for s in sources],0)
    lv=np.mean([s['train']['teacher_latent'].var(0)+(s['train']['teacher_latent'].mean(0)-lm)**2 for s in sources],0)
    for s in sources:s.update(latent_mean=lm,latent_std=np.sqrt(lv).astype(np.float32),raw_zero_standardized=-mu/(sigma+1e-6))
    inputs[str(unit/'teacher/best.pt')]=a.sha(unit/'teacher/best.pt')
    b.seed_all(seed);student=copy.deepcopy(teacher.rna_encoder).requires_grad_(True).to(device)
    student=b.train_stage(student,sources,unit/'student',seed,device,False,inputs)
    del teacher,student;torch.cuda.empty_cache()
    # AE gets identical RNA arrays and splits, no Teacher fields or labels enter its objective.
    train_ae(b,sources,unit/'rna_ae',seed,inputs,device)
    a.write(unit/'TRAINING_COMPLETE.json',dict(status='COMPLETE',fold=fold,seed=seed,target_used=False,
        checkpoints={str(unit/k/'best.pt'):a.sha(unit/k/'best.pt') for k in ['teacher','student','rna_ae']}))
    print('FORMAL_TRAINING_COMPLETE',fold,seed,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',required=True,choices=list(a.CACHE))
    p.add_argument('--seed',required=True,type=int);p.add_argument('--smoke',action='store_true')
    v=p.parse_args();main(v.fold,v.seed,v.smoke)
