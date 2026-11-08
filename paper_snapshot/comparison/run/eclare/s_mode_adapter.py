"""Source-only ECLARE ensemble using upstream CLIP and KD/OT losses."""
import sys

import numpy as np
import torch

from pbmc_intra_common import ROOT


def modules():
    sys.path.insert(0,str(ROOT/'comparison/methods/eclare/official/src'))
    from eclare.models import CLIP
    from eclare.losses_and_distances_utils import Knowledge_distillation_fn, clip_loss
    return CLIP, Knowledge_distillation_fn, clip_loss


def make_model(peaks, device, layers=3):
    CLIP,_,_ = modules()
    return CLIP(n_peaks=peaks,n_genes=2000,paired=True,num_units=64,
        num_layers=layers,dropout_p=0.3,decoder_loss=None).to(device)


def inputs(data, split, rows, device):
    r = torch.as_tensor(data[f'rna_log_{split}'][rows].toarray(),dtype=torch.float32,device=device)
    a = torch.as_tensor(data[f'atac_tfidf_{split}'][rows].toarray(),dtype=torch.float32,device=device)
    return r,a


def components(model, teachers, kd, r, a):
    sr,sa = model(r,0)[0],model(a,1)[0]
    losses = []
    for teacher in teachers:
        with torch.no_grad():
            tr,ta = teacher(r,0)[0],teacher(a,1)[0]
        losses.append(kd(sr,sa,tr,ta,'teacher'))
    stack = [torch.stack([row[i] for row in losses]) for i in (0,1,2,3,4,5)]
    distil = kd.distil_loss_weighting(stack[0],stack[1],stack[4]-stack[2],stack[5]-stack[3])[0]
    alignment = kd(sr,sa,tr,ta,'student')[2]
    if kd.all_teacher_ot_plans or kd.all_teacher_ot_values:
        raise RuntimeError('OT plans leaked between batches')
    return distil,alignment


def fit(data, config, unit, seed, smoke, device, cache, train):
    CLIP,KD,clip_loss = modules()
    peaks = data['atac_train'].shape[1]
    teachers = []
    params = 0
    for index in range(config['teachers']):
        torch.manual_seed(seed+index)
        teacher = make_model(peaks,device,layers=1)
        params += sum(p.numel() for p in teacher.parameters())
        def teacher_loss(split, rows):
            r,a = inputs(data,split,rows,device)
            value = clip_loss(logits=None,atac_latents=teacher(a,1)[0],rna_latents=teacher(r,0)[0],temperature=1.)
            return 0.5*(value[0]+value[1])
        train(teacher, teacher_loss, unit/f'teacher_{index}.pt', f'teacher_{index}_CLIP')
        teacher.eval().requires_grad_(False)
        teachers.append(teacher)
    torch.manual_seed(seed)
    student = make_model(peaks,device)
    hparams = CLIP.get_hparams()
    kd = KD(device=device,paired=True,weigh_distil_by_align_type='sample',
        **{k:float(hparams[k]['default']) for k in ('student_temperature','teacher_temperature','weights_temperature')})
    params += sum(p.numel() for p in student.parameters())+sum(p.numel() for p in kd.parameters())
    student.eval()
    with torch.no_grad():
        d,a = [],[]
        for offset in range(0,data['rna_val'].shape[0],config['batch_size']):
            rows = np.arange(offset,min(offset+config['batch_size'],data['rna_val'].shape[0]))
            x,y = components(student,teachers,kd,*inputs(data,'val',rows,device))
            d.append(float(x)*len(rows)); a.append(float(y)*len(rows))
        scale = sum(d)/sum(a)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid Source-validation alignment calibration')
    kd.align_loss_scale = scale
    def student_loss(split, rows):
        d,a = components(student,teachers,kd,*inputs(data,split,rows,device))
        return config['distil_lambda']*d+(1-config['distil_lambda'])*a
    best = train(student, student_loss, unit/'best.pt','ensemble_KL_OT',extra=kd)
    state = torch.load(unit/'best.pt',map_location='cpu',weights_only=False)
    state.update(method='eclare',peaks=peaks,alignment_scale=scale,teacher_count=len(teachers))
    torch.save(state,unit/'best.pt')
    return student,dict(best_epoch=best['best_epoch'],best_validation_loss=best['validation_loss'],
        trainable_parameters=params,teacher_count=len(teachers),alignment_scale=scale,
        adaptation='ECLARE-S: three independent Source teachers on the same dataset; official ensemble KL and OT; Source-only student')


def predict(model, matrix, checkpoint, device, batch):
    with torch.inference_mode():
        return np.concatenate([model(torch.as_tensor(matrix[i:i+batch].toarray(),dtype=torch.float32,device=device),0)[0].cpu().numpy() for i in range(0,matrix.shape[0],batch)])
