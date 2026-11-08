"""Official scMomer modules with label-free Source S2/S3 adaptation."""
import csv
import ast
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from pbmc_intra_common import ROOT, sha, write


def assets(cache):
    root = ROOT/'comparison/methods/scmomer'
    reference = root/'pretrained_assets/panglao_10000.h5ad'
    weights = root/'pretrained_assets/pretrained_scmomer.pth'
    annotation = root/'adapter_assets/processed_gene_id_to_symbol.tsv'
    provenance = root/'adapter_assets/bmmc_gene_mapping_provenance.json'
    import h5py
    with h5py.File(reference) as h:
        genes = h['var'][h['var'].attrs['_index']].asstr()[:]
    with annotation.open() as h:
        names = {row['gene_id']:row['gene_symbol'] for row in csv.DictReader(h,delimiter='\t')}
    from pbmc_intra_common import read
    for path, expected in read(provenance)['sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'Changed processed gene mapping: {path}')
    positions = {name:i for i,name in enumerate(genes)}
    mapping = np.asarray([positions.get(names.get(g.split('.')[0]), -1) for g in np.load(cache/'gene_ids.npy')])
    valid = mapping[mapping >= 0]
    if not len(valid) or len(np.unique(valid)) != len(valid):
        raise ValueError('Empty or ambiguous reference gene mapping')
    return mapping, {str(p):sha(p) for p in (reference, weights, annotation, provenance)}


def aligned(matrix, mapping, device):
    values = np.zeros((matrix.shape[0], 16906), dtype=np.float32)
    keep = mapping >= 0
    values[:, mapping[keep]] = matrix[:, keep].toarray()/np.log(2.)
    return torch.as_tensor(values, device=device)


def make_model(peaks, device, paired=False):
    official = ROOT/'comparison/methods/scmomer/official'
    sys.path.insert(0, str(official))
    from model.performer import performer_pytorch as performer
    from model.scmomer import scMomer
    from model.atac_encoder import MAEConfig
    stage3 = official/'s3_pretrain_missing.py'
    student = next(node for node in ast.parse(stage3.read_text()).body if isinstance(node,ast.ClassDef) and node.name=='StudentEncoder')
    namespace = {'nn':nn}
    exec(compile(ast.Module(body=[student],type_ignores=[]),str(stage3),'exec'),namespace)
    from utils import Reconstruct_net
    state = torch.load(ROOT/'comparison/methods/scmomer/pretrained_assets/pretrained_scmomer.pth', map_location='cpu', weights_only=False)['model_state_dict']
    # Replace only the upstream hardcoded positional-file constructor; exact weights come from the checkpoint.
    class CheckpointPositions(nn.Module):
        def __init__(self, dim, max_seq_len):
            super().__init__()
            self.emb = nn.Embedding.from_pretrained(state['rna_model.model.pos_emb.emb.weight'].clone())
        def forward(self, x):
            return self.emb.weight[:x.shape[1]]
    original = performer.Gene2VecPositionalEmbedding
    performer.Gene2VecPositionalEmbedding = CheckpointPositions
    try:
        model = scMomer(SimpleNamespace(projection_dim=128, normalize=True),
            MAEConfig(feature_size=peaks) if paired else None,
            Reconstruct_net(16906,128) if paired else None,
            Reconstruct_net(peaks,128) if paired else None,
            encoder=namespace['StudentEncoder']())
    finally:
        performer.Gene2VecPositionalEmbedding = original
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith(('atac_model.', 'atac_projection.', 'RNA.', 'ATAC.')) for k in missing):
        raise ValueError((missing, unexpected))
    model.rna_model.requires_grad_(False)
    model.rna_projection.requires_grad_(False)
    model.rna_ln.requires_grad_(False)
    return model.to(device)


def rna_features(model, matrix, mapping, device, batch):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(matrix.shape[0]):
            r = aligned(matrix[i:i+1],mapping,device)
            out.append(model.rna_ln(model._get_rna_features(r.clamp(0,5).long())).cpu())
    return torch.cat(out)


def fit(data, config, unit, seed, smoke, device, cache, train):
    mapping, hashes = assets(cache)
    write(unit/'assets.json',dict(sha256=hashes,observed_hvg_genes=len(mapping),mapped_genes=int((mapping>=0).sum()),reference_genes=16906))
    np.save(unit/'reference_columns.npy', mapping)
    peaks = data['atac_train'].shape[1]
    model = make_model(peaks,device,paired=True)
    features = {s:rna_features(model,data[f'rna_log_{s}'],mapping,device,config['rna_batch_size']) for s in ('train','val')}
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.],device=device))
    def paired_loss(split, rows):
        r = aligned(data[f'rna_log_{split}'][rows],mapping,device)
        a = torch.as_tensor(data[f'atac_{split}'][rows].toarray(),dtype=torch.float32,device=device)
        ar = model.atac_ln(model._get_atac_features(a))
        z = model.to_out(torch.cat([ar,features[split][rows].to(device)],1))
        return bce(model.ATAC(z),a)+nn.functional.mse_loss(model.RNA(z),r)
    train(model, paired_loss, unit/'source_teacher.pt', 'source_paired_reconstruction')
    model.eval().requires_grad_(False)
    targets = {}
    with torch.no_grad():
        for split in ('train','val'):
            targets[split] = torch.cat([model.atac_ln(model._get_atac_features(torch.as_tensor(data[f'atac_{split}'][i:i+config['batch_size']].toarray(),dtype=torch.float32,device=device))).cpu() for i in range(0,data[f'atac_{split}'].shape[0],config['batch_size'])])
    for part in (model.encoder,model.to_out,model.RNA,model.ATAC):
        part.requires_grad_(True)
    def student_loss(split, rows):
        r = aligned(data[f'rna_log_{split}'][rows],mapping,device)
        a = torch.as_tensor(data[f'atac_{split}'][rows].toarray(),dtype=torch.float32,device=device)
        predicted = model.encoder(r)
        z = model.to_out(torch.cat([predicted,features[split][rows].to(device)],1))
        return bce(model.ATAC(z),a)+nn.functional.mse_loss(model.RNA(z),r)+nn.functional.mse_loss(predicted,targets[split][rows].to(device))
    best = train(model, student_loss, unit/'best.pt', 'source_S3_distillation')
    # Export only modules required by official get_cell_embedding.
    state = torch.load(unit/'best.pt',map_location='cpu',weights_only=False)
    state['state_dict'] = {k:v for k,v in state['state_dict'].items() if not k.startswith(('atac_model.','atac_projection.','RNA.','ATAC.'))}
    state.update(method='scmomer',peaks=peaks,mapping=mapping,asset_sha256=hashes)
    torch.save(state,unit/'best.pt')
    frozen = make_model(peaks,device)
    frozen.load_state_dict(state['state_dict'],strict=True)
    return frozen, dict(best_epoch=best['best_epoch'],best_validation_loss=best['validation_loss'],trainable_parameters=params,
        mapped_hvg_genes=int((mapping>=0).sum()),pretrained_test_exposure='unknown; upstream training-cell manifest unavailable',
        adaptation='scMomer-S: frozen pretrained RNA; Source paired reconstruction teacher; official S3 losses; no label-conditioned adversarial pairs')


def predict(model, matrix, checkpoint, device, batch):
    output = []
    with torch.inference_mode():
        # Upstream Performer key-softmax stabilizes across the batch. A fixed single-cell
        # microbatch prevents another query cell from changing this cell's embedding.
        for i in range(matrix.shape[0]):
            r = aligned(matrix[i:i+1],checkpoint['mapping'],device)
            output.append(model.get_cell_embedding(r).cpu().numpy())
    return np.concatenate(output)
