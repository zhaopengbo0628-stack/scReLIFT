"""Inference from a saved unit; this entry point never loads benchmark ATAC/labels."""
import argparse
import os
from pathlib import Path

import numpy as np
from scipy import sparse

from pbmc_intra_common import read, sha
from s_mode_intra import make_model, predict


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit',type=Path,required=True)
    parser.add_argument('--rna',type=Path,required=True,
        help='CSR NPZ: selected-gene raw counts for VAEs; full-library 1e4/log1p selected genes for Palette-S')
    parser.add_argument('--gene-ids',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    if args.device=='cpu':
        os.environ['CUDA_VISIBLE_DEVICES']=''
    if args.output.suffix!='.npy':
        parser.error('--output must end with .npy')
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest=read(args.unit/'manifest.json')
    np.testing.assert_array_equal(np.load(args.gene_ids),np.load(args.unit/'gene_ids.npy'))
    x=sparse.load_npz(args.rna).tocsr().astype(np.float32)
    assert x.shape[0]>0 and x.shape[1]==manifest['input_dimensions'][0]
    assert np.isfinite(x.data).all() and (x.data>=0).all()
    if manifest['method']=='Palette-S':
        checkpoint=args.unit/'frozen_map.npz'
        assert sha(checkpoint)==manifest['frozen_map_sha256']
        z=x@np.load(checkpoint)['weights']
    else:
        import torch
        torch.set_num_threads(2)
        assert (x.data==np.floor(x.data)).all(), 'VAEs require raw counts'
        checkpoint=args.unit/'best.pt'
        assert sha(checkpoint)==manifest['checkpoint_sha256']
        saved=torch.load(checkpoint,map_location=args.device,weights_only=False)
        model=make_model(saved['method'],saved['dims'],saved['latent_dim'],args.device)
        model.load_state_dict(saved['state_dict'])
        model.eval().requires_grad_(False)
        z=predict(model,saved['method'],x,saved['dims'][1],args.device,256)
    assert np.isfinite(z).all()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.save(args.output,z.astype(np.float32))
