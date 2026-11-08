"""Standalone frozen inference; reads only RNA NPZ and its ordered gene IDs."""
import argparse
import os
from pathlib import Path

import numpy as np
from scipy import sparse

from neural_s_mode_intra import adapter
from pbmc_intra_common import sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--rna-log',type=Path,required=True,help='Full-library 1e4 natural-log1p, then frozen HVG columns')
    p.add_argument('--gene-ids',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a = p.parse_args()
    if a.output.exists():
        raise RuntimeError('Existing inference output preserved')
    if a.device=='cpu':
        os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(2)
    digest = sha(a.checkpoint)
    state = torch.load(a.checkpoint,map_location=a.device,weights_only=False)
    np.testing.assert_array_equal(np.load(a.gene_ids),np.load(a.checkpoint.parent/'gene_ids.npy'))
    module = adapter(state['method'])
    if state['method']=='scmomer':
        for path, expected in state['asset_sha256'].items():
            if sha(path)!=expected:
                raise ValueError(f'Changed scMomer reference asset: {path}')
    model = module.make_model(state['peaks'],a.device)
    model.load_state_dict(state['state_dict'],strict=True)
    model.eval().requires_grad_(False)
    z = module.predict(model,sparse.load_npz(a.rna_log).tocsr(),state,a.device,a.batch_size)
    if not np.isfinite(z).all() or digest!=sha(a.checkpoint):
        raise ValueError('Invalid frozen inference')
    np.save(a.output,z)


if __name__=='__main__':
    main()
