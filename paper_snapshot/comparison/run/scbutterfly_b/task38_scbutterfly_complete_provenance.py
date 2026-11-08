import csv
import hashlib
import json
import os
import platform
import sys
import types
from pathlib import Path

import numpy as np
import torch
import anndata as ad
from scipy import sparse

# The frozen-model provenance pass does not call preprocessing functions.
scanpy_stub = types.ModuleType('scanpy')
episcanpy_stub = types.ModuleType('episcanpy')
episcanpy_api_stub = types.ModuleType('episcanpy.api')
episcanpy_stub.api = episcanpy_api_stub
adjust_text_stub = types.ModuleType('adjustText')
adjust_text_stub.adjust_text = lambda *args, **kwargs: None
sys.modules.setdefault('scanpy', scanpy_stub)
sys.modules.setdefault('episcanpy', episcanpy_stub)
sys.modules.setdefault('episcanpy.api', episcanpy_api_stub)
sys.modules.setdefault('adjustText', adjust_text_stub)

ROOT = Path('/home/u202522802045/share/zpb')
SRC = ROOT / 'task38_corrected_pbmc002_ablation'
OUT = ROOT / 'final_scbutterfly_corrected_pbmc002'
sys.path.insert(0, str(ROOT / 'dmode_dev/experiments/v5d'))
sys.path.insert(0, str(ROOT / 'result_contain/34_strict_d_scbutterfly_goal2_20261002/source'))
import run_goal2_strict_scbutterfly as runner


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def ids_for(cache, manifest):
    ids = []
    source_files = manifest['source_files_sha256']
    first_rna = next(Path(path) for path in source_files if path.endswith('/rna.npy'))
    source_root = first_rna.parents[1]
    for name in manifest['sources']:
        source = runner.strict.RNAOnlySource(source_root, name)
        ids.extend([f'{name}:source_row_{i}' for i in range(len(source.train) + len(source.val))])
    return ids


def infer(model, x, device):
    out = None
    with torch.no_grad():
        for start in range(0, len(x), 64):
            batch = torch.from_numpy(np.asarray(x[start:start + 64], dtype=np.float32)).to(device)
            latent = model.RNA_encoder(batch)
            _, translated, _, _ = model.translator.test_model(latent, 'RNA')
            y = model.ATAC_decoder(translated)
            if out is None:
                out = np.empty((len(x), y.shape[1]), dtype=np.float32)
            y[y < 1e-4] = 0
            out[start:start + len(batch)] = y.cpu().numpy()
    return out


def main():
    device = torch.device('cpu')
    runner.np.Inf, runner.np.NaN = np.inf, np.nan
    sys.path.insert(0, str(ROOT / 'goal2_dependencies'))
    from scButterfly.butterfly import Butterfly
    for fold in runner.strict.parent.FOLDS:
        cache = SRC / 'scbutterfly_cache' / fold
        cm = runner.strict.verify_cache(cache)
        cache_manifest = json.loads((cache / 'manifest.json').read_text())
        for seed in runner.strict.SEEDS:
            unit = OUT / fold / f'seed_{seed}'
            source = SRC / 'scbutterfly' / fold / f'seed_{seed}'
            ckpt = source / 'best_all.pt'
            assert ckpt.exists() and (source / 'manifest.json').exists()
            rna = sparse.load_npz(cache / 'source_rna.npz')
            train = np.load(cache / 'train_index.npy')
            val = np.load(cache / 'val_index.npy')
            butterfly = Butterfly()
            butterfly.load_data(ad.AnnData(rna), ad.AnnData(sparse.load_npz(cache / 'source_atac.npz')), train.tolist(), [], val.tolist())
            butterfly.RNA_data_p, butterfly.ATAC_data_p = butterfly.RNA_data, butterfly.ATAC_data
            runner.branches.seed_all(seed)
            butterfly.construct_model(cm['chrom_list'], R_encoder_dim_list=[256, 128], A_encoder_dim_list=[32, 128], R_decoder_dim_list=[128, 256], A_decoder_dim_list=[128, 32], logging_path=str(unit))
            state = torch.load(ckpt, map_location='cpu', weights_only=False)
            for key in runner.MODULES:
                getattr(butterfly.model, key).load_state_dict(state['modules'][key])
            butterfly.model.set_eval()
            train_pred = infer(butterfly.model, rna[train].toarray(), device)
            val_pred = infer(butterfly.model, rna[val].toarray(), device)
            emb = unit / 'embeddings'
            emb.mkdir(exist_ok=True)
            np.save(emb / 'source_train_embedding_raw.npy', train_pred)
            np.save(emb / 'source_val_embedding_raw.npy', val_pred)
            ids = ids_for(cache, cache_manifest)
            assert len(ids) == rna.shape[0]
            with (emb / 'source_train_cells.tsv').open('w', newline='', encoding='utf-8') as f:
                w = csv.writer(f, delimiter='\t'); w.writerow(['cell_id', 'split']); w.writerows((ids[i], 'train') for i in train)
            with (emb / 'source_val_cells.tsv').open('w', newline='', encoding='utf-8') as f:
                w = csv.writer(f, delimiter='\t'); w.writerow(['cell_id', 'split']); w.writerows((ids[i], 'val') for i in val)
            with (emb / 'target_cells.tsv').open('w', newline='', encoding='utf-8') as f:
                w = csv.writer(f, delimiter='\t'); w.writerow(['cell_id', 'split']); w.writerows((f'target_row_{i}', 'target') for i in range(len(np.load(SRC / 'cache' / fold / 'target_rna_test.npy', mmap_mode='r'))))
            manifest = json.loads((unit / 'manifest.json').read_text())
            manifest['source_embedding_provenance'] = {
                'generated_from_frozen_best_all': str(ckpt),
                'checkpoint_sha256': sha(ckpt),
                'source_cache_manifest_sha256': sha(cache / 'manifest.json'),
                'source_train_embedding_raw_sha256': sha(emb / 'source_train_embedding_raw.npy'),
                'source_val_embedding_raw_sha256': sha(emb / 'source_val_embedding_raw.npy'),
                'source_train_cells_sha256': sha(emb / 'source_train_cells.tsv'),
                'source_val_cells_sha256': sha(emb / 'source_val_cells.tsv'),
                'target_cells_sha256': sha(emb / 'target_cells.tsv'),
                'source_only_inference': True,
                'target_atac_used': False,
                'target_labels_used_before_embedding_freeze': False,
            }
            write(unit / 'checkpoint_manifest.json', {'best_all.pt': sha(ckpt), 'best_R2R.pt': sha(source / 'best_R2R.pt'), 'best_A2A.pt': sha(source / 'best_A2A.pt'), 'latest.pt': sha(source / 'latest.pt')})
            write(unit / 'leakage_audit.json', {'target_atac_used': False, 'target_labels_used_before_embedding_freeze': False, 'target_statistics_used': False, 'target_in_feature_construction': False, 'source_cache_manifest_sha256': sha(cache / 'manifest.json')})
            write(unit / 'embedding_manifest.json', {'source_train_embedding_raw_sha256': sha(emb / 'source_train_embedding_raw.npy'), 'source_val_embedding_raw_sha256': sha(emb / 'source_val_embedding_raw.npy'), 'target_embedding_raw_sha256': sha(unit / 'predicted_atac.npy'), 'target_embedding_standardized_sha256': sha(unit / 'target_embedding_standardized.npy'), 'cell_order_files': ['source_train_cells.tsv', 'source_val_cells.tsv', 'target_cells.tsv']})
            write(unit / 'manifest.json', manifest)
            print('SCB_PROVENANCE', fold, seed, flush=True)


if __name__ == '__main__':
    main()
