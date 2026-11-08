#!/usr/bin/env python
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
from scipy import sparse
ROOT = Path('/home/u202522802045/share/zpb')
RUN = ROOT/'comparison/run'
sys.path.insert(0, str(RUN))
from bmmc_intra_common import prepare, load, write, sha, evaluate
SEEDS = [20020130, 20020131, 20020132]
DATASETS = ['BMMC_Site1','BMMC_Site2','BMMC_Site3','BMMC_Site4']
def make_adata(rna, atac, cells, genes, peaks):
    import anndata as ad
    feature_names = np.concatenate([genes.astype(str), np.char.add('ATAC_', peaks.astype(str))])
    x = sparse.hstack([rna.astype(np.float32), atac.astype(np.float32)], format='csr')
    a = ad.AnnData(x, obs={'cell': cells.astype(str)}, var={'feature': feature_names})
    a.obs_names = cells.astype(str); a.var_names = feature_names
    return a, feature_names
def rna_query(rna, cells, genes, peaks):
    import anndata as ad
    feature_names = np.concatenate([genes.astype(str), np.char.add('ATAC_', peaks.astype(str))])
    zeros = sparse.csr_matrix((rna.shape[0], peaks.size), dtype=np.float32)
    x = sparse.hstack([rna.astype(np.float32), zeros], format='csr')
    a = ad.AnnData(x, obs={'cell': cells.astype(str)}, var={'feature': feature_names})
    a.obs_names = cells.astype(str); a.var_names = feature_names
    return a
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True, choices=['smoke','unit'])
    p.add_argument('--dataset', required=True, choices=DATASETS)
    p.add_argument('--seed', required=True, type=int, choices=SEEDS)
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--output-root', type=Path, default=ROOT/'comparison/result/BMMC_intra_multivi')
    p.add_argument('--epochs', type=int, default=200)
    args = p.parse_args()
    if args.device == 'cpu': os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    import scvi
    from scvi.model import MULTIVI
    torch.set_num_threads(2); scvi.settings.seed = args.seed
    cache = prepare(args.dataset)
    unit = args.output_root/('smoke' if args.stage == 'smoke' else 'pre_final')/args.dataset/f'seed_{args.seed}'
    if unit.exists(): raise RuntimeError(f'Existing unit preserved: {unit}')
    unit.mkdir(parents=True); lock = None
    try:
        if args.device == 'cuda':
            import fcntl
            lock = (ROOT/'.benchmark_a30.lock').open('a')
            print('Waiting for benchmark GPU lock', flush=True); fcntl.flock(lock, fcntl.LOCK_EX)
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        data, provenance = load(cache, smoke=(args.stage == 'smoke'))
        genes = np.load(cache/'gene_ids.npy', allow_pickle=True)
        peaks = np.load(cache/'peak_ids.npy', allow_pickle=True)
        train_n, val_n, test_n = data['rna_train'].shape[0], data['rna_val'].shape[0], data['rna_test'].shape[0]
        source_rna = sparse.vstack([data['rna_val'], data['rna_train']], format='csr')
        source_atac = sparse.vstack([data['atac_val'], data['atac_train']], format='csr')
        source_cells = np.asarray([f'{args.dataset}_source_{i}' for i in range(source_rna.shape[0])], dtype=str)
        fit, feature_names = make_adata(source_rna, source_atac, source_cells, genes, peaks)
        MULTIVI.setup_anndata(fit)
        model = MULTIVI(fit, n_genes=int(genes.size), n_regions=int(peaks.size), n_latent=64, fully_paired=False)
        fit_started = time.perf_counter(); epochs = 1 if args.stage == 'smoke' else args.epochs
        model.train(max_epochs=epochs, lr=1e-4, accelerator='gpu' if args.device == 'cuda' else 'cpu',
                    devices=1, train_size=train_n/source_rna.shape[0], validation_size=val_n/source_rna.shape[0],
                    shuffle_set_split=False, batch_size=256, early_stopping=(args.stage != 'smoke'),
                    save_best=True)
        if args.device == 'cuda': torch.cuda.synchronize()
        training_seconds = time.perf_counter() - fit_started
        model.save(unit/'model', overwrite=True)
        test_cells = np.asarray([f'{args.dataset}_test_{i}' for i in range(test_n)], dtype=str)
        query = rna_query(data['rna_test'], test_cells, genes, peaks)
        MULTIVI.setup_anndata(query)
        inf_started = time.perf_counter()
        z = model.get_latent_representation(query, modality='expression').astype(np.float32)
        if args.device == 'cuda': torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inf_started
        if not np.isfinite(z).all(): raise ValueError('Nonfinite MultiVI target embedding')
        np.save(unit/'rna_test_embedding.npy', z)
        model_dir = unit/'model'
        manifest = dict(status='SMOKE_PASS' if args.stage == 'smoke' else 'TRAINED',
            run_tier='temporary_smoke' if args.stage == 'smoke' else 'pre_final_explicit_3seeds',
            method='MultiVI', family='s_mode_intra', dataset=args.dataset, seed=args.seed,
            configured_seeds=SEEDS, seed_count=3, epochs=epochs, latent_dim=64, batch_size=256,
            learning_rate=1e-4, train_cells=int(train_n), validation_cells=int(val_n), test_cells=int(test_n),
            input_dimensions=[int(genes.size), int(peaks.size)],
            source_training='paired RNA+ATAC from source train/val only',
            target_input='RNA-only; zero ATAC placeholder for MultiVI expression modality',
            test_atac_loaded=False, labels_used_for_fitting=False, test_used_for_fitting=False,
            target_feature_selection=False, target_alignment=False, training_seconds=training_seconds,
            inference_seconds=inference_seconds, total_wall_seconds=time.perf_counter()-started,
            cells_per_second=float(test_n/max(inference_seconds,1e-12)), checkpoint=str(model_dir),
            checkpoint_sha256=sha(model_dir/'model.pt') if (model_dir/'model.pt').exists() else None,
            prediction_sha256=sha(unit/'rna_test_embedding.npy'),
            trainable_parameters=int(sum(p.numel() for p in model.module.parameters() if p.requires_grad)),
            inference_parameters=int(sum(p.numel() for p in model.module.parameters())),
            gpu_model=torch.cuda.get_device_name() if args.device == 'cuda' else None,
            peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated()) if args.device == 'cuda' else None,
            peak_gpu_reserved_bytes=int(torch.cuda.max_memory_reserved()) if args.device == 'cuda' else None,
            cache_manifest_sha256=sha(cache/'manifest.json'),
            runtime_versions={'torch':torch.__version__,'scvi-tools':scvi.__version__},
            command=sys.argv)
        write(unit/'manifest.json', manifest)
        if args.stage == 'unit':
            evaluate(unit, cache, args.dataset); manifest['status'] = 'COMPLETE'; write(unit/'manifest.json', manifest)
        print(manifest['status'], args.dataset, args.seed, flush=True)
    except Exception as e:
        write(unit/'failure.json', {'status':'FAILED','error':repr(e),'command':sys.argv}); raise
    finally:
        if lock is not None: lock.close()
if __name__ == '__main__': main()
