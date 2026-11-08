from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, adjusted_mutual_info_score, adjusted_rand_score,
                             balanced_accuracy_score, f1_score, normalized_mutual_info_score)
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import run_goal2_strict_d_branches as branches

strict, ROOT, GOAL1 = branches.strict, branches.ROOT, branches.GOAL1
PREP = ROOT / 'result_contain/goal2_strict_preparation_20261002'
UPSTREAM = 'eb31e04bb8c4abdf85c4cbecd044fcd359105caa'
MODULES = ['RNA_encoder', 'ATAC_encoder', 'RNA_decoder', 'ATAC_decoder',
           'R_translator', 'A_translator', 'translator', 'discriminator_A', 'discriminator_R']
CONFIG = {'method': 'scButterfly-B', 'version': '0.0.9', 'upstream_commit': UPSTREAM,
          'run_tier': 'pre_final_user_declared_3seeds', 'strict_protocol_version': 'goal1_strict_lodo_v1',
          'primary_output': 'RNA -> predicted ATAC, upstream threshold values <1e-4 to zero',
          'R2R_pretrain_epoch': 100, 'A2A_pretrain_epoch': 100, 'translator_epoch': 200,
          'patience': 50, 'batch_size': 64, 'main_lr': 0.001, 'discriminator_lr': 0.005,
          'RNA_hidden': [256, 128], 'ATAC_hidden': [32, 128], 'translator_dim': 128,
          'no_augmentation': True, 'source_labels_used': False, 'target_atac_used': False,
          'target_labels_used_before_frozen_prediction': False,
          'EVALUATION_TIME_UNSUPERVISED_SCALING': True, 'KNOWN_K_EVALUATION': True,
          'cohort_caveat': 'Inherited fixed paired/QC cohort and label-stratified splits, as Goal1; strict model fitting/inference only.',
          'default_deviations': ['strict source-only LODO feature protocol', 'seed before model construction',
                                'independent best SOURCE-validation checkpoint retained', 'RNA-only inference instead of paired test routine',
                                'controlled StandardScaler+known-K KMeans evaluator', 'NumPy2 Inf/NaN compatibility aliases',
                                'source validation TF-IDF clipped to [0,1] for BCE; no validation-fitted scaling']}


def prepare():
    base = PREP / 'scbutterfly_cache'
    for fold in strict.parent.FOLDS:
        out = base / fold
        if (out / 'manifest.json').exists():
            strict.verify_cache(out)
            continue
        cm = strict.verify_cache(GOAL1 / 'cache' / fold)
        hroot = Path(cm['source_harmonized_root'])
        peak_manifest = branches.read(hroot / 'peak_harmonization_manifest.json')
        assert peak_manifest['target_excluded_from_feature_universe'] is True
        assert set(peak_manifest['feature_reference_datasets']) == set(cm['sources'])
        peak_array = np.load(hroot / 'canonical_peak_names.npy', allow_pickle=True)
        assert hashlib.sha256(peak_array.tobytes()).hexdigest() == peak_manifest['canonical_peak_names_sha256']
        peaks = peak_array.astype(str)
        rnas, atacs, train_ids, val_ids, provenance = [], [], [], [], dict(cm['source_files_sha256'])
        offset = 0
        for name in cm['sources']:
            source = strict.RNAOnlySource(hroot, name)
            ids = np.concatenate([source.train, source.val])
            ap = source.root / 'atac_binary.npz'
            atac = sparse.load_npz(ap).tocsr()[ids].astype(np.float32)
            atac.data[:] = 1
            assert atac.shape[1] == len(peaks)
            rnas.append(sparse.csr_matrix(np.asarray(source.rna[ids], dtype=np.float32)))
            atacs.append(atac)
            train_ids.extend(range(offset, offset + len(source.train)))
            val_ids.extend(range(offset + len(source.train), offset + len(ids)))
            offset += len(ids)
            for p in [ap, source.root / 'peak_mapping.npy', source.root / 'peak_mapping_manifest.json']:
                provenance[str(p)] = strict.sha(p)
        rna, atac = sparse.vstack(rnas, format='csr'), sparse.vstack(atacs, format='csr')
        train, val = np.asarray(train_ids), np.asarray(val_ids)
        keep = np.asarray(atac[train].sum(0)).ravel() >= np.ceil(0.005 * len(train))
        selected = peaks[keep]
        assert len(selected) > 0
        atac = atac[:, keep].tocsr()
        df = np.asarray(atac[train].sum(0)).ravel()
        idf = np.log1p(len(train) / np.maximum(df, 1)).astype(np.float32)
        tf = atac.multiply(1 / np.maximum(np.asarray(atac.sum(1)).ravel(), 1)[:, None])
        values = tf.multiply(idf).tocsr().astype(np.float32)
        scale = max(float(values[train].max()), 1e-8)
        values.data = np.clip(values.data / scale, 0, 1)
        chromosomes = np.asarray([p.split(':')[0] for p in selected])
        changes = np.r_[0, np.flatnonzero(chromosomes[1:] != chromosomes[:-1]) + 1, len(selected)]
        chrom_list = np.diff(changes).tolist()
        assert len(set(chromosomes)) == len(chrom_list)
        out.mkdir(parents=True)
        sparse.save_npz(out / 'source_rna.npz', rna)
        sparse.save_npz(out / 'source_atac.npz', values)
        np.save(out / 'train_index.npy', train)
        np.save(out / 'val_index.npy', val)
        np.save(out / 'peak_names.npy', selected)
        np.save(out / 'train_idf.npy', idf)
        provenance[str(hroot / 'canonical_peak_names.npy')] = strict.sha(hroot / 'canonical_peak_names.npy')
        strict.write(out / 'manifest.json', {'fold': fold, 'sources': cm['sources'], 'target': cm['target'],
                     'rna_genes': rna.shape[1], 'atac_peaks': values.shape[1], 'source_train_cells': len(train),
                     'source_val_cells': len(val), 'chrom_list': chrom_list, 'source_train_max_tfidf': scale,
                     'ATAC_rule': 'reuse deterministic SOURCE-only half-open overlapping interval merged union; adjacent not merged; per-cell binary OR, pooled SOURCE train prevalence>=ceil(0.005*N); TF-IDF train df and train max; validation clipped to BCE domain [0,1]',
                     'RNA_rule': 'reuse Goal1 SOURCE-defined log1p/library-normalized genes without target statistics',
                     'target_atac_used': False, 'labels_used': False, 'strict_protocol_version': CONFIG['strict_protocol_version'],
                     'source_files_sha256': provenance, 'cache_sha256': {p.name: strict.sha(p) for p in out.iterdir() if p.is_file()}})
        print('SCB_CACHE', fold, rna.shape, values.shape, flush=True)


def saved_state(model, epoch, stage, config_hash):
    return {'modules': {k: getattr(model, k).state_dict() for k in MODULES},
            'optimizer_state_dict': {k: v.state_dict() for k, v in vars(model).items() if k.startswith('optimizer_')},
            'scheduler_state_dict': None, 'epoch': epoch, 'stage': stage, 'config_hash': config_hash,
            'rng_state': {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch_cpu': torch.get_rng_state(),
                          'torch_cuda': torch.cuda.get_rng_state_all()}}


def run_unit(directory, fold, seed, smoke):
    import anndata as ad
    np.Inf, np.NaN = np.inf, np.nan
    sys.path.insert(0, str(ROOT / 'goal2_dependencies'))
    from scButterfly.butterfly import Butterfly
    import scButterfly.train_model as training
    cache = PREP / 'scbutterfly_cache' / fold
    cm = strict.verify_cache(cache)
    unit = directory / fold / f'seed_{seed}'
    if (unit / 'manifest.json').exists():
        prior = branches.read(unit / 'manifest.json')
        if prior['status'] == 'EMBEDDING_FROZEN':
            assert strict.sha(unit / 'predicted_atac.npy') == prior['prediction_sha256']
            return
        raise RuntimeError('Partial unit exists; preserve files and use declared retry/resume policy.')
    unit.mkdir(parents=True, exist_ok=True)
    config = {**CONFIG, 'fold': fold, 'seed': seed, 'smoke': smoke, 'runner_sha256': strict.sha(__file__)}
    strict.write(unit / 'config.json', config)
    strict.write(unit / 'data_manifest.json', cm)
    (unit / 'environment.txt').write_text(f'{platform.platform()}\nPython {sys.version}\nTorch {torch.__version__}\nNumPy {np.__version__}\nGPU {torch.cuda.get_device_name(0)}\n')
    rna = sparse.load_npz(cache / 'source_rna.npz')
    atac = sparse.load_npz(cache / 'source_atac.npz')
    train = np.load(cache / 'train_index.npy').tolist()
    val = np.load(cache / 'val_index.npy').tolist()
    if smoke:
        train, val = train[:128], val[:64]
    butterfly = Butterfly()
    butterfly.load_data(ad.AnnData(rna), ad.AnnData(atac), train, [], val)
    butterfly.RNA_data_p, butterfly.ATAC_data_p = butterfly.RNA_data, butterfly.ATAC_data
    butterfly.is_processed = True
    # Butterfly.__init__ resets the seed; restore the requested seed before construction.
    branches.seed_all(seed)
    butterfly.construct_model(cm['chrom_list'], R_encoder_dim_list=[256, 128], A_encoder_dim_list=[32, 128],
                              R_decoder_dim_list=[128, 256], A_decoder_dim_list=[128, 32], logging_path=str(unit))
    model = butterfly.model
    original = training.EarlyStopping
    history = []
    class AuditedStopping(original):
        def __call__(self, loss, current, path):
            assert np.isfinite(loss), 'Nonfinite validation loss'
            stage = next(k for k in ['R2R', 'A2A', 'all'] if getattr(current, 'early_stopping_' + k) is self)
            epoch = sum(r['stage'] == stage for r in history) + 1
            prior = self.best_score
            super().__call__(loss, current, path)
            state = saved_state(current, epoch, stage, strict.sha(unit / 'config.json'))
            assert all(torch.isfinite(p).all() for k in MODULES for p in getattr(current, k).parameters())
            if smoke:
                grads = [p.grad for k in MODULES for p in getattr(current, k).parameters() if p.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads)
            torch.save(state, unit / 'latest.pt')
            if prior is None or self.best_score != prior or self.counter == 0:
                torch.save(state, unit / f'best_{stage}.pt')
            history.append({'stage': stage, 'epoch': epoch, 'source_validation_reconstruction': float(loss),
                            'early_stopping_counter': self.counter})
            strict.write(unit / 'history.json', history)
            print('SCB_EPOCH', fold, seed, stage, epoch, float(loss), flush=True)
    training.EarlyStopping = AuditedStopping
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        butterfly.train_model(R2R_pretrain_epoch=1 if smoke else 100, A2A_pretrain_epoch=1 if smoke else 100,
                              translator_epoch=1 if smoke else 200, patience=50, batch_size=64,
                              loss_weight=[1, 2, 1], output_path=str(unit), logging_path=str(unit), seed=seed)
    finally:
        training.EarlyStopping = original
    training_seconds = time.perf_counter() - start
    best = torch.load(unit / 'best_all.pt', map_location='cpu', weights_only=False)
    for k in MODULES:
        getattr(model, k).load_state_dict(best['modules'][k])
    model.set_eval()
    target = np.load(GOAL1 / 'cache' / fold / 'target_rna_test.npy', mmap_mode='r')
    if smoke:
        target = target[:64]
    inference_start = time.perf_counter()
    prediction = np.lib.format.open_memmap(unit / 'predicted_atac.npy', mode='w+', dtype=np.float32,
                                         shape=(len(target), atac.shape[1]))
    with torch.no_grad():
        for start in range(0, len(target), 64):
            x = torch.from_numpy(np.array(target[start:start + 64], dtype=np.float32)).cuda()
            latent = model.RNA_encoder(x)
            _, translated, _, _ = model.translator.test_model(latent, 'RNA')
            y = model.ATAC_decoder(translated)
            assert torch.isfinite(y).all()
            y[y < 1e-4] = 0
            prediction[start:start + len(x)] = y.cpu().numpy()
    prediction.flush()
    inference_seconds = time.perf_counter() - inference_start
    strict.write(unit / 'manifest.json', {'status': 'EMBEDDING_FROZEN', **config, 'best_epoch': best['epoch'],
                 'best_checkpoint_sha256': strict.sha(unit / 'best_all.pt'), 'prediction_sha256': strict.sha(unit / 'predicted_atac.npy'),
                 'source_manifest_sha256': strict.sha(cache / 'manifest.json'), 'target_rna_sha256': strict.sha(GOAL1 / 'cache' / fold / 'target_rna_test.npy'),
                 'resources': {'training_seconds': training_seconds, 'inference_seconds': inference_seconds,
                               'total_wall_seconds': training_seconds + inference_seconds, 'cells_per_second': len(target) / inference_seconds,
                               'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
                               'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved(), 'GPU': torch.cuda.get_device_name(0),
                               'trainable_parameters': sum(p.numel() for k in MODULES for p in getattr(model, k).parameters() if p.requires_grad)}})
    print('SCB_FROZEN', fold, seed, 'smoke', smoke, flush=True)


def evaluate(directory):
    rows = []
    for fold in strict.parent.FOLDS:
        cm = branches.read(GOAL1 / 'cache' / fold / 'manifest.json')
        raw = Path(cm['raw_target_source'])
        processed = raw.parent if raw.suffix == '.npz' else raw.parent.parent / 'processed'
        test = np.load(GOAL1 / 'cache' / fold / 'target_test_index.npy')
        for seed in strict.SEEDS:
            unit = directory / fold / f'seed_{seed}'
            manifest = branches.read(unit / 'manifest.json')
            assert strict.sha(unit / 'predicted_atac.npy') == manifest['prediction_sha256']
            with (processed / 'label.csv').open(newline='') as handle:
                labels = np.asarray([r['cell_type'] for r in csv.DictReader(handle)])[test]
            _, labels = np.unique(labels, return_inverse=True)
            x = np.load(unit / 'predicted_atac.npy', mmap_mode='r')
            clusters = KMeans(n_clusters=len(np.unique(labels)), n_init=20, random_state=seed).fit_predict(StandardScaler().fit_transform(x))
            table = np.zeros((len(np.unique(labels)), len(np.unique(labels))), dtype=np.int64)
            np.add.at(table, (labels, clusters), 1)
            a, b = linear_sum_assignment(-table)
            mapping = dict(zip(b, a))
            matched = np.asarray([mapping[k] for k in clusters])
            metrics = {'ari': float(adjusted_rand_score(labels, clusters)), 'nmi': float(normalized_mutual_info_score(labels, clusters)),
                       'ami': float(adjusted_mutual_info_score(labels, clusters)), 'cluster_acc': float(accuracy_score(labels, matched)),
                       'accuracy': float(accuracy_score(labels, matched)), 'balanced_accuracy': float(balanced_accuracy_score(labels, matched)),
                       'macro_f1': float(f1_score(labels, matched, average='macro')), 'weighted_f1': float(f1_score(labels, matched, average='weighted')),
                       'annotation_score_scope': 'posthoc target Hungarian-mapped clusters, NOT source-supervised label-transfer accuracy',
                       'target_atac_used': False, 'labels_loaded_after_frozen_prediction': True,
                       'EVALUATION_TIME_UNSUPERVISED_SCALING': True, 'KNOWN_K_EVALUATION': True}
            strict.write(unit / 'metrics.json', metrics)
            rows.append({'fold': fold, 'seed': seed, **metrics})
            print('SCB_METRICS', fold, seed, metrics['ari'], metrics['nmi'], flush=True)
    strict.write(directory / 'RESULTS.json', rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'smoke', 'formal', 'evaluate'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == 'prepare':
        prepare()
        return
    if args.stage == 'smoke':
        with branches.gpu_lock():
            run_unit(PREP / 'scbutterfly_smoke', 'target_D1', 20020130, True)
        strict.write(PREP / 'scbutterfly_smoke/SMOKE_GATE.json', {'status': 'PASS', 'forward_backward_finite': True,
                     'best_latest_checkpoint_saved': True, 'RNA_only_target_inference': True, 'target_atac_used': False, 'labels_used': False})
        return
    assert branches.read(PREP / 'scbutterfly_smoke/SMOKE_GATE.json')['status'] == 'PASS'
    assert branches.read(Path(branches.read(PREP / 'TASK_ASSIGNMENTS.json')['capacity']['result_path']) / 'ADJUDICATION_seed_20020130.json')['CAPACITY_BRANCH_PROMISING'] == 'NO'
    assert branches.read(PREP / 'ot_gate/GW_SOURCE_GATE_ADJUDICATION.json')['OT_BRANCH_READY'] == 'NO'
    directory, task_id = branches.reserve(PREP, 'scbutterfly')
    strict.write(directory / 'TASK.json', {'task_id': task_id, **CONFIG, 'seeds': strict.SEEDS})
    if args.stage == 'formal':
        with branches.gpu_lock():
            for seed in strict.SEEDS:
                for fold in strict.parent.FOLDS:
                    run_unit(directory, fold, seed, False)
                    torch.cuda.empty_cache()
    else:
        evaluate(directory)


if __name__ == '__main__':
    main()
