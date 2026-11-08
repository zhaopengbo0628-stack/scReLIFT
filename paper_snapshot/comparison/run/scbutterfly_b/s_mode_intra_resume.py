"""Resume audited scButterfly Source units and score intra only."""
import argparse
import fcntl
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from pbmc_intra_common import ROOT, SEEDS, prepare, read, write, sha, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['D1', 'D2', 'D3', 'D4'])
    parser.add_argument('--seed', required=True, type=int, choices=SEEDS[:3])
    parser.add_argument('--output-root', required=True, type=Path)
    args = parser.parse_args()
    cache = prepare(args.dataset)
    previous = ROOT/'result_contain/49_smode_scbutterfly_corrected_pbmc_3seed_20261012'
    runner = previous/'source/s_mode_corrected_pbmc_v2.py'
    spec = importlib.util.spec_from_file_location('audited_scb_intra', runner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import torch
    torch.set_num_threads(2)
    np.Inf, np.NaN = np.inf, np.nan
    module.sc.PREP = previous
    module.sc.GOAL1 = previous
    unit = args.output_root/args.dataset/f'seed_{args.seed}'
    with (ROOT/'.benchmark_a30.lock').open('a') as lock:
        print('Waiting for benchmark GPU lock', flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (unit/'training_manifest.json').exists():
            module.sc.run_unit(args.output_root, args.dataset, args.seed, False)
            training_manifest = read(unit/'manifest.json')
            assert training_manifest['status'] == 'EMBEDDING_FROZEN'
            assert sha(unit/'best_all.pt') == training_manifest['best_checkpoint_sha256']
            assert sha(unit/'predicted_atac.npy') == training_manifest['prediction_sha256']
            write(unit/'training_manifest.json', training_manifest)
        else:
            training_manifest = read(unit/'training_manifest.json')
        prediction = unit/'predicted_atac.npy'
        assert sha(prediction) == training_manifest['prediction_sha256']
        if not (unit/'rna_test_embedding.npy').exists():
            os.link(prediction, unit/'rna_test_embedding.npy')
        np.testing.assert_array_equal(np.load(cache/'gene_ids.npy'), np.load(previous/'scbutterfly_cache'/args.dataset/'gene_ids.npy'))
        for split in ['train', 'val', 'test']:
            np.save(unit/f'{split}_index.npy', np.load(cache/f'{split}_index.npy'))
        np.save(unit/'gene_ids.npy', np.load(cache/'gene_ids.npy'))
        resources = training_manifest['resources']
        history = [r for r in read(unit/'history.json') if r['stage']=='all']
        best = next(r for r in history if r['epoch']==training_manifest['best_epoch'])
        assert best['source_validation_reconstruction'] == min(r['source_validation_reconstruction'] for r in history)
        result = dict(resources, status='TRAINED', method='scButterfly-B-S', dataset=args.dataset,
                      seed=args.seed, configured_seeds=SEEDS[:3], seed_count=3, run_tier='pre_final',
                      gpu_model=resources['GPU'], checkpoint_file='best_all.pt',
                      checkpoint_sha256=sha(unit/'best_all.pt'),
                      prediction_sha256=sha(prediction), best_epoch=training_manifest['best_epoch'],
                      best_validation_loss=best['source_validation_reconstruction'],
                      source_cache=str(previous/'scbutterfly_cache'/args.dataset),
                      total_time_scope='retained training + intra inference; excludes queue, scoring and historical cache build',
                      representation='predicted ATAC from frozen RNA translator; retained task49 representation',
                      test_used_for_fitting=False, test_atac_loaded=False, labels_used_for_fitting=False)
        write(unit/'manifest.json', result)
        evaluate(unit, cache, args.dataset)
        result['status'] = 'COMPLETE'
        write(unit/'manifest.json', result)
    print('COMPLETE scButterfly-B-S', args.dataset, args.seed, flush=True)


if __name__ == '__main__':
    main()
