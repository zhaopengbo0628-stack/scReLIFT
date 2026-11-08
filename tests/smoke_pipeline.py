"""Integration checks using synthetic data; never a scientific benchmark."""
from pathlib import Path
import argparse
import ast
import copy
import json
import tempfile
import unittest

import numpy as np
from scipy import sparse
import torch
from threadpoolctl import threadpool_limits

from screlift import workflow as w
from screlift import _frozen_core as core
from screlift.preprocessing import map_rna

ROOT = Path(__file__).resolve().parents[1]

class PipelineChecks(unittest.TestCase):
    def test_extraction_matches_original(self):
        original = ast.parse((ROOT / 'paper_snapshot/main/core/run_b2_strict_lodo.py').read_text(encoding='utf-8-sig'))
        derived = ast.parse((ROOT / 'src/screlift/_frozen_core.py').read_text(encoding='utf-8'))
        sources = {n.name: n for n in original.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for node in derived.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                self.assertEqual(ast.dump(node), ast.dump(sources[node.name]), node.name)

    def test_target_changes_cannot_refit_source(self):
        counts = sparse.csr_matrix([[1, 2, 3], [3, 0, 2]], dtype=np.float32)
        genes = np.array(['ENSG1', 'ENSG2', 'ENSG3'])
        wanted = np.array(['ENSG1', 'ENSG2', 'ENSG9'])
        mean, std = np.array([1., 2., 3.], np.float32), np.array([2., 3., 4.], np.float32)
        before = mean.copy(), std.copy()
        x, hit = map_rna(counts, genes, wanted, mean, std)
        bigger = sparse.vstack([counts, counts * 100])
        repeated, _ = map_rna(bigger, genes, wanted, mean, std)
        np.testing.assert_array_equal(x, repeated[:2])
        np.testing.assert_array_equal(mean, before[0])
        np.testing.assert_array_equal(std, before[1])
        np.testing.assert_array_equal(hit, [True, True, False])
        np.testing.assert_allclose(x[:, 2], -mean[2] / (std[2] + 1e-6))

    def test_summary_enforces_seed_and_fold_policy(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            paths = []
            for fold, offset in [('D1', 0.), ('D2', 2.)]:
                for seed, score in zip(w.SEEDS, [1., 2., 3.]):
                    path = td / f'{fold}_{seed}.json'
                    w.write(path, dict(model='scReLIFT', mode='D', fold=fold, seed=seed,
                                      training_status='TRAINING_COMPLETE', metrics={'ari': score + offset}))
                    paths.append(path)
            result = w.summarize(paths, td / 'summary.json')
            macro = result['folds_and_macro']['equal_fold_macro']
            self.assertEqual(macro['mean']['ari'], 3.)
            self.assertEqual(macro['sample_sd']['ari'], 1.)
            with self.assertRaises(ValueError):
                w.summarize(paths[:-1], td / 'bad.json')
            with self.assertRaises(ValueError):
                w.summarize(paths + paths[:1], td / 'bad.json')

    def test_frozen_network_backward(self):
        core.seed_all(w.SEEDS[0])
        teacher = core.Teacher(12, 12)
        x, field = torch.randn(8, 12), torch.randn(8, 12)
        losses = core.teacher_losses(teacher, x, field)
        sum(losses).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in teacher.parameters() if p.grad is not None))
        teacher.eval().requires_grad_(False)
        student = copy.deepcopy(teacher.rna_encoder).requires_grad_(True)
        target = teacher(x, field)[2].detach()
        torch.nn.functional.mse_loss(student(x), target).backward()
        self.assertEqual(student(x).shape, (8, 64))

def synthetic_data(base):
    rng = np.random.default_rng(20020130)
    n, g, p = 96, 24, 40
    latent = rng.normal(size=(n, 4))
    rates = np.exp(np.clip(latent @ rng.normal(size=(4, g)) * .4, -2, 2)) * 3
    counts = rng.poisson(rates).astype(np.float32)
    accessibility = latent @ rng.normal(size=(4, p)) + rng.normal(size=(n, p)) * .3
    atac = (accessibility > np.median(accessibility, axis=0)).astype(np.float32)
    genes = np.array([f'ENSG{i:011d}' for i in range(g)])
    peaks = np.array([f'chr1:{100000 + i*2000}-{100500 + i*2000}' for i in range(p)])
    sparse.save_npz(base / 'rna_counts.npz', sparse.csr_matrix(counts))
    sparse.save_npz(base / 'atac_counts.npz', sparse.csr_matrix(atac))
    np.save(base / 'gene_ids.npy', genes)
    np.save(base / 'peak_names.npy', peaks)
    np.save(base / 'cell_ids.npy', np.array([f'cell_{i}' for i in range(n)]))
    np.save(base / 'labels.npy', (np.arange(n) % 3).astype(str))
    np.savez(base / 'metadata.npz', train_index=np.arange(64), val_index=np.arange(64, 80), test_index=np.arange(80, n))
    lines = [f'chr1\tsynthetic\tgene\t{120000+i*1000}\t{120100+i*1000}\t.\t+\t.\tgene_id "{gene}";' for i, gene in enumerate(genes)]
    (base / 'genes.gtf').write_text('\n'.join(lines) + '\n')
    cfg = dict(source='synthetic', rna_counts='rna_counts.npz', gene_ids='gene_ids.npy',
               atac_counts='atac_counts.npz', peak_names='peak_names.npy', splits='metadata.npz',
               gtf='genes.gtf', hvg_count=12, smoke_only=True)
    w.write(base / 's.json', cfg)
    return counts, genes, atac, peaks

def integration(base, cuda=False):
    base.mkdir(parents=True, exist_ok=False)
    counts, genes, atac, peaks = synthetic_data(base)
    scache = w.prepare_s(base / 's.json', base / 'cache_s')
    sources = []
    for i in range(3):
        folder = base / f'source_{i}'
        folder.mkdir()
        order = np.roll(np.arange(len(genes)), i)
        np.save(folder / 'rna.npy', w.normalize_counts(sparse.csr_matrix(counts))[:, order].toarray())
        np.save(folder / 'gene_ids.npy', genes[order])
        np.save(folder / 'peak_names.npy', peaks)
        sparse.save_npz(folder / 'atac_binary.npz', sparse.csr_matrix(atac))
        np.savez(folder / 'metadata.npz', train_index=np.arange(64), val_index=np.arange(64, 80), test_index=np.arange(80, 96))
        sources.append(dict(name=f'source_{i}', prepared_dir=f'source_{i}'))
    w.write(base / 'd.json', dict(target='held_out', sources=sources, gtf='genes.gtf', smoke_only=True))
    dcache = w.prepare_d(base / 'd.json', base / 'cache_d')
    result = dict(status='PASS', synthetic_only=True, cuda_training=cuda, modes={})
    for mode, cache in [('S', scache), ('D', dcache)]:
        sources, mean, std, manifest = w.load_cache(cache)
        assert len(sources) == (1 if mode == 'S' else 3)
        assert np.isfinite(mean).all() and np.isfinite(std).all()
        if cuda:
            bundle = w.train(cache, base / f'model_{mode}', w.SEEDS[0], smoke=True)
            # Resume must re-use exact checkpoints, without another optimizer step.
            before = core.sha(bundle / 'student/best.pt')
            w.train(cache, bundle, w.SEEDS[0], smoke=True)
            assert before == core.sha(bundle / 'student/best.pt')
            inf = w.infer(bundle, base / 'rna_counts.npz', base / 'gene_ids.npy',
                          base / f'inference_{mode}', base / 'cell_ids.npy')
            metrics = w.evaluate(inf, base / 'labels.npy', base / 'cell_ids.npy', base / f'metrics_{mode}.json', 'synthetic')
            assert np.isfinite(metrics['metrics']['ari'])
            result['modes'][mode] = dict(preparation=True, training=True, resume=True, inference=True, scoring=True)
        else:
            result['modes'][mode] = dict(preparation=True, training=False)
    # Mutating a source cache must be rejected before any training starts.
    with (scache / 'gene_ids.npy').open('ab') as handle:
        handle.write(b'tamper')
    try:
        w.load_cache(scache)
        raise AssertionError('Changed cache was not rejected')
    except ValueError:
        result['tamper_rejected'] = True
    w.write(base / 'integration_result.json', result)
    print(json.dumps(result))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--integration-output', type=Path)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(PipelineChecks)
        checked = unittest.TextTestRunner(verbosity=2).run(suite)
        if not checked.wasSuccessful():
            raise SystemExit(1)
        if args.integration_output:
            integration(args.integration_output, args.cuda)
