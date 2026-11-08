import unittest
import numpy as np
from scipy import sparse
from scipy.stats import spearmanr

from main.core.regulatory.s_mode.single_source import (
    build_bs, build_field, canonical_genes, graph_weights, map_rna,
    normalize_counts, select_hvg)
from v5d.dmode_reliable_selector import _resamples, DModeReliabilityConfig


class SingleSourceTests(unittest.TestCase):
    def test_full_library_and_gene_merge(self):
        raw = sparse.csr_matrix([[1., 2., 7.], [2., 0., 8.]])
        counts, ids, inverse = canonical_genes(raw, ['ENSG1.1', 'ENSG1.2', 'ENSG2'])
        self.assertEqual(ids.tolist(), ['ENSG1', 'ENSG2'])
        np.testing.assert_array_equal(counts.toarray(), [[3, 7], [2, 8]])
        log = normalize_counts(counts)
        np.testing.assert_allclose(log[:, 0].toarray().ravel(), np.log1p([3000, 2000]))
        selected, scores = select_hvg(log, count=1)
        self.assertEqual(len(selected), 1)
        self.assertTrue(np.isfinite(scores).all())

    def test_target_zero_before_source_scaler(self):
        raw = sparse.csr_matrix([[1., 9.], [5., 5.]])
        result, hit = map_rna(raw, ['ENSG1', 'ENSG9'], ['ENSG1', 'ENSG2'],
                              np.array([2., 3.]), np.array([1., 2.]))
        self.assertEqual(hit.tolist(), [True, False])
        np.testing.assert_allclose(result[:, 0], (np.log1p([1000, 5000])-2)/(1+1e-6), rtol=1e-6)
        np.testing.assert_allclose(result[:, 1], -3/(2+1e-6))

    def test_signed_median_weights_not_stability_weights(self):
        rho = np.tile([.5, -.8, .7, .1], (10, 1))
        rho[:4, 0] = -.5
        matrix, d = graph_weights(rho, np.array([0, 0, 0, 1]), np.arange(4), 3, 4,
                                  np.array([True, True, True, False]))
        self.assertEqual(d['selected'].tolist(), [False, True, True, False])
        np.testing.assert_allclose(matrix[:, 0].toarray().ravel(), [0, -.8/1.5, .7/1.5, 0])
        self.assertEqual(matrix[:, 1:].nnz, 0)
        self.assertEqual(d['zero_denominator_genes'].tolist(), [1])

    def test_spearman_resampling_matches_reference(self):
        rng = np.random.default_rng(4)
        rna = rng.normal(size=(100, 2)).astype(np.float32)
        atac = sparse.csr_matrix((rna > 0).astype(np.float32))
        graph = dict(candidate_index=np.array([[0, 1], [0, 1]]),
                     candidate_mask=np.ones((2, 2), bool))
        matrix, d = build_bs(rna, atac, graph, seed=20020130)
        samples = _resamples(np.arange(100), DModeReliabilityConfig(), 20020130)
        expected = np.array([[spearmanr(rna[rows, g], atac[rows, p].toarray().ravel()).statistic
                              for g, p in zip(d['edge_gene'], d['edge_peak'])] for rows in samples])
        np.testing.assert_allclose(d['fold_rho'], expected, atol=1e-6)
        np.testing.assert_allclose(np.asarray(abs(matrix).sum(0)).ravel(), 1)

    def test_validation_does_not_fit_field(self):
        train = sparse.csr_matrix([[1., 0.], [0., 1.], [1., 1.]])
        matrix = sparse.eye(2, dtype=np.float32).tocsr()
        a, sa, _ = build_field(sparse.vstack([train, [[0., 0.]]]).tocsr(), matrix, np.arange(3))
        b, sb, _ = build_field(sparse.vstack([train, [[1., 1.]]]).tocsr(), matrix, np.arange(3))
        for key in sa:
            np.testing.assert_array_equal(sa[key], sb[key])
        np.testing.assert_array_equal(a[:3], b[:3])

    def test_numeric_errors_are_fatal(self):
        for values in ([[float('nan')]], [[-1.]], [[.5]]):
            with self.assertRaises(ValueError):
                normalize_counts(sparse.csr_matrix(values))


if __name__ == '__main__':
    unittest.main()
