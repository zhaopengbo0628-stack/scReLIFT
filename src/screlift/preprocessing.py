"""Single-source RNA and S10x20 graph adapter; no model implementation."""
import numpy as np
from scipy import sparse
from types import SimpleNamespace

from ._regulatory.dmode_reliable_selector import DModeReliabilityConfig, _source_stability
from ._regulatory.dmode_teacher import fit_dmode_field


def finite(values, name):
    values = values.data if sparse.issparse(values) else np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError(f'{name}: NaN/Inf')


def normalize_counts(counts):
    counts = counts.tocsr().astype(np.float32)
    finite(counts, 'raw RNA')
    if (counts.data < 0).any() or (counts.data != np.floor(counts.data)).any():
        raise ValueError('RNA must contain nonnegative integer counts')
    totals = np.asarray(counts.sum(1)).ravel()
    normalized = sparse.diags(10000 / np.maximum(totals, 1.)).dot(counts).tocsr()
    normalized.data = np.log1p(normalized.data)
    finite(normalized, 'normalized RNA')
    return normalized


def canonical_genes(counts, gene_ids):
    ids = np.asarray([str(g).split('.')[0] if str(g).startswith('ENSG') else str(g)
                      for g in gene_ids])
    if counts.shape[1] != len(ids) or any(not g or g == 'nan' for g in ids):
        raise ValueError('Gene IDs missing or dimensions differ')
    # Merge only the same stable identifier; never infer equivalence from symbols.
    _, first = np.unique(ids, return_index=True)
    unique = ids[np.sort(first)]
    lookup = {gene:i for i,gene in enumerate(unique)}
    inverse = np.asarray([lookup[gene] for gene in ids])
    mapping = sparse.csr_matrix((np.ones(len(ids)), (np.arange(len(ids)), inverse)),
                                shape=(len(ids), len(unique)))
    merged = (counts @ mapping).tocsr().astype(np.float32)
    np.testing.assert_allclose(np.asarray(merged.sum(1)), np.asarray(counts.sum(1)))
    return merged, unique, inverse


def select_hvg(normalized_train, count=2000):
    finite(normalized_train, 'HVG input')
    if normalized_train.shape[1] < count:
        raise ValueError(f'Only {normalized_train.shape[1]} genes; requested {count}')
    mean = np.asarray(normalized_train.mean(0)).ravel()
    variance = np.maximum(np.asarray(normalized_train.power(2).mean(0)).ravel() - mean**2, 0.)
    score = variance / np.maximum(mean, 1e-6)
    selected = np.argsort(score, kind='stable')[-count:]
    return selected, score


def map_rna(counts, ids, wanted, mean, std):
    counts, ids, _ = canonical_genes(counts, ids)
    log = normalize_counts(counts)
    lookup = {g: i for i, g in enumerate(ids)}
    index = np.asarray([lookup.get(str(g), -1) for g in wanted])
    hit = index >= 0
    values = np.zeros((counts.shape[0], len(wanted)), np.float32)
    values[:, hit] = log[:, index[hit]].toarray()
    # Frozen Full B2 uses std+1e-6; missing genes are zero BEFORE this transform.
    result = ((values - mean) / (std + 1e-6)).astype(np.float32)
    finite(result, 'RNA input')
    return result, hit


def graph_weights(fold_rho, edge_gene, edge_peak, gene_count, peak_count, valid):
    finite(fold_rho, 'Spearman resamples')
    if fold_rho.shape != (10, len(edge_gene)):
        raise ValueError('S10x20 requires ten resamples')
    median = np.median(fold_rho, axis=0).astype(np.float32)
    consistency = np.abs(np.mean(np.sign(fold_rho), axis=0)).astype(np.float32)
    score = np.abs(median) * consistency
    selected = np.zeros(len(edge_gene), bool)
    weights = np.zeros(len(edge_gene), np.float32)
    zero_denominator = []
    for gene in range(gene_count):
        positions = np.flatnonzero(edge_gene == gene)
        if not len(positions):
            continue
        eligible = positions[valid[positions]]
        order = eligible[np.argsort(-score[eligible], kind='stable')]
        selected[order[:int(np.ceil(.5 * len(positions)))]] = True
        raw = np.where(selected[positions], median[positions], 0.)
        denominator = np.abs(raw).sum()
        if denominator > 0:
            weights[positions] = raw / denominator
        else:
            zero_denominator.append(gene)
    nonzero = weights != 0
    matrix = sparse.coo_matrix((weights[nonzero], (edge_peak[nonzero], edge_gene[nonzero])),
                              shape=(peak_count, gene_count), dtype=np.float32).tocsr()
    return matrix, dict(fold_rho=fold_rho, median_rho=median, C_sign=consistency,
                        S=score, selected=selected, valid=valid, edge_gene=edge_gene,
                        edge_peak=edge_peak, edge_weight=weights,
                        zero_denominator_genes=np.asarray(zero_denominator, np.int64))


def build_bs(rna_train, atac_train, graph, seed=20020130):
    finite(rna_train, 'selector RNA'); finite(atac_train, 'selector ATAC')
    if rna_train.shape[0] != atac_train.shape[0]:
        raise ValueError('Paired rows differ')
    mask = graph['candidate_mask']
    edge_gene = np.repeat(np.arange(mask.shape[0]), mask.shape[1])[mask.ravel()]
    edge_peak = graph['candidate_index'][mask]
    source = SimpleNamespace(rna=rna_train, atac=atac_train,
                             train=np.arange(len(rna_train)),
                             canonical_peak_available=np.ones(atac_train.shape[1], bool))
    stability = _source_stability(source, graph['candidate_index'], mask,
                                  DModeReliabilityConfig(), seed)
    matrix, diagnostics = graph_weights(stability['fold_rho'], edge_gene, edge_peak,
                                        rna_train.shape[1], atac_train.shape[1], stability['valid'])
    diagnostics['zero_variance_genes'] = np.flatnonzero(rna_train.var(0) <= 1e-8)
    peak_mean = np.asarray(atac_train.mean(0)).ravel()
    diagnostics['zero_variance_peaks'] = np.flatnonzero(
        np.asarray(atac_train.power(2).mean(0)).ravel() - peak_mean**2 <= 1e-8)
    diagnostics['no_candidate_genes'] = np.flatnonzero(~mask.any(1))
    diagnostics['unsupported_genes'] = np.flatnonzero(np.asarray(abs(matrix).sum(0)).ravel() == 0)
    finite(matrix, 'B_S')
    return matrix, diagnostics


def build_field(atac_train_val, matrix, train_rows):
    finite(atac_train_val, 'ATAC'); finite(matrix, 'B_S')
    field, state, report = fit_dmode_field(atac_train_val, matrix, train_rows)
    finite(field, 'F_S')
    report['shared_graph'] = 'single-source B_S'
    report['nondegenerate_genes'] = int(np.count_nonzero(state['field_std'] >= 1e-6))
    report['zero_std_gene_indices'] = np.flatnonzero(state['field_std'] < 1e-6).tolist()
    report['std_epsilon'] = 'unchanged frozen Full B2 std+1e-6, not graph-edge fabrication'
    return field, state, report
