from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.stats import rankdata

from .data import V5DData
from .utils import sha256_array, stable_name_seed


@dataclass(frozen=True)
class DModeReliabilityConfig:
    resample_count: int = 10
    resample_fraction: float = 0.2
    edge_density: float = 0.5
    variance_epsilon: float = 1e-8


def _normalized_ranks(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values, axis=0, method="average").astype(np.float32)
    ranks -= ranks.mean(axis=0, keepdims=True)
    return ranks / np.maximum(np.sqrt(np.sum(ranks * ranks, axis=0, keepdims=True)), 1e-8)


def _edge_rho(
    rna: np.ndarray,
    atac: sparse.csr_matrix,
    rows: np.ndarray,
    edge_gene: np.ndarray,
    edge_peak: np.ndarray,
) -> np.ndarray:
    rna_rank = _normalized_ranks(np.asarray(rna[rows], dtype=np.float32))
    result = np.empty(edge_gene.size, dtype=np.float32)
    for start in range(0, edge_gene.size, 2048):
        stop = min(start + 2048, edge_gene.size)
        peaks = np.unique(edge_peak[start:stop])
        atac_rank = _normalized_ranks(atac[rows][:, peaks].toarray().astype(np.float32))
        peak_columns = np.searchsorted(peaks, edge_peak[start:stop])
        result[start:stop] = np.sum(
            rna_rank[:, edge_gene[start:stop]] * atac_rank[:, peak_columns], axis=0
        )
    return np.nan_to_num(result, nan=0.0).astype(np.float32)


def _resamples(train: np.ndarray, config: DModeReliabilityConfig, seed: int) -> list[np.ndarray]:
    if config.resample_count != 10 or config.resample_fraction != 0.2:
        raise ValueError("D-mode baseline requires exactly 10 resamples at fraction 0.2")
    if train.size < 2:
        raise ValueError("D-mode stability requires at least two train cells")
    size = max(2, int(np.ceil(train.size * config.resample_fraction)))
    rng = np.random.default_rng(seed)
    return [rng.choice(train, size=size, replace=False) for _ in range(config.resample_count)]


def _source_stability(
    data: V5DData,
    candidate_index: np.ndarray,
    candidate_mask: np.ndarray,
    config: DModeReliabilityConfig,
    seed: int,
) -> dict[str, np.ndarray]:
    edge_gene = np.repeat(np.arange(candidate_index.shape[0]), candidate_index.shape[1])[candidate_mask.ravel()]
    edge_peak = candidate_index[candidate_mask]
    rna = data.rna
    atac = data.atac
    train_rna = np.asarray(rna[data.train], dtype=np.float32)
    train_atac = atac[data.train]
    gene_var = train_rna.var(axis=0)
    peak_mean = np.asarray(train_atac.mean(axis=0)).ravel()
    peak_var = np.asarray(train_atac.power(2).mean(axis=0)).ravel() - peak_mean * peak_mean
    valid_gene = gene_var > config.variance_epsilon
    valid_peak = data.canonical_peak_available & (peak_var > config.variance_epsilon)
    valid = valid_gene[edge_gene] & valid_peak[edge_peak]
    folds = np.stack([
        _edge_rho(rna, atac, rows, edge_gene, edge_peak)
        for rows in _resamples(data.train, config, seed)
    ])
    median = np.median(folds, axis=0).astype(np.float32)
    c_sign = np.abs(np.mean(np.sign(folds), axis=0)).astype(np.float32)
    c_sign[(median == 0) | ~np.isfinite(median)] = 0.0
    median[~valid] = np.nan
    c_sign[~valid] = 0.0
    within = (np.abs(np.nan_to_num(median)) * c_sign).astype(np.float32)
    return {
        "fold_rho": folds,
        "median_rho": median,
        "C_sign": c_sign,
        "S_within": within,
        "valid": valid,
    }


def build_source_only_b(
    datasets: list[V5DData],
    candidate_graph: dict[str, np.ndarray],
    gene_ids: np.ndarray,
    config: DModeReliabilityConfig = DModeReliabilityConfig(),
    seed: int = 20020130,
) -> dict:
    if len(datasets) != 3:
        raise ValueError("D-mode baseline requires exactly three source datasets")
    candidate_index = candidate_graph["candidate_index"].astype(np.int64)
    candidate_mask = candidate_graph["candidate_mask"].astype(bool)
    edge_gene = np.repeat(np.arange(candidate_index.shape[0]), candidate_index.shape[1])[candidate_mask.ravel()]
    edge_peak = candidate_index[candidate_mask]
    scores = [
        _source_stability(
            data,
            candidate_index,
            candidate_mask,
            config,
            stable_name_seed(seed, data.name),
        )
        for data in datasets
    ]
    rho = np.stack([score["median_rho"] for score in scores])
    within = np.stack([score["S_within"] for score in scores])
    valid = np.stack([score["valid"] for score in scores])
    q_values = []
    for score in scores:
        support = score["valid"] & np.isfinite(score["median_rho"])
        numerator = np.sum(np.abs(score["median_rho"][support]) * score["C_sign"][support])
        denominator = np.sum(np.abs(score["median_rho"][support]))
        q_values.append(float(numerator / max(denominator, 1e-8)))
    q_dataset = np.asarray(q_values, dtype=np.float32)
    n_train = np.asarray([data.train.size for data in datasets], dtype=np.float32)
    raw_dataset_weight = n_train * q_dataset
    if not np.any(raw_dataset_weight > 0):
        raise RuntimeError("No source dataset has valid regulatory stability")
    omega = raw_dataset_weight / raw_dataset_weight.sum()

    evidence = omega[:, None] * np.nan_to_num(within, nan=0.0)
    support = (omega[:, None] * valid).sum(axis=0).astype(np.float32)
    weighted_sign = evidence * np.sign(np.nan_to_num(rho, nan=0.0))
    mass = evidence.sum(axis=0).astype(np.float32)
    cross_sign = np.divide(
        np.abs(weighted_sign.sum(axis=0)),
        mass,
        out=np.zeros_like(mass),
        where=mass > 0,
    ).astype(np.float32)
    reliability = (mass * cross_sign * support).astype(np.float32)
    consensus_sign = np.sign(weighted_sign.sum(axis=0)).astype(np.float32)

    selected = np.zeros(edge_gene.size, dtype=bool)
    for gene in range(gene_ids.size):
        positions = np.flatnonzero(edge_gene == gene)
        if positions.size:
            keep = max(1, int(np.ceil(positions.size * config.edge_density)))
            order = positions[np.argsort(-reliability[positions], kind="stable")]
            selected[order[:keep]] = True
    raw = np.where(selected, consensus_sign * reliability, 0.0).astype(np.float32)
    b_values = np.zeros_like(raw)
    for gene in range(gene_ids.size):
        positions = np.flatnonzero(edge_gene == gene)
        denominator = np.sum(np.abs(raw[positions]))
        if denominator > 0:
            b_values[positions] = raw[positions] / denominator
    nonzero = np.abs(b_values) > 1e-8
    b_reg = sparse.coo_matrix(
        (b_values[nonzero], (edge_peak[nonzero], edge_gene[nonzero])),
        shape=(candidate_graph["peak_names"].size, gene_ids.size),
        dtype=np.float32,
    ).tocsr()
    return {
        "B_D": b_reg,
        "fold_rho": [score["fold_rho"] for score in scores],
        "median_rho": [score["median_rho"] for score in scores],
        "C_sign": [score["C_sign"] for score in scores],
        "S_within": [score["S_within"] for score in scores],
        "valid": [score["valid"] for score in scores],
        "n_train": n_train,
        "Q_d": q_dataset,
        "raw_dataset_weight": raw_dataset_weight,
        "omega": omega,
        "edge_support": support,
        "C_cross": cross_sign,
        "R_D": reliability,
        "sign_D": consensus_sign,
        "selected": selected,
        "edge_gene": edge_gene,
        "edge_peak": edge_peak,
        "manifest": {
            "method": "D-mode source-only cross-dataset reliability screening",
            "resample_count": config.resample_count,
            "resample_fraction": config.resample_fraction,
            "score": "R_D = M_D * C_cross * V",
            "dataset_weight": "omega_d = n_d * Q_d / sum(n_k * Q_k)",
            "within_score": "S_d = abs(median_rho_d) * abs(mean(sign(rho_d)))",
            "selection": "per-gene exact Top50% by R_D",
            "b_normalization": "per-gene L1 on sign_D * R_D",
            "labels_used": False,
            "target_used": False,
            "predictability_used": False,
            "distance_prior_used": False,
            "gene_confidence_used": False,
            "B_floor_used": False,
            "candidate_edge_count": int(edge_gene.size),
            "selected_edge_count": int(selected.sum()),
            "effective_edge_count": int(b_reg.nnz),
            "datasets": [data.name for data in datasets],
            "dataset_names_sha256": sha256_array(np.asarray([data.name for data in datasets], dtype=str)),
        },
    }
