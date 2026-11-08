from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


def _normalized_ranks(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values, axis=0, method="average").astype(np.float32)
    ranks -= ranks.mean(axis=0, keepdims=True)
    return ranks / np.maximum(np.sqrt(np.sum(ranks * ranks, axis=0, keepdims=True)), 1e-8)


def _edge_rho(rna: np.ndarray, atac: sparse.csr_matrix, rows: np.ndarray, candidate_index: np.ndarray, candidate_mask: np.ndarray) -> np.ndarray:
    covered = np.unique(candidate_index[candidate_mask])
    rna_rank = _normalized_ranks(np.asarray(rna[rows], dtype=np.float32))
    atac_rank = _normalized_ranks(atac[rows][:, covered].toarray().astype(np.float32))
    correlation = rna_rank.T @ atac_rank
    edge_gene = np.repeat(np.arange(candidate_index.shape[0]), candidate_index.shape[1])[candidate_mask.ravel()]
    edge_peak = np.searchsorted(covered, candidate_index[candidate_mask])
    return np.nan_to_num(correlation[edge_gene, edge_peak], nan=0.0).astype(np.float32)


def _edge_oof_r2(
    rna: np.ndarray,
    atac: sparse.csr_matrix,
    split_folds: list[np.ndarray],
    edge_peak: np.ndarray,
    seed: int,
    n_components: int = 64,
    alpha: float = 1.0,
    chunk_size: int = 2048,
) -> np.ndarray:
    unique_peaks = np.unique(edge_peak)
    sse = np.zeros(unique_peaks.size, dtype=np.float64)
    sst = np.zeros(unique_peaks.size, dtype=np.float64)
    rows = np.concatenate(split_folds)
    for start in range(0, unique_peaks.size, chunk_size):
        stop = min(start + chunk_size, unique_peaks.size)
        peaks = unique_peaks[start:stop]
        target_all = atac[rows][:, peaks].toarray().astype(np.float32)
        target_mean = target_all.mean(axis=0)
        sst[start:stop] = np.sum((target_all - target_mean) ** 2, axis=0)
    for fold_number, validation in enumerate(split_folds):
        training = np.setdiff1d(rows, validation, assume_unique=False)
        x_train = np.asarray(rna[training], dtype=np.float32)
        x_validation = np.asarray(rna[validation], dtype=np.float32)
        x_mean = x_train.mean(axis=0)
        x_std = x_train.std(axis=0)
        x_train = (x_train - x_mean) / np.maximum(x_std, 1e-6)
        x_validation = (x_validation - x_mean) / np.maximum(x_std, 1e-6)
        components = min(n_components, x_train.shape[0] - 1, x_train.shape[1])
        pca = PCA(n_components=components, svd_solver="randomized", random_state=seed + fold_number)
        x_train = pca.fit_transform(x_train).astype(np.float32)
        x_validation = pca.transform(x_validation).astype(np.float32)
        for start in range(0, unique_peaks.size, chunk_size):
            stop = min(start + chunk_size, unique_peaks.size)
            peaks = unique_peaks[start:stop]
            y_train = atac[training][:, peaks].toarray().astype(np.float32)
            y_validation = atac[validation][:, peaks].toarray().astype(np.float32)
            prediction = Ridge(alpha=alpha).fit(x_train, y_train).predict(x_validation)
            sse[start:stop] += np.sum((prediction - y_validation) ** 2, axis=0)
    peak_scores = np.zeros(unique_peaks.size, dtype=np.float32)
    valid = sst > 1e-8
    peak_scores[valid] = np.maximum(1.0 - sse[valid] / sst[valid], 0.0).astype(np.float32)
    return peak_scores[np.searchsorted(unique_peaks, edge_peak)]


def _sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def select_soft_edges(
    data_dir: Path,
    folds: int,
    seed: int,
    predictability_epsilon: float = 0.1,
    gene_confidence_epsilon: float = 0.1,
    atac_row_permutation: np.ndarray | None = None,
) -> dict:
    rna = np.load(data_dir / "rna.npy", mmap_mode="r")
    atac = sparse.load_npz(data_dir / "atac_binary.npz").tocsr()
    with np.load(data_dir / "metadata.npz", allow_pickle=True) as meta:
        train_index = meta["train_index"].astype(np.int64)
        gene_names = meta["gene_names"].astype(str)
        peak_names = meta["peak_names"].astype(str)
    with np.load(data_dir / "regulatory_graph.npz", allow_pickle=True) as graph:
        candidate_index = graph["candidate_index"].astype(np.int64)
        candidate_mask = graph["candidate_mask"].astype(bool)
        distance_bp = graph["distance_bp"].astype(np.float32)
    if rna.shape[0] != atac.shape[0] or rna.shape[1] != candidate_index.shape[0] or atac.shape[1] != peak_names.size:
        raise RuntimeError("V5-S data and graph dimensions do not match")
    if folds != 5 or folds > train_index.size:
        raise ValueError("V5-S soft selection requires exactly 5 folds")
    if not 0.0 <= predictability_epsilon <= 1.0:
        raise ValueError("predictability_epsilon must be between 0 and 1")
    if not 0.0 <= gene_confidence_epsilon <= 1.0:
        raise ValueError("gene_confidence_epsilon must be between 0 and 1")
    if atac_row_permutation is None:
        atac_row_permutation = np.arange(atac.shape[0], dtype=np.int64)
    else:
        atac_row_permutation = np.asarray(atac_row_permutation, dtype=np.int64)
        if atac_row_permutation.shape != (atac.shape[0],) or np.unique(atac_row_permutation).size != atac.shape[0]:
            raise ValueError("atac_row_permutation must be a permutation of all cells")
        atac = atac[atac_row_permutation]
    rng = np.random.default_rng(seed)
    split_folds = list(np.array_split(rng.permutation(train_index), folds))
    fold_rho = np.stack([_edge_rho(rna, atac, rows, candidate_index, candidate_mask) for rows in split_folds], axis=0)
    median_rho = np.median(fold_rho, axis=0).astype(np.float32)
    sign = np.sign(median_rho)
    sign_stability = np.mean(np.sign(fold_rho) == sign[None], axis=0).astype(np.float32)
    sign_stability[sign == 0] = 0.0
    rho_mad = np.median(np.abs(fold_rho - median_rho[None]), axis=0).astype(np.float32)
    rho_iqr = (np.percentile(fold_rho, 75, axis=0) - np.percentile(fold_rho, 25, axis=0)).astype(np.float32)
    q_s = (np.abs(median_rho) * sign_stability).astype(np.float32)
    edge_gene = np.repeat(np.arange(candidate_index.shape[0]), candidate_index.shape[1])[candidate_mask.ravel()]
    edge_peak = candidate_index[candidate_mask]
    edge_distance = distance_bp[candidate_mask]
    r2_oof = _edge_oof_r2(rna, atac, split_folds, edge_peak, seed)
    r2_q90 = float(np.percentile(r2_oof, 90))
    r2_normalized = np.clip(r2_oof / r2_q90, 0.0, 1.0) if r2_q90 > 1e-8 else np.zeros_like(r2_oof)
    predictability_factor = predictability_epsilon + (1.0 - predictability_epsilon) * r2_normalized
    distance_prior = 0.5 + 0.5 * np.exp(-edge_distance / 100_000.0)
    q_gp = q_s * predictability_factor
    gene_confidence_raw = np.zeros(gene_names.size, dtype=np.float32)
    for gene in range(gene_names.size):
        positions = np.flatnonzero(edge_gene == gene)
        denominator = np.sum(q_s[positions])
        if denominator > 0:
            gene_confidence_raw[gene] = np.sum(q_s[positions] * r2_normalized[positions]) / denominator
    gene_confidence = gene_confidence_epsilon + (1.0 - gene_confidence_epsilon) * gene_confidence_raw
    raw_weight = np.sign(median_rho) * q_gp * distance_prior
    b_reg = np.zeros((peak_names.size, gene_names.size), dtype=np.float32)
    for gene in range(gene_names.size):
        positions = np.flatnonzero(edge_gene == gene)
        denominator = np.sum(np.abs(raw_weight[positions]))
        if denominator > 0:
            b_reg[edge_peak[positions], gene] = raw_weight[positions] / denominator
    edge_rows = []
    for position in range(edge_gene.size):
        edge_rows.append({
            "gene_index": int(edge_gene[position]),
            "gene_name": str(gene_names[edge_gene[position]]),
            "peak_index": int(edge_peak[position]),
            "peak_coordinate": str(peak_names[edge_peak[position]]),
            "tss_distance": float(edge_distance[position]),
            "rho_fold_1": float(fold_rho[0, position]),
            "rho_fold_2": float(fold_rho[1, position]),
            "rho_fold_3": float(fold_rho[2, position]),
            "rho_fold_4": float(fold_rho[3, position]),
            "rho_fold_5": float(fold_rho[4, position]),
            "median_rho": float(median_rho[position]),
            "abs_median_rho": float(abs(median_rho[position])),
            "sign_stability": float(sign_stability[position]),
            "rho_mad": float(rho_mad[position]),
            "rho_iqr": float(rho_iqr[position]),
            "Q_S": float(q_s[position]),
            "rna_to_atac_r2_oof": float(r2_oof[position]),
            "rna_to_atac_r2_normalized": float(r2_normalized[position]),
            "Q_gp": float(q_gp[position]),
            "gene_confidence": float(gene_confidence[edge_gene[position]]),
        })
    candidate_counts = candidate_mask.sum(axis=1).astype(np.float32)
    statistics = {
        "model_gene_count": int(gene_names.size),
        "genes_with_candidate_peaks": int(candidate_mask.any(axis=1).sum()),
        "genes_with_nonzero_edges": int(np.unique(edge_gene[np.abs(raw_weight) > 1e-8]).size),
        "candidate_edge_count": int(candidate_mask.sum()),
        "effective_edge_count": int(np.count_nonzero(np.abs(b_reg) > 1e-8)),
        "edges_per_gene": {
            "mean": float(np.mean(candidate_counts)),
            "median": float(np.median(candidate_counts)),
            "q25": float(np.percentile(candidate_counts, 25)),
            "q75": float(np.percentile(candidate_counts, 75)),
        },
        "peak_coverage": float(np.unique(edge_peak).size / peak_names.size),
        "median_abs_rho": float(np.median(np.abs(median_rho))),
        "median_rna_to_atac_r2_oof": float(np.median(r2_oof)),
        "p_rna_to_atac_r2_gt_0": float(np.mean(r2_oof > 0)),
        "rna_to_atac_r2_q90": r2_q90,
        "median_Q_gp": float(np.median(q_gp)),
        "gene_confidence_mean": float(np.mean(gene_confidence)),
        "gene_confidence_median": float(np.median(gene_confidence)),
        "gene_confidence_q25": float(np.percentile(gene_confidence, 25)),
        "gene_confidence_q75": float(np.percentile(gene_confidence, 75)),
        "p_abs_rho_gt_0.01": float(np.mean(np.abs(median_rho) > 0.01)),
        "p_abs_rho_gt_0.03": float(np.mean(np.abs(median_rho) > 0.03)),
        "p_abs_rho_gt_0.05": float(np.mean(np.abs(median_rho) > 0.05)),
    }
    manifest = {
        "experiment": "V5-S Stage 1 soft-stable regulatory graph",
        "dataset": "PBMC10k",
        "selection_split": "train_only",
        "folds": int(folds),
        "seed": int(seed),
        "score": "Q_gp = abs(median Spearman rho) * sign stability * (epsilon + (1 - epsilon) * clipped OOF R2 full RNA profile->ATAC(p))",
        "predictability_epsilon": float(predictability_epsilon),
        "gene_confidence_epsilon": float(gene_confidence_epsilon),
        "distance_prior": "0.5 + 0.5 * exp(-distance_bp / 100000)",
        "b_reg_normalization": "per gene sum abs(B_reg[:, gene]) = 1 when nonzero",
        "b_reg_role": "B_local: within-gene peak structure; final gene-level contribution uses sqrt(gene_confidence)",
        "hard_biological_threshold": None,
        "rna_to_atac_predictability": "5-fold OOF Ridge prediction of each candidate ATAC peak from the full RNA profile after fold-fitted PCA (up to 64 components); R2 clipped at zero and normalized by train-edge Q90",
        "labels_used_for_selection": False,
        "atac_row_permutation_sha256": _sha256(atac_row_permutation),
        "val_or_test_used_for_selection": False,
        "train_index_sha256": _sha256(train_index),
        **statistics,
    }
    return {
        "graph": {"candidate_index": candidate_index, "candidate_mask": candidate_mask, "distance_bp": distance_bp, "peak_names": peak_names},
        "B_reg": b_reg,
        "gene_confidence": gene_confidence,
        "fold_rho": fold_rho,
        "median_rho": median_rho,
        "sign_stability": sign_stability,
        "rho_mad": rho_mad,
        "rho_iqr": rho_iqr,
        "Q_S": q_s,
        "edges": edge_rows,
        "statistics": statistics,
        "manifest": manifest,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20020130)
    parser.add_argument("--predictability-epsilon", type=float, default=0.1)
    parser.add_argument("--gene-confidence-epsilon", type=float, default=0.1)
    args = parser.parse_args()
    result = select_soft_edges(
        args.data_dir,
        args.folds,
        args.seed,
        args.predictability_epsilon,
        args.gene_confidence_epsilon,
    )
    args.outdir.mkdir(parents=True, exist_ok=False)
    for name in ("fold_rho", "median_rho", "sign_stability", "rho_mad", "rho_iqr", "Q_S", "B_reg", "gene_confidence"):
        np.save(args.outdir / f"{name}.npy", result[name])
    np.savez_compressed(args.outdir / "regulatory_graph.npz", **result["graph"])
    (args.outdir / "edges.json").write_text(json.dumps(result["edges"], indent=2), encoding="utf-8")
    (args.outdir / "edge_statistics.json").write_text(json.dumps(result["statistics"], indent=2), encoding="utf-8")
    (args.outdir / "manifest.json").write_text(json.dumps(result["manifest"], indent=2), encoding="utf-8")
    print(json.dumps(result["manifest"], indent=2), flush=True)


if __name__ == "__main__":
    main()
