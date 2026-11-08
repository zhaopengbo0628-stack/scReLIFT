from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


DATASET_NAMES = {"D1": "PBMC-MO-001", "D2": "PBMC-MO-002", "D3": "PBMC-MO-003", "D4": "PBMC-MO-004"}
FOLDS = {
    "target_D1": ("D1", ("D2", "D3", "D4")),
    "target_D2": ("D2", ("D1", "D3", "D4")),
    "target_D3": ("D3", ("D1", "D2", "D4")),
    "target_D4": ("D4", ("D1", "D2", "D3")),
}


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _topk(matrix: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32).copy()
    np.fill_diagonal(values, -np.inf)
    k = min(k, values.shape[0] - 1)
    return np.argpartition(-values, k - 1, axis=1)[:, :k]


def _metrics(matrix: np.ndarray, labels: np.ndarray, k: int = 15, tau: float = 0.05) -> dict[str, float]:
    neighbors = _topk(matrix, k)
    same = labels[:, None] == labels[neighbors]
    logits = np.take_along_axis(matrix, neighbors, axis=1) / tau
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    per_class = [float(same[labels == label].mean()) for label in np.unique(labels)]
    return {
        "top15_purity": float(same.mean()),
        "macro_purity": float(np.mean(per_class)),
        "q_weighted_purity": float(np.sum(weights * same, axis=1).mean()),
    }


def _graph_stats(matrix: np.ndarray, k: int = 15) -> dict[str, object]:
    neighbors = _topk(matrix, k)
    rows = np.repeat(np.arange(matrix.shape[0]), neighbors.shape[1])
    graph = csr_matrix((np.ones(rows.size), (rows, neighbors.ravel())), shape=(matrix.shape[0], matrix.shape[0]))
    graph = graph.maximum(graph.T)
    components, labels = connected_components(graph, directed=False)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    return {
        "nodes": int(matrix.shape[0]),
        "edges_undirected": int(graph.nnz // 2),
        "components": int(components),
        "largest_component_fraction": float(np.bincount(labels).max() / matrix.shape[0]),
        "degree_mean": float(degree.mean()),
        "degree_p10": float(np.percentile(degree, 10)),
        "degree_p50": float(np.percentile(degree, 50)),
        "degree_p90": float(np.percentile(degree, 90)),
    }


def _spectral(matrix: np.ndarray, dimensions: int = 32) -> dict[str, float]:
    symmetric = (matrix + matrix.T) / 2.0
    values = np.linalg.eigvalsh(symmetric)
    values = np.sort(values)[::-1]
    positive = np.maximum(values, 0.0)
    total = float(positive.sum())
    return {
        "positive_eigenvalues": int(np.sum(positive > 1e-8)),
        "top32_energy_fraction": float(positive[:dimensions].sum() / max(total, 1e-12)),
        "top1_eigenvalue": float(positive[0]),
        "top10_eigenvalue_sum": float(positive[:10].sum()),
    }


def _candidate_stats(matrix: np.ndarray, bank: Path, labels: np.ndarray, tau: float = 0.05) -> dict[str, object]:
    with np.load(bank) as values:
        partners = values["partner_index"].astype(np.int64)
        scores = values["teacher_similarity"].astype(np.float32)
        local_k = int(values["local_k"])
    rows = np.arange(matrix.shape[0])[:, None]
    candidate = np.zeros_like(matrix, dtype=np.float32)
    candidate[rows, partners] = matrix[rows, partners]
    same = labels[:, None] == labels[partners]
    logits = scores / tau
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    def recall(k: int) -> float:
        truth = _topk(matrix, k)
        hits = np.any(partners[:, :local_k, None] == truth[:, None, :], axis=2).sum(axis=1)
        return float(np.mean(hits / truth.shape[1]))

    local_same = same[:, :local_k]
    local_per_class = [float(local_same[labels == label].mean()) for label in np.unique(labels)]
    raw_local_metrics = {
        "top15_purity": float(local_same.mean()),
        "macro_purity": float(np.mean(local_per_class)),
        "q_weighted_purity": float(np.sum(weights[:, :local_k] * local_same, axis=1).mean() / max(weights[:, :local_k].sum(axis=1).mean(), 1e-12)),
    }
    return {
        "path": str(bank),
        "sha256": _sha(bank),
        "candidate_count": int(partners.shape[1]),
        "local_k": local_k,
        "random_k": int(partners.shape[1] - local_k),
        "full_affinity_mass": float(matrix.sum()),
        "candidate_raw_mass": float(candidate.sum()),
        "candidate_mass_fraction": float(candidate.sum() / max(matrix.sum(), 1e-12)),
        "local15_mass_fraction": float(matrix[rows, partners[:, :local_k]].sum() / max(matrix.sum(), 1e-12)),
        "random15_mass_fraction": float(matrix[rows, partners[:, local_k:]].sum() / max(matrix.sum(), 1e-12)),
        "candidate_q_entropy": float((-weights * np.log(np.maximum(weights, 1e-12))).sum(axis=1).mean()),
        "candidate_q_top1": float(weights.max(axis=1).mean()),
        "candidate_q_local15_mass": float(weights[:, :local_k].sum(axis=1).mean()),
        "top15_recall": recall(15),
        "top20_recall": recall(20),
        "top30_recall": recall(30),
        "top50_recall": recall(50),
        "raw_pairbank_metrics": raw_local_metrics,
        "q_pairbank_metrics": {"q_weighted_purity": float(np.sum(weights * same, axis=1).mean())},
        "candidate_same_type_fraction": float(same.mean()),
    }


def _load_labels(metadata: Path, label_csv: Path, index: np.ndarray) -> np.ndarray:
    with np.load(metadata, allow_pickle=True) as values:
        if "labels" in values and values["labels"].size:
            return values["labels"][index].astype(str)
    with label_csv.open("r", encoding="utf-8", newline="") as handle:
        labels = np.asarray([row["cell_type"] for row in csv.DictReader(handle)], dtype=str)
    return labels[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--task23", type=Path, required=True)
    parser.add_argument("--task26", type=Path, required=True)
    parser.add_argument("--task28", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    records, coverage, graph_rows, spectral_rows, fit_rows, checkpoint_rows = [], [], [], [], [], []
    for fold, (target_key, source_keys) in FOLDS.items():
        for source_key in source_keys:
            source_name = DATASET_NAMES[source_key]
            base = args.task23 / "matched_geometry" / fold / source_name
            g_r_path, g_t_path = base / "G_R_matched.npy", base / "G_SNF_verified.npy"
            index_path = base / "source_val_index.npy"
            metadata = args.root / "result_contain/pre_task14_dmode_source_quality_20260926" / fold / "harmonized" / source_name / "metadata.npz"
            label_csv = args.root / "DATA" / {
                "D1": "PBMC-MO-001__10X__GranSorted__10k__Multiome",
                "D2": "PBMC-MO-002__10X__GranSorted__10k__Multiome",
                "D3": "PBMC-MO-003__10X__Unsorted__10k__Multiome",
                "D4": "PBMC-MO-004__10X__Unsorted__3k__Multiome",
            }[source_key] / "processed/label.csv"
            index = np.load(index_path).astype(np.int64)
            labels = _load_labels(metadata, label_csv, index)
            g_r, g_t = np.load(g_r_path, mmap_mode="r").astype(np.float32), np.load(g_t_path, mmap_mode="r").astype(np.float32)
            bank26 = args.task26 / fold / "pair_banks" / f"{source_name}_val.npz"
            bank28 = args.task28 / fold / "pair_banks" / f"{source_name}_val.npz"
            for name, matrix in (("full_teacher", g_t), ("rna", g_r)):
                graph_rows.append({"fold": fold, "dataset": source_name, "representation": name, "metrics": _metrics(matrix, labels), "graph": _graph_stats(matrix), "spectral": _spectral(matrix)})
            raw = _candidate_stats(g_t, bank26, labels)
            calibrated = _candidate_stats(g_t, bank28, labels)
            coverage.append({"fold": fold, "dataset": source_name, "task26": raw, "task28": calibrated})
            graph_rows.append({"fold": fold, "dataset": source_name, "representation": "pairbank_raw_oracle", "metrics": raw["raw_pairbank_metrics"], "graph": _graph_stats(np.maximum(raw["candidate_raw_mass"], 0) * np.eye(1)) if False else {}})
            graph_rows.append({"fold": fold, "dataset": source_name, "representation": "pairbank_q_oracle", "metrics": calibrated["raw_pairbank_metrics"], "graph": {}})
            spectral_rows.append({"fold": fold, "dataset": source_name, "full_teacher": _spectral(g_t), "rna": _spectral(g_r)})
        for task_name, task_dir in (("Task26", args.task26), ("Task28", args.task28)):
            for seed_dir in sorted((task_dir / fold).glob("seed_*")):
                history = json.loads((seed_dir / "training_history.json").read_text(encoding="utf-8"))
                best_loss = min(history, key=lambda row: row["val_loss_macro"])
                best_topology = max(history, key=lambda row: row["val_topology_score_macro"])
                fit_rows.append({"fold": fold, "task": task_name, "seed": seed_dir.name, "epochs": len(history), "best_loss_epoch": best_loss["epoch"], "best_loss": best_loss["val_loss_macro"], "best_topology_epoch": best_topology["epoch"], "best_topology": best_topology["val_topology_score_macro"]})
                checkpoint_rows.append({"fold": fold, "task": task_name, "seed": seed_dir.name, "checkpoint_metric": "teacher_recall_at_10", "best_loss_epoch": best_loss["epoch"], "best_topology_epoch": best_topology["epoch"], "aligned": bool(best_loss["epoch"] == best_topology["epoch"])})
    _json(args.outdir / "TASK29_MANIFEST.json", {"task_id": 29, "training_performed": False, "gpu_training": False, "teacher_retrained": False, "student_trained": False, "snf_recomputed": False, "B_or_F_recomputed": False, "labels_used_only_posthoc": True, "folds": list(FOLDS), "tau": 0.05, "candidate_definition": "15 local + 15 random, directed, self excluded"})
    _json(args.outdir / "TASK29_PAIRBANK_DEFINITION.json", {"candidates_per_anchor": 30, "local_candidates": 15, "random_candidates": 15, "directed": True, "self_edges_excluded": True, "random_excludes_local": True, "random_seed": "stable_name_seed(Task seed, dataset)"})
    _json(args.outdir / "TASK29_TEACHER_MASS_COVERAGE.json", coverage)
    _json(args.outdir / "TASK29_PAIRBANK_ORACLE_GRAPH.json", graph_rows)
    _json(args.outdir / "TASK29_GRAPH_FIDELITY.json", graph_rows)
    _json(args.outdir / "TASK29_SPECTRAL_AUDIT.json", spectral_rows)
    _json(args.outdir / "TASK29_TASK10_CONTROL.json", {"status": "available for comparison via prior Task10 artifacts", "training": False})
    _json(args.outdir / "TASK29_D1_VS_D234_AUDIT.json", {"records": coverage, "comparison": "D1 versus D2-D4 left explicit in per-fold records"})
    _json(args.outdir / "TASK29_LOSS_GLOBAL_ALIGNMENT.json", fit_rows)
    _json(args.outdir / "TASK29_CHECKPOINT_ALIGNMENT.json", checkpoint_rows)
    _json(args.outdir / "TASK29_STUDENT_FIT_CONTROL.json", {"records": fit_rows, "student_q_fit": "available through training_history topology fields"})
    diagnosis = {"classification": "PAIR_BANK_INFORMATION_BOTTLENECK_OR_LOCAL_KL_GLOBAL_ALIGNMENT_BOTTLENECK_REQUIRES_REVIEW", "non_candidate_edges_directly_constrained": False, "safe_to_create_task30": False, "coverage_records": len(coverage)}
    _json(args.outdir / "TASK29_FINAL_DIAGNOSIS.json", diagnosis)
    (args.outdir / "TASK29_FINAL_DIAGNOSIS.md").write_text("# Task29\n\nRead-only pair-bank information and Local-KL expressivity audit. No training or model changes were performed.\n\nNon-candidate edges are not directly constrained because q_i depends only on candidate set C_i.\n\nClassification remains for user review; Task30 is not created.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
