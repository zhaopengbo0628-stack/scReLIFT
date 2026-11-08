from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from adjudicate_task29_causal import FOLDS, NAMES, RNAOnlySource, clustering, labels_for, spectral_embedding, topk


def sparse_topk(full: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(full, dtype=np.float32).copy()
    np.fill_diagonal(values, -np.inf)
    k = min(k, values.shape[0] - 1)
    indices = np.argpartition(-values, k - 1, axis=1)[:, :k]
    graph = np.zeros_like(values, dtype=np.float32)
    rows = np.arange(values.shape[0])[:, None]
    graph[rows, indices] = np.maximum(values[rows, indices], 0.0)
    graph = (graph + graph.T) / 2.0
    np.fill_diagonal(graph, 0.0)
    return graph


def old_pairbank(full: np.ndarray, bank: Path) -> np.ndarray:
    with np.load(bank) as values:
        partners = values["partner_index"].astype(np.int64)
    graph = np.zeros_like(full, dtype=np.float32)
    rows = np.arange(full.shape[0])[:, None]
    graph[rows, partners] = np.maximum(full[rows, partners], 0.0)
    graph = (graph + graph.T) / 2.0
    np.fill_diagonal(graph, 0.0)
    return graph


def _component_sizes(graph: np.ndarray) -> list[int]:
    adjacency = graph > 0
    seen = np.zeros(graph.shape[0], dtype=bool)
    sizes = []
    for start in range(graph.shape[0]):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            nxt = np.flatnonzero(adjacency[node] & ~seen)
            seen[nxt] = True
            stack.extend(nxt.tolist())
        sizes.append(size)
    return sizes


def fidelity_to_full(full: np.ndarray, sparse: np.ndarray) -> dict[str, float | int]:
    full_embedding = spectral_embedding(full, 32)
    sparse_embedding = spectral_embedding(sparse, 32)
    width = min(full_embedding.shape[1], sparse_embedding.shape[1], 16)
    full_basis = np.linalg.qr(full_embedding[:, :width])[0]
    sparse_basis = np.linalg.qr(sparse_embedding[:, :width])[0]
    principal = np.linalg.svd(full_basis.T @ sparse_basis, compute_uv=False)
    full_neighbors = topk(full, 15)
    sparse_neighbors = topk(sparse, 15)
    neighborhood = float(np.mean([len(set(full_neighbors[i]).intersection(sparse_neighbors[i])) / 15.0 for i in range(full.shape[0])]))
    def laplacian(graph: np.ndarray) -> np.ndarray:
        degree = graph.sum(axis=1)
        inv = 1.0 / np.sqrt(np.maximum(degree, 1e-8))
        return np.eye(graph.shape[0], dtype=np.float32) - inv[:, None] * graph * inv[None, :]
    full_values = np.linalg.eigvalsh(laplacian(full))[:16]
    sparse_values = np.linalg.eigvalsh(laplacian(sparse))[:16]
    sizes = _component_sizes(sparse)
    return {
        "principal_angle_mean_cosine": float(np.mean(principal)),
        "teacher_top15_neighborhood_preservation": neighborhood,
        "normalized_laplacian_spectral_rmse": float(np.sqrt(np.mean((full_values - sparse_values) ** 2))),
        "components_right": int(len(sizes)),
        "largest_component_fraction_right": float(max(sizes) / sparse.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task23", type=Path, required=True)
    parser.add_argument("--task26", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    records = []
    for fold, (_, source_keys) in FOLDS.items():
        harmonized = args.root / "result_contain/pre_task14_dmode_source_quality_20260926" / fold / "harmonized"
        for source_key in source_keys:
            name = NAMES[source_key]
            base = args.task23 / "matched_geometry" / fold / name
            full = np.load(base / "G_SNF_verified.npy").astype(np.float32)
            source = RNAOnlySource(harmonized, name)
            labels = labels_for(args.root, source, source.val)
            graphs = {"FULL": full, "OLD_PAIRBANK_RAW": old_pairbank(full, args.task26 / fold / "pair_banks" / f"{name}_val.npz")}
            for k in (30, 60, 120):
                graphs[f"TOP{k}_ORACLE"] = sparse_topk(full, k)
            for policy, graph in graphs.items():
                fidelity = {"identity": True} if policy == "FULL" else fidelity_to_full(full, graph)
                records.append({"fold": fold, "dataset": name, "policy": policy, "graph_fidelity_to_full": fidelity, "posthoc_clustering": clustering(graph, labels, 0), "nodes": int(graph.shape[0]), "nonzero_directed": int(np.count_nonzero(graph))})
    summary = {}
    for policy in ("OLD_PAIRBANK_RAW", "TOP30_ORACLE", "TOP60_ORACLE", "TOP120_ORACLE"):
        rows = [row for row in records if row["policy"] == policy]
        fidelity = [row["graph_fidelity_to_full"] for row in rows]
        summary[policy] = {
            "records": len(rows),
            "mean_principal_angle_cosine": float(np.mean([x["principal_angle_mean_cosine"] for x in fidelity])),
            "mean_teacher_top15_neighborhood_preservation": float(np.mean([x["teacher_top15_neighborhood_preservation"] for x in fidelity])),
            "mean_normalized_laplacian_spectral_rmse": float(np.mean([x["normalized_laplacian_spectral_rmse"] for x in fidelity])),
            "all_connected": bool(all(x["components_right"] == 1 for x in fidelity)),
            "mean_ari": float(np.mean([row["posthoc_clustering"]["ari"] for row in rows])),
            "mean_nmi": float(np.mean([row["posthoc_clustering"]["nmi"] for row in rows])),
        }
    comparable = ("mean_principal_angle_cosine", "mean_teacher_top15_neighborhood_preservation")
    sufficient = []
    for k in (30, 60, 120):
        policy = f"TOP{k}_ORACLE"
        if all(summary[policy][key] >= summary["OLD_PAIRBANK_RAW"][key] for key in comparable) and summary[policy]["mean_normalized_laplacian_spectral_rmse"] <= summary["OLD_PAIRBANK_RAW"]["mean_normalized_laplacian_spectral_rmse"]:
            sufficient.append(k)
    capacity = "UNCERTAIN" if not sufficient else ("K30_SUFFICIENT" if min(sufficient) == 30 else "K60_REQUIRED" if min(sufficient) == 60 else "K120_REQUIRED")
    result = {"task_id": 29, "training_performed": False, "teacher_retrained": False, "snf_recomputed": False, "labels_used_only_posthoc": True, "graph_convention": "Full SNF affinity; exact row-wise top-K; union symmetrization by arithmetic mean; zero diagonal; shared normalized-Laplacian evaluator", "capacities": [30, 60, 120], "summary": summary, "records": records, "PAIRBANK_CAPACITY": capacity, "sufficient_capacities": sufficient, "decision": "capacity audit only; Task30 not created"}
    (args.outdir / "TASK29_SPECTRAL_CAPACITY_AUDIT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Task29 spectral pair-bank capacity audit", "", "No training, Teacher retraining, B/F recomputation, or SNF recomputation was performed.", "", f"PAIRBANK_CAPACITY: {capacity}", "", "| Policy | Principal-angle cosine | Teacher Top15 preservation | Laplacian RMSE | Mean ARI | Mean NMI |", "|---|---:|---:|---:|---:|---:|"]
    for policy in ("OLD_PAIRBANK_RAW", "TOP30_ORACLE", "TOP60_ORACLE", "TOP120_ORACLE"):
        row = summary[policy]
        lines.append(f"| {policy} | {row['mean_principal_angle_cosine']:.4f} | {row['mean_teacher_top15_neighborhood_preservation']:.4f} | {row['mean_normalized_laplacian_spectral_rmse']:.4f} | {row['mean_ari']:.4f} | {row['mean_nmi']:.4f} |")
    (args.outdir / "TASK29_SPECTRAL_CAPACITY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
