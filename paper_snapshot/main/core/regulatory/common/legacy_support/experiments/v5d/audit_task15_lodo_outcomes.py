from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


DATASET_NAMES = {
    "D1": "PBMC-MO-001",
    "D2": "PBMC-MO-002",
    "D3": "PBMC-MO-003",
    "D4": "PBMC-MO-004",
}
LABEL_PATHS = {
    "D1": "DATA/PBMC-MO-001__10X__GranSorted__10k__Multiome/processed/label.csv",
    "D2": "DATA/PBMC-MO-002__10X__GranSorted__10k__Multiome/processed/label.csv",
    "D3": "DATA/PBMC-MO-003__10X__Unsorted__10k__Multiome/processed/label.csv",
    "D4": "DATA/PBMC-MO-004__10X__Unsorted__3k__Multiome/processed/label.csv",
}
FOLDS = {f"target_{key}": key for key in DATASET_NAMES}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _load_labels(root: Path, dataset_key: str, rows: np.ndarray) -> np.ndarray:
    path = root / LABEL_PATHS[dataset_key]
    with path.open("r", encoding="utf-8", newline="") as handle:
        values = np.asarray([row["cell_type"] for row in csv.DictReader(handle)], dtype=str)
    return values[rows]


def _class_metrics(labels: np.ndarray, clusters: np.ndarray) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for label in np.unique(labels):
        mask = labels == label
        class_count = int(mask.sum())
        candidates = np.unique(clusters[mask])
        best = max(candidates, key=lambda cluster: int(np.sum(mask & (clusters == cluster))))
        overlap = int(np.sum(mask & (clusters == best)))
        cluster_size = int(np.sum(clusters == best))
        result[str(label)] = {
            "cell_count": class_count,
            "best_cluster": int(best),
            "recall": overlap / max(class_count, 1),
            "precision": overlap / max(cluster_size, 1),
            "cluster_count_touched": int(candidates.size),
        }
    return result


def _cluster_diagnostics(labels: np.ndarray, clusters: np.ndarray) -> dict[str, object]:
    class_names = np.unique(labels)
    cluster_ids = np.unique(clusters)
    class_cluster = np.zeros((class_names.size, cluster_ids.size), dtype=np.int64)
    for row, label in enumerate(class_names):
        for col, cluster in enumerate(cluster_ids):
            class_cluster[row, col] = int(np.sum((labels == label) & (clusters == cluster)))
    split_counts = class_cluster.astype(bool).sum(axis=1)
    merged_pair_count = 0
    severe_merged_pair_count = 0
    cluster_class_counts: list[int] = []
    for col in range(cluster_ids.size):
        counts = class_cluster[:, col]
        present = counts[counts > 0]
        cluster_class_counts.append(int(present.size))
        pairs = int(present.size * (present.size - 1) // 2)
        merged_pair_count += pairs
        if present.size >= 3:
            severe_merged_pair_count += pairs
    return {
        "cluster_sizes": {str(int(cluster)): int(np.sum(clusters == cluster)) for cluster in cluster_ids},
        "cluster_size_fraction": {
            str(int(cluster)): float(np.mean(clusters == cluster)) for cluster in cluster_ids
        },
        "class_split_counts": {str(label): int(count) for label, count in zip(class_names, split_counts)},
        "split_class_count": int(np.sum(split_counts >= 2)),
        "severely_split_class_count": int(np.sum(split_counts >= 3)),
        "merged_class_pair_count": merged_pair_count,
        "severely_merged_class_pair_count": severe_merged_pair_count,
        "cluster_class_counts": {
            str(int(cluster)): count for cluster, count in zip(cluster_ids, cluster_class_counts)
        },
        "class_cluster_counts": class_cluster.tolist(),
        "class_order": [str(value) for value in class_names],
        "cluster_order": [int(value) for value in cluster_ids],
    }


def _evaluate(representation: np.ndarray, labels: np.ndarray, seed: int, k: int) -> dict[str, object]:
    representation = np.asarray(representation, dtype=np.float32)
    scaled = StandardScaler().fit_transform(representation)
    clusters = KMeans(
        n_clusters=np.unique(labels).size,
        n_init=20,
        random_state=seed,
    ).fit_predict(scaled)
    neighbor_count = min(k + 1, representation.shape[0])
    neighbors = NearestNeighbors(n_neighbors=neighbor_count, metric="cosine").fit(representation)
    indices = neighbors.kneighbors(representation, return_distance=False)[:, 1:]
    purity = np.mean(labels[indices] == labels[:, None], axis=1)
    return {
        "cells": int(representation.shape[0]),
        "dimensions": int(representation.shape[1]),
        "ari": float(adjusted_rand_score(labels, clusters)),
        "nmi": float(normalized_mutual_info_score(labels, clusters)),
        "knn_purity_at_15": float(np.mean(purity)),
        "knn_purity_per_cell_type": {
            str(label): float(np.mean(purity[labels == label])) for label in np.unique(labels)
        },
        "per_class": _class_metrics(labels, clusters),
        "cluster_diagnostics": _cluster_diagnostics(labels, clusters),
    }


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task15 read-only audit of Task14 LODO outcomes")
    parser.add_argument("--root", type=Path, default=Path("/home/u202522802045/share/zpb"))
    parser.add_argument(
        "--task14-dir",
        type=Path,
        default=Path("/home/u202522802045/share/zpb/result_contain/14_dmode_lodo_snf_student_seed20020130_retry1"),
    )
    parser.add_argument(
        "--source-quality-dir",
        type=Path,
        default=Path("/home/u202522802045/share/zpb/result_contain/pre_task14_dmode_source_quality_20260926"),
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20020130)
    parser.add_argument("--fold", choices=tuple(FOLDS), action="append")
    return parser.parse_args()


def main() -> None:
    args = _parse()
    folds = args.fold or list(FOLDS)
    args.outdir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []
    for fold in folds:
        dataset_key = FOLDS[fold]
        dataset_name = DATASET_NAMES[dataset_key]
        seed_dir = args.task14_dir / fold / f"seed_{args.seed}"
        metadata_path = args.source_quality_dir / fold / "harmonized" / dataset_name / "metadata.npz"
        with np.load(metadata_path, allow_pickle=True) as metadata:
            test_rows = metadata["test_index"].astype(np.int64)
        labels = _load_labels(args.root, dataset_key, test_rows)
        rna = np.load(seed_dir / "z_rna_shared_test.npy")
        student = np.load(seed_dir / "z_student_test.npy")
        if rna.shape[0] != labels.size or student.shape[0] != labels.size:
            raise RuntimeError(f"{fold}: saved representations and labels have different cell counts")
        evaluated = {
            "RNA": _evaluate(rna, labels, args.seed, 15),
            "Student": _evaluate(student, labels, args.seed, 15),
        }
        result = {
            "task_id": 15,
            "fold": fold,
            "dataset": dataset_name,
            "seed": args.seed,
            "labels_source": str(args.root / LABEL_PATHS[dataset_key]),
            "test_index_source": str(metadata_path),
            "training_performed": False,
            "target_atac_accessed": False,
            "representations": {
                "RNA": str(seed_dir / "z_rna_shared_test.npy"),
                "Student": str(seed_dir / "z_student_test.npy"),
            },
            "evaluation": "Task14 StandardScaler + KMeans(n_init=20) and cosine Top15 neighbors",
            "metrics": evaluated,
        }
        _write_json(args.outdir / f"{fold}.json", result)
        rows.append({
            "fold": fold,
            "dataset": dataset_name,
            "seed": args.seed,
            "rna_ari": evaluated["RNA"]["ari"],
            "student_ari": evaluated["Student"]["ari"],
            "rna_nmi": evaluated["RNA"]["nmi"],
            "student_nmi": evaluated["Student"]["nmi"],
            "rna_knn_purity_at_15": evaluated["RNA"]["knn_purity_at_15"],
            "student_knn_purity_at_15": evaluated["Student"]["knn_purity_at_15"],
            "rna_split_class_count": evaluated["RNA"]["cluster_diagnostics"]["split_class_count"],
            "student_split_class_count": evaluated["Student"]["cluster_diagnostics"]["split_class_count"],
            "rna_merged_class_pair_count": evaluated["RNA"]["cluster_diagnostics"]["merged_class_pair_count"],
            "student_merged_class_pair_count": evaluated["Student"]["cluster_diagnostics"]["merged_class_pair_count"],
        })
    with (args.outdir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_json(args.outdir / "manifest.json", {
        "task_id": 15,
        "experiment": "Task14 D-mode LODO outcome audit",
        "git_commit": _git_commit(),
        "seed": args.seed,
        "folds": folds,
        "training_performed": False,
        "teacher_modified": False,
        "student_modified": False,
        "target_atac_accessed": False,
        "labels_used_only_for_diagnosis": True,
        "checkpoint_selection_performed": False,
        "kmeans": {"standard_scaler": True, "n_init": 20, "random_state": args.seed},
        "neighbors": {"metric": "cosine", "k": 15},
    })


if __name__ == "__main__":
    main()
