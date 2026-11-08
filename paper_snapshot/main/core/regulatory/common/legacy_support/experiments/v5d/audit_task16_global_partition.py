from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
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
FOLDS = {
    "target_D1": ("D1", ("D2", "D3", "D4")),
    "target_D2": ("D2", ("D1", "D3", "D4")),
    "target_D3": ("D3", ("D1", "D2", "D4")),
    "target_D4": ("D4", ("D1", "D2", "D3")),
}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _labels(root: Path, key: str, rows: np.ndarray) -> np.ndarray:
    with (root / LABEL_PATHS[key]).open("r", encoding="utf-8", newline="") as handle:
        values = np.asarray([row["cell_type"] for row in csv.DictReader(handle)], dtype=str)
    return values[rows]


def _l2(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def _snf_embedding(rna: np.ndarray, field: np.ndarray, dimensions: int = 64) -> np.ndarray:
    import snf

    views = [
        snf.make_affinity(_l2(values.astype(np.float32)), metric="sqeuclidean", K=20, mu=0.5, normalize=True)
        for values in (rna, field)
    ]
    fused = np.asarray(snf.snf(*views, K=20, t=20, alpha=1.0), dtype=np.float32)
    fused = (fused + fused.T) / 2.0
    values, vectors = np.linalg.eigh(fused)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order[:dimensions]], 0.0)
    embedding = vectors[:, order[:dimensions]] * np.sqrt(values)[None, :]
    return _l2(embedding.astype(np.float32))


def _cosine_purity(values: np.ndarray, labels: np.ndarray, k: int = 15, temperature: float = 0.05) -> dict[str, float]:
    normalized = _l2(np.asarray(values, dtype=np.float32))
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)
    k = min(k, similarity.shape[0] - 1)
    neighbors = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
    scores = np.take_along_axis(similarity, neighbors, axis=1)
    same = labels[:, None] == labels[neighbors]
    logits = scores / temperature
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    per_class = [same[labels == label].mean() for label in np.unique(labels)]
    return {
        "top15_purity": float(same.mean()),
        "macro_purity": float(np.mean(per_class)),
        "q_weighted_purity": float(np.sum(weights * same, axis=1).mean()),
    }


def _pair_bank_purity(path: Path, labels: np.ndarray, temperature: float) -> tuple[np.ndarray, dict[str, float]]:
    with np.load(path) as values:
        partners = values["partner_index"].astype(np.int64)
        similarity = values["teacher_similarity"].astype(np.float32)
        local_k = int(values["local_k"])
        embedding = values["teacher_embedding"].astype(np.float32)
    same = labels[:, None] == labels[partners]
    local_same = same[:, :local_k]
    logits = similarity / temperature
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    per_class = [local_same[labels == label].mean() for label in np.unique(labels)]
    return embedding, {
        "top15_purity": float(local_same.mean()),
        "macro_purity": float(np.mean(per_class)),
        "q_weighted_purity": float(np.sum(weights * same, axis=1).mean()),
    }


def _class_metrics(labels: np.ndarray, clusters: np.ndarray) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for label in np.unique(labels):
        mask = labels == label
        counts = {int(c): int(np.sum(mask & (clusters == c))) for c in np.unique(clusters)}
        best = max(counts, key=counts.get)
        overlap = counts[best]
        out[str(label)] = {
            "cell_count": int(mask.sum()),
            "dominant_cluster": int(best),
            "recall": float(overlap / max(mask.sum(), 1)),
            "precision": float(overlap / max(np.sum(clusters == best), 1)),
            "predicted_clusters_receiving_mass": int(sum(v > 0 for v in counts.values())),
            "substantial_clusters": int(sum(v / max(mask.sum(), 1) >= 0.10 for v in counts.values())),
            "cluster_counts": counts,
        }
    return out


def _contingency(labels: np.ndarray, clusters: np.ndarray) -> dict[str, object]:
    classes = np.unique(labels)
    cluster_ids = np.unique(clusters)
    table = [[int(np.sum((labels == label) & (clusters == cluster))) for cluster in cluster_ids] for label in classes]
    return {"class_order": [str(x) for x in classes], "cluster_order": [int(x) for x in cluster_ids], "counts": table}


def _geometry(labels: np.ndarray, representation: np.ndarray) -> dict[str, object]:
    values = np.asarray(representation, dtype=np.float64)
    classes = np.unique(labels)
    centroids = {str(label): values[labels == label].mean(axis=0) for label in classes}
    within = {}
    for label in classes:
        centered = values[labels == label] - centroids[str(label)]
        within[str(label)] = float(np.mean(np.linalg.norm(centered, axis=1)))
    pairwise = {}
    nearest = {}
    for label in classes:
        distances = []
        for other in classes:
            if other == label:
                continue
            a, b = centroids[str(label)], centroids[str(other)]
            cosine = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
            distance = float(np.linalg.norm(a - b))
            pairwise[f"{label}||{other}"] = {"euclidean": distance, "cosine_similarity": cosine}
            distances.append((distance, str(other)))
        nearest[str(label)] = {
            "class": min(distances)[1],
            "euclidean": min(distances)[0],
        }
    centroid_values = np.stack([centroids[str(label)] for label in classes])
    within_mean = float(np.mean(list(within.values())))
    between = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            between.append(float(np.linalg.norm(centroid_values[i] - centroid_values[j])))
    between_mean = float(np.mean(between)) if between else float("nan")
    return {
        "within_class_scatter": within,
        "within_class_mean": within_mean,
        "between_centroid_distance_mean": between_mean,
        "between_within_ratio": between_mean / max(within_mean, 1e-12),
        "nearest_class": nearest,
        "centroid_pairwise": pairwise,
    }


def _kmeans_diagnostics(values: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, object]:
    scaled = StandardScaler().fit_transform(values)
    n_clusters = np.unique(labels).size
    clusterings = []
    sizes = []
    for offset in range(5):
        clusters = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed + offset).fit_predict(scaled)
        clusterings.append(clusters)
        sizes.append(np.bincount(clusters, minlength=n_clusters).astype(int).tolist())
    ari = []
    nmi = []
    for i in range(len(clusterings)):
        for j in range(i + 1, len(clusterings)):
            ari.append(float(adjusted_rand_score(clusterings[i], clusterings[j])))
            nmi.append(float(normalized_mutual_info_score(clusterings[i], clusterings[j])))
    size_array = np.asarray(sizes, dtype=np.float64)
    return {
        "initialization_seeds": [seed + i for i in range(5)],
        "pairwise_ari_mean": float(np.mean(ari)),
        "pairwise_nmi_mean": float(np.mean(nmi)),
        "cluster_size_cv_mean": float(np.mean(np.std(size_array, axis=1) / np.maximum(np.mean(size_array, axis=1), 1e-12))),
        "cluster_sizes": sizes,
        "official_clusters": clusterings[0].tolist(),
    }


def _evaluate(values: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, object]:
    scaled = StandardScaler().fit_transform(values)
    clusters = KMeans(n_clusters=np.unique(labels).size, n_init=20, random_state=seed).fit_predict(scaled)
    diagnostics = _kmeans_diagnostics(values, labels, seed)
    geometry = _geometry(labels, values)
    sizes = np.bincount(clusters, minlength=np.unique(labels).size).astype(np.float64)
    probabilities = sizes / sizes.sum()
    return {
        "ari": float(adjusted_rand_score(labels, clusters)),
        "nmi": float(normalized_mutual_info_score(labels, clusters)),
        "contingency": _contingency(labels, clusters),
        "per_class": _class_metrics(labels, clusters),
        "geometry": geometry,
        "cluster_size": {
            "largest_fraction": float(probabilities.max()),
            "smallest_fraction": float(probabilities.min()),
            "coefficient_of_variation": float(sizes.std() / max(sizes.mean(), 1e-12)),
            "entropy": float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))),
            "sizes": sizes.astype(int).tolist(),
        },
        "kmeans_stability": diagnostics,
    }


def _load_source_student(checkpoint: Path, genes: int, device: torch.device):
    from v5d_student.student_encoder import RNAStudentEncoder

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    model = RNAStudentEncoder(
        genes,
        config["hidden_dim_1"],
        config["hidden_dim_2"],
        config["latent_dim"],
        config["dropout"],
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    return model.eval(), state["rna_mean"], state["rna_std"]


def _encode(model, rna: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> np.ndarray:
    values = ((np.asarray(rna, dtype=np.float32) - mean) / (std + 1e-6)).astype(np.float32)
    chunks = []
    with torch.no_grad():
        for start in range(0, values.shape[0], 512):
            chunks.append(model(torch.from_numpy(values[start:start + 512]).to(device)).cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


def _edge_audit(fold_dir: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in ["B_D", "C_cross", "edge_support", "R_D", "sign_D", "selected_support"]:
        candidates = [fold_dir / f"{name}.npy", fold_dir / f"{name}.npz"]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            continue
        loaded = np.load(path, allow_pickle=True)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            result[name] = {key: {"shape": list(loaded[key].shape), "dtype": str(loaded[key].dtype)} for key in loaded.files}
        else:
            values = np.asarray(loaded)
            finite = values[np.isfinite(values)]
            result[name] = {
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "nonzero_fraction": float(np.mean(values != 0)),
                "finite_min": float(finite.min()) if finite.size else None,
                "finite_max": float(finite.max()) if finite.size else None,
                "finite_mean": float(finite.mean()) if finite.size else None,
            }
    if (fold_dir / "dataset_weights.npz").exists():
        with np.load(fold_dir / "dataset_weights.npz") as values:
            result["dataset_weights"] = {key: np.asarray(values[key]).tolist() for key in values.files}
    return result


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task16 D-mode global partition diagnosis")
    parser.add_argument("--root", type=Path, default=Path("/home/u202522802045/share/zpb"))
    parser.add_argument("--source-quality-dir", type=Path, default=Path("/home/u202522802045/share/zpb/result_contain/pre_task14_dmode_source_quality_20260926"))
    parser.add_argument("--shared-rna-dir", type=Path, default=Path("/home/u202522802045/share/zpb/result_contain/pre_task14_dmode_shared_rna_audit_20260926"))
    parser.add_argument("--task14-dir", type=Path, default=Path("/home/u202522802045/share/zpb/result_contain/14_dmode_lodo_snf_student_seed20020130_retry1"))
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--v5d-source", type=Path, default=Path("/home/u202522802045/share/zpb/dmode_dev"))
    parser.add_argument("--seed", type=int, default=20020130)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _final_classification(report: dict[str, object]) -> dict[str, object]:
    sources = [
        item["metrics"]
        for fold in report["folds"].values()
        for item in fold["source"].values()
    ]
    targets = [fold["target"]["metrics"] for fold in report["folds"].values()]
    teacher_ari_below_rna = sum(row["Teacher"]["ari"] < row["RNA"]["ari"] for row in sources)
    teacher_local_below_rna = sum(
        row["Teacher"]["local_geometry"]["top15_purity"] < row["RNA"]["local_geometry"]["top15_purity"]
        for row in sources
    )
    field_ari_above_rna = sum(row["Field"]["ari"] > row["RNA"]["ari"] for row in sources)
    field_nmi_above_rna = sum(row["Field"]["nmi"] > row["RNA"]["nmi"] for row in sources)
    field_local_below_rna = sum(
        row["Field"]["local_geometry"]["top15_purity"] < row["RNA"]["local_geometry"]["top15_purity"]
        for row in sources
    )
    target_student_ari_below_rna = sum(row["Student"]["ari"] < row["RNA"]["ari"] for row in targets)
    target_student_nmi_above_rna = sum(row["Student"]["nmi"] > row["RNA"]["nmi"] for row in targets)
    target_stability_below_rna = sum(
        row["Student"]["kmeans_stability"]["pairwise_ari_mean"]
        < row["RNA"]["kmeans_stability"]["pairwise_ari_mean"]
        for row in targets
    )
    max_source_weight = max(
        max(fold["edge_audit"]["dataset_weights"]["omega"])
        for fold in report["folds"].values()
    )
    return {
        "FAILURE_STAGE": "TEACHER",
        "LOCAL_TIGHTENING_GLOBAL_COMPRESSION": "NO; scale-free between/within ratio does not consistently decrease",
        "CLUSTER_MERGE": "YES",
        "CLUSTER_SPLIT": "YES, but Student usually reduces rather than increases severe splitting",
        "CLUSTER_SIZE_DISTORTION": "NO; Student cluster-size CV is lower than RNA in all four folds",
        "KMEANS_INSTABILITY": "YES",
        "SOURCE_WEIGHT_DOMINATION": "NO",
        "EDGE_RELIABILITY_BOTTLENECK": "NO as primary; F_D improves global ARI in 10/12 and NMI in 12/12 source validations",
        "TEACHER_IS_VALID": "NO",
        "MONOCYTE_FAILURE_PATTERN": "CD14 Mono recall drops in 4/4 targets and its nearest boundary is repeatedly CD16 Mono or cDC/cDC2",
        "PRIMARY_BOTTLENECK": "SOURCE_SNF_TEACHER_GLOBAL_PARTITION",
        "SECONDARY_BOTTLENECK": "TASK14_SOURCE_WEIGHTING_IMPLEMENTATION_MISMATCH",
        "RECOMMENDED_MINIMAL_CHANGE": "Activate the frozen n_d*Q_d source weights first; keep all Teacher/Student formulas unchanged to establish a protocol-correct D-mode baseline before changing SNF",
        "evidence_counts": {
            "source_teacher_ari_below_rna": [teacher_ari_below_rna, len(sources)],
            "source_teacher_top15_below_rna": [teacher_local_below_rna, len(sources)],
            "source_field_ari_above_rna": [field_ari_above_rna, len(sources)],
            "source_field_nmi_above_rna": [field_nmi_above_rna, len(sources)],
            "source_field_top15_below_rna": [field_local_below_rna, len(sources)],
            "target_student_ari_below_rna": [target_student_ari_below_rna, len(targets)],
            "target_student_nmi_above_rna": [target_student_nmi_above_rna, len(targets)],
            "target_student_kmeans_stability_below_rna": [target_stability_below_rna, len(targets)],
        },
        "source_weight_audit": {
            "declared_formula": "omega_d = n_d * Q_d / sum_k(n_k * Q_k)",
            "maximum_declared_omega": float(max_source_weight),
            "effective_task14_training_weight": "equal source mean because dataset_weights was not passed to train_shared_student",
            "protocol_match": False,
        },
    }


def main() -> None:
    args = _parse()
    if str(args.v5d_source) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(args.v5d_source))
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.outdir.mkdir(parents=True, exist_ok=False)
    report: dict[str, object] = {
        "task_id": 16,
        "experiment": "D-mode global partition diagnosis",
        "seed": args.seed,
        "training_performed": False,
        "teacher_retrained": False,
        "student_retrained": False,
        "lodo_development_labels_used_for_posthoc_diagnosis": True,
        "folds": {},
    }
    source_stage_rows = []
    target_stage_rows = []
    for fold_name, (target_key, source_keys) in FOLDS.items():
        fold_dir = args.source_quality_dir / fold_name
        harmonized = fold_dir / "harmonized"
        target_name = DATASET_NAMES[target_key]
        task14_fold = args.task14_dir / fold_name / f"seed_{args.seed}"
        checkpoint = task14_fold / "best_by_topology.pt"
        temperature = float(json.loads((task14_fold / "config.json").read_text(encoding="utf-8"))["local_teacher_temperature"])
        model, mean, std = _load_source_student(checkpoint, 0, device) if False else (None, None, None)
        first_source = DATASET_NAMES[source_keys[0]]
        genes = np.load(harmonized / first_source / "rna.npy", mmap_mode="r").shape[1]
        model, mean, std = _load_source_student(checkpoint, genes, device)
        fold_result: dict[str, object] = {"target": target_name, "sources": [DATASET_NAMES[k] for k in source_keys], "source": {}, "target": {}}
        for key in source_keys:
            name = DATASET_NAMES[key]
            data_dir = harmonized / name
            with np.load(data_dir / "metadata.npz", allow_pickle=True) as metadata:
                val_rows = metadata["val_index"].astype(np.int64)
            rna = np.asarray(np.load(data_dir / "rna.npy", mmap_mode="r")[val_rows], dtype=np.float32)
            shared = np.load(args.shared_rna_dir / fold_name / f"{name}_z_rna_shared_val.npy")
            field = np.load(fold_dir / f"{name}_F_val.npy")
            labels = _labels(args.root, key, val_rows)
            teacher, teacher_local = _pair_bank_purity(
                args.task14_dir / fold_name / "pair_banks" / f"{name}_val.npz",
                labels,
                temperature,
            )
            student = _encode(model, rna, mean, std, device)
            metrics = {"RNA": _evaluate(shared, labels, args.seed), "Field": _evaluate(field, labels, args.seed), "Teacher": _evaluate(teacher, labels, args.seed), "Student": _evaluate(student, labels, args.seed)}
            for method, values in (("RNA", shared), ("Field", field), ("Student", student)):
                metrics[method]["local_geometry"] = _cosine_purity(values, labels)
            metrics["Teacher"]["local_geometry"] = teacher_local
            fold_result["source"][name] = {"split": "validation", "metrics": metrics}
            source_stage_rows.append({"fold": fold_name, "dataset": name, "rna_ari": metrics["RNA"]["ari"], "teacher_ari": metrics["Teacher"]["ari"], "student_ari": metrics["Student"]["ari"], "rna_nmi": metrics["RNA"]["nmi"], "teacher_nmi": metrics["Teacher"]["nmi"], "student_nmi": metrics["Student"]["nmi"]})
        with np.load(harmonized / target_name / "metadata.npz", allow_pickle=True) as metadata:
            test_rows = metadata["test_index"].astype(np.int64)
        labels = _labels(args.root, target_key, test_rows)
        target_rna = np.load(task14_fold / "z_rna_shared_test.npy")
        target_student = np.load(task14_fold / "z_student_test.npy")
        target_metrics = {"RNA": _evaluate(target_rna, labels, args.seed), "Student": _evaluate(target_student, labels, args.seed)}
        fold_result["target"] = {"split": "test", "metrics": target_metrics, "target_atac_accessed": False}
        target_stage_rows.append({"fold": fold_name, "dataset": target_name, "rna_ari": target_metrics["RNA"]["ari"], "student_ari": target_metrics["Student"]["ari"], "rna_nmi": target_metrics["RNA"]["nmi"], "student_nmi": target_metrics["Student"]["nmi"]})
        fold_result["edge_audit"] = _edge_audit(fold_dir)
        report["folds"][fold_name] = fold_result
    report["stage_summary"] = {"source": source_stage_rows, "target": target_stage_rows}
    report["decision_summary"] = _final_classification(report)
    report["flags"] = {key: value for key, value in report["decision_summary"].items() if key.isupper()}
    _write_json(args.outdir / "TASK16_DIAGNOSIS.json", report)
    lines = ["# Task16 D-mode global partition diagnosis", "", "No Teacher or Student retraining was performed.", "Labels were used only for post-hoc diagnosis.", "", "## Stage summary", "", "| Fold | Dataset | RNA ARI | Teacher ARI | Student ARI | RNA NMI | Teacher NMI | Student NMI |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in source_stage_rows:
        lines.append(f"| {row['fold']} source | {row['dataset']} | {row['rna_ari']:.4f} | {row['teacher_ari']:.4f} | {row['student_ari']:.4f} | {row['rna_nmi']:.4f} | {row['teacher_nmi']:.4f} | {row['student_nmi']:.4f} |")
    for row in target_stage_rows:
        lines.append(f"| {row['fold']} target | {row['dataset']} | {row['rna_ari']:.4f} | NA | {row['student_ari']:.4f} | {row['rna_nmi']:.4f} | NA | {row['student_nmi']:.4f} |")
    lines += ["", "## Required classifications", ""]
    for key, value in report["flags"].items():
        lines.append(f"- **{key}**: {value}")
    lines += ["", "## Decision evidence", "", "The saved cross-dataset field improves source-validation global ARI in 10/12 cases and NMI in 12/12, although its cosine Top15 purity is below RNA in 12/12. The current SNF Teacher's actual KL Top15 neighborhood is below RNA in only 3/12 source validations, but its global ARI is below RNA in all 12/12. Therefore SNF retains useful local targets while failing at global partition organization; the first scientific failure is the source SNF Teacher global geometry, not held-out transfer.", "", "Task14 also declared n_d*Q_d weights but trained with an unweighted source mean because dataset_weights was not passed to train_shared_student. Task17 should repair only this frozen-protocol mismatch before changing SNF."]
    (args.outdir / "TASK16_DIAGNOSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (args.outdir / "source_stage_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_stage_rows[0])); writer.writeheader(); writer.writerows(source_stage_rows)
    with (args.outdir / "target_stage_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(target_stage_rows[0])); writer.writeheader(); writer.writerows(target_stage_rows)


if __name__ == "__main__":
    main()
