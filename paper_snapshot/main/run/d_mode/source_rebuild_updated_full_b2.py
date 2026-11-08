from __future__ import annotations

import argparse
import csv
import json
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path('/home/u202522802045/share/zpb/main/core/regulatory/common/legacy_support')
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v5d.data import V5DData
from v5d.dmode_reliable_selector import DModeReliabilityConfig, build_source_only_b
import v5d.dmode_reliable_selector as selector
from v5d.dmode_teacher import fit_dmode_field
from v5d.harmonize import prepare_harmonized
from v5d.regulatory_graph import normalize_chromosome, parse_peak, read_tss
from v5d.dataset_spec import DatasetSpec


DATASET_PATHS = {
    "D1": "main/result/d_mode/corrected_pbmc_legacy/source_inputs/D1",
    "D2": "main/result/d_mode/corrected_pbmc_legacy/source_inputs/D2",
    "D3": "main/result/d_mode/corrected_pbmc_legacy/source_inputs/D3",
    "D4": "main/result/d_mode/corrected_pbmc_legacy/source_inputs/D4",
}
DATASET_NAMES = {
    "D1": "PBMC-MO-001",
    "D2": "PBMC-MO-002",
    "D3": "PBMC-MO-003",
    "D4": "PBMC-MO-004",
}
LABEL_PATHS = {key: "" for key in DATASET_PATHS}
FOLDS = {
    "target_D1": ("D1", ("D2", "D3", "D4")),
    "target_D2": ("D2", ("D1", "D3", "D4")),
    "target_D3": ("D3", ("D1", "D2", "D4")),
    "target_D4": ("D4", ("D1", "D2", "D3")),
}


def build_candidate_graph(gene_ids, peak_names, tss, window_bp, max_candidates):
    by_chrom = {}
    for peak_index, peak_name in enumerate(peak_names):
        chrom, start, end = parse_peak(str(peak_name))
        by_chrom.setdefault(chrom, []).append(((start + end) // 2, peak_index))
    for values in by_chrom.values():
        values.sort()
    indexed = {chrom: (values, [item[0] for item in values]) for chrom, values in by_chrom.items()}
    candidates = []
    for gene in map(str, gene_ids):
        gene_key = gene.split('.')[0]
        if gene_key not in tss:
            raise ValueError(f'Missing TSS for common gene {gene}')
        chrom, position = tss[gene_key]
        values, positions = indexed.get(normalize_chromosome(chrom), ([], []))
        left = bisect_left(positions, position - window_bp)
        right = bisect_right(positions, position + window_bp)
        nearest = sorted((abs(midpoint - position), peak_index) for midpoint, peak_index in values[left:right])[:max_candidates]
        candidates.append(nearest)
    selected = np.asarray(sorted({peak for values in candidates for _, peak in values}), dtype=np.int64)
    index = np.zeros((len(gene_ids), max_candidates), dtype=np.int64)
    mask = np.zeros_like(index, dtype=bool)
    distance = np.zeros_like(index, dtype=np.float32)
    for gene_index, values in enumerate(candidates):
        for column, (distance_value, peak_index) in enumerate(values):
            index[gene_index, column] = peak_index
            mask[gene_index, column] = True
            distance[gene_index, column] = distance_value
    return {'candidate_index': index, 'candidate_mask': mask, 'distance_bp': distance}, selected


def _edge_rho_cached(rna, atac, rows, edge_gene, edge_peak):
    rna_rank = selector._normalized_ranks(np.asarray(rna[rows], dtype=np.float32))
    peaks = np.unique(edge_peak)
    atac_rank = selector._normalized_ranks(atac[rows][:, peaks].toarray().astype(np.float32))
    peak_columns = np.searchsorted(peaks, edge_peak)
    result = np.empty(edge_gene.size, dtype=np.float32)
    for start in range(0, edge_gene.size, 16384):
        stop = min(start + 16384, edge_gene.size)
        result[start:stop] = np.sum(
            rna_rank[:, edge_gene[start:stop]] * atac_rank[:, peak_columns[start:stop]], axis=0
        )
    return np.nan_to_num(result, nan=0.0).astype(np.float32)


selector._edge_rho = _edge_rho_cached


def _l2(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def _snf_q(rna: np.ndarray, field: np.ndarray) -> np.ndarray:
    import snf

    views = []
    for values in (rna, field):
        values = _l2(np.asarray(values, dtype=np.float32))
        views.append(snf.make_affinity(values, metric="sqeuclidean", K=20, mu=0.5, normalize=True))
    fused = np.asarray(snf.snf(*views, K=20, t=20, alpha=1.0), dtype=np.float32)
    np.fill_diagonal(fused, 0.0)
    return fused / np.maximum(fused.sum(axis=1, keepdims=True), 1e-12)


def _purity(q: np.ndarray, labels: np.ndarray, k: int = 15) -> dict[str, float]:
    q = np.asarray(q, dtype=np.float32).copy()
    np.fill_diagonal(q, 0.0)
    k = min(k, q.shape[0] - 1)
    neighbors = np.argpartition(-q, k - 1, axis=1)[:, :k]
    scores = np.take_along_axis(q, neighbors, axis=1)
    same = labels[:, None] == labels[neighbors]
    top = same.mean(axis=1)
    weights = scores / np.maximum(scores.sum(axis=1, keepdims=True), 1e-12)
    per_type = {
        str(label): float(top[labels == label].mean())
        for label in np.unique(labels)
    }
    return {
        "top15_purity": float(top.mean()),
        "macro_purity": float(np.mean(list(per_type.values()))),
        "q_weighted_purity": float(np.sum(weights * same, axis=1).mean()),
        "per_cell_type_purity": per_type,
    }


def _labels(data: V5DData, rows: np.ndarray, fallback_path: Path | None = None) -> np.ndarray:
    with np.load(data.root / "metadata.npz", allow_pickle=True) as metadata:
        values = metadata.get("labels")
    if values is not None and values.size:
        return values[rows].astype(str)
    if fallback_path is None or not fallback_path.exists():
        raise RuntimeError(f"{data.name}: validation labels are required for quality evaluation")
    with fallback_path.open("r", encoding="utf-8", newline="") as handle:
        values = np.asarray([row["cell_type"] for row in csv.DictReader(handle)], dtype=str)
    if values.size != data.rna.shape[0]:
        raise RuntimeError(f"{data.name}: label count does not match harmonized cells")
    return values[rows]


def _shared_rna_projection(sources: list[V5DData]) -> tuple[dict[str, np.ndarray], dict]:
    train_blocks = [np.asarray(source.rna[source.train], dtype=np.float32) for source in sources]
    scaler = StandardScaler().fit(np.concatenate(train_blocks, axis=0))
    transformed = [scaler.transform(block) for block in train_blocks]
    components = min(64, transformed[0].shape[1], sum(block.shape[0] for block in transformed) - 1)
    reducer = PCA(n_components=components, svd_solver="randomized", random_state=0)
    reducer.fit(np.concatenate(transformed, axis=0))
    result = {}
    for source in sources:
        values = scaler.transform(np.asarray(source.rna, dtype=np.float32))
        result[source.name] = reducer.transform(values).astype(np.float32)
    return result, {
        "representation": "one source-train-fitted StandardScaler + PCA",
        "fit_datasets": [source.name for source in sources],
        "fit_split": "train_only",
        "components": int(components),
        "target_inference_input": "RNA only with source-missing genes zero-filled by harmonization",
    }


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/u202522802045/share/zpb"))
    parser.add_argument("--outdir", type=Path, default=Path("/home/u202522802045/share/zpb/main/result/d_mode/corrected_pbmc_legacy/source_artifacts"))
    parser.add_argument("--gtf", type=Path, default=Path("/home/u202522802045/glue_156478/graph_files/gencode.v32.primary_assembly.annotation.gtf.gz"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    raise RuntimeError(
        "BLOCKED: candidate staging bypasses historical SOURCE HVG/RNA normalization, "
        "and prepare_harmonized loads held-out target ATAC. Do not reuse these artifacts."
    )
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.outdir}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    specs = [DatasetSpec(DATASET_NAMES[key], args.root / DATASET_PATHS[key]) for key in DATASET_PATHS]
    tss = read_tss(args.gtf)
    config = DModeReliabilityConfig()
    fold_reports = {}
    for fold_name, (target_key, source_keys) in FOLDS.items():
        source_names = [DATASET_NAMES[key] for key in source_keys]
        target_name = DATASET_NAMES[target_key]
        fold_dir = args.outdir / fold_name
        fold_dir.mkdir()
        harmonized_dir = fold_dir / "harmonized"
        if not harmonized_dir.exists():
            prepare_harmonized(specs, harmonized_dir, feature_reference_names=source_names, held_out_target_name=target_name)
        gene_ids = np.load(harmonized_dir / "common_gene_ids.npy", allow_pickle=True).astype(str)
        peak_names = np.load(harmonized_dir / "canonical_peak_names.npy", allow_pickle=True).astype(str)
        graph, _ = build_candidate_graph(gene_ids, peak_names, tss, 250_000, 50)
        graph["peak_names"] = peak_names
        sources = [V5DData(harmonized_dir, name) for name in source_names]
        selection = build_source_only_b(sources, graph, gene_ids, config, 20020130)
        np.save(fold_dir / "gene_ids.npy", gene_ids)
        np.save(fold_dir / "canonical_peak_names.npy", peak_names)
        np.save(fold_dir / "edge_support.npy", selection["edge_support"])
        np.save(fold_dir / "C_cross.npy", selection["C_cross"])
        np.save(fold_dir / "R_D.npy", selection["R_D"])
        np.save(fold_dir / "sign_D.npy", selection["sign_D"])
        np.save(fold_dir / "selected_support.npy", selection["selected"])
        np.savez_compressed(fold_dir / "dataset_weights.npz", n_train=selection["n_train"], Q_d=selection["Q_d"], omega=selection["omega"])
        from scipy import sparse
        sparse.save_npz(fold_dir / "B_D.npz", selection["B_D"])
        for index, source in enumerate(sources):
            np.save(fold_dir / f"{source.name}_fold_rho_10.npy", selection["fold_rho"][index])
            np.save(fold_dir / f"{source.name}_median_rho10.npy", selection["median_rho"][index])
            np.save(fold_dir / f"{source.name}_C_sign10.npy", selection["C_sign"][index])
            np.save(fold_dir / f"{source.name}_S10.npy", selection["S_within"][index])

        shared_rna, rna_manifest = _shared_rna_projection(sources)
        np.savez_compressed(fold_dir / "shared_rna_pca.npz", **{name: value for name, value in shared_rna.items()})
        source_reports = {}
        quality_flags = []
        for source in sources:
            field, field_state, field_manifest = fit_dmode_field(source.atac, selection["B_D"], source.train)
            np.save(fold_dir / f"{source.name}_F_train.npy", field[source.train])
            np.save(fold_dir / f"{source.name}_F_val.npy", field[source.val])
            train_f = field[source.train]
            f_reducer = PCA(n_components=min(64, train_f.shape[1], train_f.shape[0] - 1), svd_solver="randomized", random_state=0).fit(train_f)
            rna_val = _l2(shared_rna[source.name][source.val])
            f_val = _l2(f_reducer.transform(field[source.val]).astype(np.float32))
            teacher_q = _snf_q(rna_val, f_val)
            rna_q = rna_val @ rna_val.T
            source_key = next(key for key, value in DATASET_NAMES.items() if value == source.name)
            labels = _labels(source, source.val, args.root / LABEL_PATHS[source_key])
            rna_metrics = _purity(rna_q, labels)
            teacher_metrics = _purity(teacher_q, labels)
            source_reports[source.name] = {
                "train_cells": int(source.train.size),
                "val_cells": int(source.val.size),
                "RNA": rna_metrics,
                "Teacher_SNF": teacher_metrics,
                "teacher_top15_non_degrading": bool(teacher_metrics["top15_purity"] >= rna_metrics["top15_purity"] - 1e-6),
                "field_manifest": field_manifest,
                "field_state_shapes": {key: list(value.shape) for key, value in field_state.items()},
                "snf_config": {"implementation": "snfpy", "affinity_metric": "sqeuclidean", "K": 20, "mu": 0.5, "normalize": True, "t": 20, "alpha": 1.0},
            }
            quality_flags.append(source_reports[source.name]["teacher_top15_non_degrading"])
        fold_reports[fold_name] = {
            "target": target_name,
            "sources": source_names,
            "shared_rna": rna_manifest,
            "selection": selection["manifest"],
            "dataset_weights": {"n_train": selection["n_train"], "Q_d": selection["Q_d"], "omega": selection["omega"]},
            "source_quality": source_reports,
            "source_teacher_quality_gate": bool(sum(quality_flags) >= 2),
            "target_atac_used": False,
            "target_labels_used_for_selection": False,
            "canonical_peak_collision_aggregation": "binary_any_hit_after_deterministic_max_overlap_mapping",
        }
        (fold_dir / "fold_manifest.json").write_text(json.dumps(_json_safe(fold_reports[fold_name]), indent=2), encoding="utf-8")
    overall = {
        "experiment": "pre-Task14 D-mode source Teacher quality audit",
        "training_performed": False,
        "student_training_performed": False,
        "task14_reserved": False,
        "seed": 20020130,
        "folds": fold_reports,
        "d_mode_formal_run_ready": bool(all(report["source_teacher_quality_gate"] for report in fold_reports.values())),
    }
    (args.outdir / "SUMMARY.json").write_text(json.dumps(_json_safe(overall), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
