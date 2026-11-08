from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .config import V5SConfig
from .data import V5SData, prepare_dataset
from .edge_selector import select_soft_edges
from .evaluate_geometry import evaluate_saved_test_geometry
from .linear_teacher import compute_linear_teacher
from .utils import git_commit, set_seed, write_json


SEEDS = (20020130, 20020131, 20020132, 20020133, 20020134)
CONDITIONS = (
    "paired_graph_linear",
    "shuffled_graph_linear",
    "cell_shuffled_negative_control_linear",
)

SELECTION_ARTIFACTS = (
    "B_reg.npy",
    "gene_confidence.npy",
    "edge_statistics.json",
    "edges.json",
    "manifest.json",
    "regulatory_graph.npz",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V5-S Stage 1 Teacher regulatory geometry")
    parser.add_argument("--raw-h5", type=Path, required=True)
    parser.add_argument("--historical-dir", type=Path, required=True)
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def copy_shared_artifacts(selection_dir: Path, condition_dir: Path) -> None:
    for name in ("edge_statistics.json", "edges.json", "manifest.json", "B_reg.npy", "gene_confidence.npy", "regulatory_graph.npz"):
        shutil.copy2(selection_dir / name, condition_dir / ("edge_" + name if name == "manifest.json" else name))


def write_selection(
    data_dir: Path,
    selection_dir: Path,
    config: V5SConfig,
    seed: int,
    atac_row_permutation: np.ndarray,
) -> None:
    result = select_soft_edges(
        data_dir,
        config.consistency_folds,
        seed,
        config.predictability_epsilon,
        config.gene_confidence_epsilon,
        atac_row_permutation,
    )
    selection_dir.mkdir(parents=True, exist_ok=False)
    for name in ("fold_rho", "median_rho", "sign_stability", "rho_mad", "rho_iqr", "Q_S", "B_reg", "gene_confidence"):
        np.save(selection_dir / f"{name}.npy", result[name])
    np.savez_compressed(selection_dir / "regulatory_graph.npz", **result["graph"])
    (selection_dir / "edges.json").write_text(json.dumps(result["edges"], indent=2), encoding="utf-8")
    write_json(selection_dir / "edge_statistics.json", result["statistics"])
    write_json(selection_dir / "manifest.json", result["manifest"])


def ensure_selection(
    data_dir: Path,
    selection_dir: Path,
    config: V5SConfig,
    seed: int,
    atac_row_permutation: np.ndarray,
) -> None:
    if selection_dir.exists():
        if not all((selection_dir / name).exists() for name in SELECTION_ARTIFACTS):
            raise RuntimeError(f"Existing V5-S selection directory is incomplete: {selection_dir}")
        manifest = json.loads((selection_dir / "manifest.json").read_text(encoding="utf-8"))
        if "Q_gp" not in manifest.get("score", ""):
            raise RuntimeError(f"Selection directory is not the V5-S soft graph: {selection_dir}")
        return
    write_selection(data_dir, selection_dir, config, seed, atac_row_permutation)


def save_representations(z: np.ndarray, data: V5SData, outdir: Path) -> None:
    np.save(outdir / "z_reg_train.npy", z[data.train])
    np.save(outdir / "z_reg_val.npy", z[data.val])
    np.save(outdir / "z_reg_test.npy", z[data.test])


def make_manifest(
    data: V5SData,
    seed: int,
    condition: str,
    config: V5SConfig,
    graph_name: str,
    selection_shuffle_seed: int,
    cell_shuffle_seed: int | None,
) -> dict:
    return {
        "experiment": "V5-S Stage 1 Teacher regulatory geometry",
        "condition": condition,
        "selector_graph": graph_name,
        "dataset": "PBMC10k paired RNA+ATAC source",
        "seed": seed,
        "git_commit": git_commit(),
        "data_split": {"train": int(data.train.size), "val": int(data.val.size), "test": int(data.test.size)},
        "model_genes": int(data.rna.shape[1]),
        "atac_peaks": int(data.atac.shape[1]),
        "tss_window_bp": config.window_bp,
        "max_candidates_per_gene": config.max_candidates,
        "edge_selection_folds": config.consistency_folds,
        "shared_b_reg_across_aggregators": True,
        "paired_and_shuffled_use_distinct_graphs": True,
        "selection_shuffle_seed": selection_shuffle_seed if graph_name == "shuffled" else None,
        "cell_shuffle_seed": cell_shuffle_seed,
        "labels_used_for_training": False,
        "labels_used_for_checkpoint_selection": False,
        "test_used_for_selection": False,
        "loss": "none",
        "topology_supervised_loss": False,
        "gain_loss": False,
        "pair_loss": False,
        "student_training": False,
        "rna_predictability_soft_weighting": True,
        "gene_confidence_weighting": "sqrt(c_g_RNA) after train-stat normalization and before geometry",
        "architecture": config.to_dict(),
    }


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    config = V5SConfig()
    if args.max_epochs is not None:
        config = V5SConfig(**{**config.to_dict(), "max_epochs": args.max_epochs})
    if args.batch_size is not None:
        config = V5SConfig(**{**config.to_dict(), "batch_size": args.batch_size})
    args.outdir.mkdir(parents=True, exist_ok=False)
    if args.prepare_data:
        prepare_dataset(args.raw_h5, args.historical_dir, args.gtf, args.data_dir, config.window_bp, config.max_candidates)
    if not (args.data_dir / "atac_tfidf.npz").exists():
        raise FileNotFoundError("Prepared V5-S data not found; use --prepare-data")

    rna = np.load(args.data_dir / "rna.npy", mmap_mode="r")
    with np.load(args.data_dir / "metadata.npz", allow_pickle=True) as metadata:
        train_index = metadata["train_index"].astype(np.int64)
        test_index = metadata["test_index"].astype(np.int64)
    identity = np.arange(rna.shape[0], dtype=np.int64)
    train_shuffle = identity.copy()
    train_shuffle[train_index] = train_index[
        np.random.default_rng(config.selection_shuffle_seed).permutation(train_index.size)
    ]
    paired_selection = args.selection_dir / "paired"
    shuffled_selection = args.selection_dir / "shuffled"
    ensure_selection(args.data_dir, paired_selection, config, config.selection_seed, identity)
    ensure_selection(args.data_dir, shuffled_selection, config, config.selection_seed, train_shuffle)

    data_by_graph = {
        "paired": V5SData(args.data_dir, paired_selection),
        "shuffled": V5SData(args.data_dir, shuffled_selection),
    }
    data = data_by_graph["paired"]
    common_reference = ((np.asarray(data.rna[data.test]) - data.train_mean) / (data.train_std + 1e-6)).astype(np.float32)
    np.save(args.outdir / "rna_reference_test.npy", common_reference)
    all_rows = []
    for seed in args.seeds:
        set_seed(seed)
        cell_shuffle_seed = seed + 90127
        cell_shuffle = identity.copy()
        cell_shuffle[test_index] = test_index[
            np.random.default_rng(cell_shuffle_seed).permutation(test_index.size)
        ]
        for condition in CONDITIONS:
            graph_name = "shuffled" if condition.startswith("shuffled_graph") else "paired"
            graph_data = data_by_graph[graph_name]
            is_cell_control = condition.startswith("cell_shuffled")
            input_permutation = cell_shuffle if is_cell_control else identity
            condition_dir = args.outdir / f"seed_{seed}" / condition
            condition_dir.mkdir(parents=True, exist_ok=False)
            copy_shared_artifacts(args.selection_dir / graph_name, condition_dir)
            np.save(condition_dir / "teacher_input_row_permutation.npy", input_permutation)
            write_json(condition_dir / "config.json", config.to_dict())
            write_json(
                condition_dir / "manifest.json",
                make_manifest(
                    graph_data,
                    seed,
                    condition,
                    config,
                    graph_name,
                    config.selection_shuffle_seed,
                    cell_shuffle_seed if is_cell_control else None,
                ),
            )
            z, linear_manifest = compute_linear_teacher(
                graph_data.atac,
                np.asarray(graph_data.b_reg),
                graph_data.train,
                np.asarray(graph_data.gene_confidence),
                row_permutation=input_permutation,
            )
            write_json(condition_dir / "linear_manifest.json", linear_manifest)
            save_representations(z, graph_data, condition_dir)
            np.savez_compressed(
                condition_dir / "teacher_best.npz",
                B_reg=np.asarray(graph_data.b_reg),
                gene_confidence=np.asarray(graph_data.gene_confidence),
                atac_train_mean=np.asarray(graph_data.atac_train_mean),
                rna_train_mean=np.asarray(graph_data.train_mean),
                rna_train_std=np.asarray(graph_data.train_std),
                seed=np.asarray(seed),
                condition=np.asarray(condition),
            )
            metrics = evaluate_saved_test_geometry(args.data_dir, z[graph_data.test], common_reference, config.evaluation_seed)
            write_json(condition_dir / "geometry_metrics.json", metrics)
            all_rows.append({"seed": seed, "condition": condition, **metrics})
            print(json.dumps({"seed": seed, "condition": condition, **metrics}), flush=True)
            del z
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary = {"seeds": list(args.seeds), "conditions": list(CONDITIONS), "per_seed": all_rows, "mean_std": {}}
    for condition in CONDITIONS:
        rows = [row for row in all_rows if row["condition"] == condition]
        metrics = [key for key in rows[0] if key not in ("seed", "condition")]
        summary["mean_std"][condition] = {key: {"mean": float(np.mean([row[key] for row in rows])), "std": float(np.std([row[key] for row in rows], ddof=1)) if len(rows) > 1 else 0.0} for key in metrics}
    summary_name = f"summary_{len(args.seeds)}seed"
    write_json(args.outdir / f"{summary_name}.json", summary)
    with (args.outdir / f"{summary_name}.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["seed", "condition"] + [key for key in all_rows[0] if key not in ("seed", "condition")]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)


if __name__ == "__main__":
    main()
