from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_task26_dmode_rna_scaffold_bounded_residual import FOLDS, DATASET_NAMES, _labels, _load_shared_rna, _raw_encode
from v5d.evaluate_teacher import evaluate_geometry
from v5d.utils import write_json
from v5d_student.config import V5DStudentConfig
from v5d_student.data import RNAOnlySource
from v5d_student.geometry_metrics import teacher_alignment_metrics
from v5d_student.pair_bank import PairBank, load_pair_bank, save_pair_bank
from v5d_student.student_encoder import RNAScaffoldBoundedResidualEncoder
from v5d_student.train_student import encode_split, train_shared_student


def _topk_indices(graph: np.ndarray, k: int = 60) -> np.ndarray:
    values = np.asarray(graph, dtype=np.float32).copy()
    np.fill_diagonal(values, -np.inf)
    width = min(k, values.shape[0] - 1)
    return np.argpartition(-values, width - 1, axis=1)[:, :width]


def _build_k60_bank(graph: np.ndarray, teacher_embedding: np.ndarray) -> PairBank:
    partners = _topk_indices(graph, 60)
    rows = np.arange(graph.shape[0])[:, None]
    targets = np.maximum(np.asarray(graph, dtype=np.float32)[rows, partners], 0.0)
    targets /= np.maximum(targets.sum(axis=1, keepdims=True), 1e-12)
    return PairBank(partners, targets.astype(np.float32), int(partners.shape[1]), teacher_embedding.astype(np.float32))


def _graph_paths(task11_dir: Path, task23_dir: Path, fold: str, name: str, split: str) -> Path:
    if split == "train":
        key = next(key for key, value in DATASET_NAMES.items() if value == name)
        return task11_dir / key / "seed_20020130" / "G_T_train.npy"
    return task23_dir / "matched_geometry" / fold / name / "G_SNF_verified.npy"


def _history_summary(path: Path) -> dict[str, object]:
    history = json.loads(path.read_text(encoding="utf-8"))
    loss = np.asarray([row["val_loss_macro"] for row in history], dtype=float)
    topology = np.asarray([row["val_topology_score_macro"] for row in history], dtype=float)
    return {
        "epochs": len(history),
        "best_loss_epoch": int(np.argmin(loss)) + 1,
        "best_loss": float(loss.min()),
        "best_topology_epoch": int(np.argmax(topology)) + 1,
        "best_topology": float(topology.max()),
        "best_loss_topology": float(topology[np.argmin(loss)]),
        "best_topology_loss": float(loss[np.argmax(topology)]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/u202522802045/share/zpb"))
    parser.add_argument("--source-quality-dir", type=Path, default=Path("/home/u202522802045/share/zpb/result_contain/pre_task14_dmode_source_quality_20260926"))
    parser.add_argument("--shared-rna-dir", type=Path, default=Path("/home/u202522802045/share/zpb/result_contain/pre_task14_dmode_shared_rna_audit_20260926"))
    parser.add_argument("--task11-dir", type=Path, default=Path("/home/u202522802045/share/zpb/result_contain/pre_task11_residual_audit_20260925"))
    parser.add_argument("--task23-dir", type=Path, default=Path("/home/u202522802045/share/zpb/result_contain/23_dmode_snf_correction_localization_audit_20260927"))
    parser.add_argument("--task26-dir", type=Path, default=Path("/home/u202522802045/share/zpb/result_contain/26_dmode_rna_scaffold_bounded_residual_20260929"))
    parser.add_argument("--task28-dir", type=Path, default=Path("/home/u202522802045/share/zpb/result_contain/28_dmode_candidate_normalized_teacher_target_20260929"))
    parser.add_argument("--v5-source", type=Path, default=Path("/home/u202522802045/share/zpb/V5/Teacher-S10x20_Student-64D-KL/source"))
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20020130)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--fold", choices=tuple(FOLDS), action="append")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.outdir}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    folds = (args.fold or list(FOLDS))[:1] if args.smoke else (args.fold or list(FOLDS))
    max_epochs = 1 if args.smoke else args.max_epochs
    fold_rows = []
    mechanism_rows = []

    for fold_name in folds:
        target_key, source_keys = FOLDS[fold_name]
        fold_dir = args.source_quality_dir / fold_name
        sources = [RNAOnlySource(fold_dir / "harmonized", DATASET_NAMES[key]) for key in source_keys]
        target = RNAOnlySource(fold_dir / "harmonized", DATASET_NAMES[target_key])
        weights = np.load(fold_dir / "dataset_weights.npz")["omega"].astype(np.float64)
        dataset_weights = {source.name: float(weight) for source, weight in zip(sources, weights)}
        shared, mean, std, backbone = _load_shared_rna(
            args.shared_rna_dir / fold_name / "shared_rna_best.pt",
            sources,
            [*sources, target],
            args.v5_source,
            device,
        )
        train_banks, val_banks = {}, {}
        pair_root = args.outdir / fold_name / "pair_banks"
        pair_root.mkdir(parents=True)
        for source in sources:
            task28_root = args.task28_dir / fold_name / "pair_banks"
            train_teacher_embedding = load_pair_bank(task28_root / f"{source.name}_train.npz").teacher_embedding
            val_teacher_embedding = load_pair_bank(task28_root / f"{source.name}_val.npz").teacher_embedding
            if train_teacher_embedding is None or val_teacher_embedding is None:
                raise RuntimeError(f"Missing Task28 full-Teacher topology embedding for {fold_name}/{source.name}")
            train_graph = np.load(_graph_paths(args.task11_dir, args.task23_dir, fold_name, source.name, "train"), mmap_mode="r")
            val_graph = np.load(_graph_paths(args.task11_dir, args.task23_dir, fold_name, source.name, "val"), mmap_mode="r")
            train_banks[source.name] = _build_k60_bank(train_graph, train_teacher_embedding)
            val_banks[source.name] = _build_k60_bank(val_graph, val_teacher_embedding)
            save_pair_bank(train_banks[source.name], pair_root / f"{source.name}_train.npz")
            save_pair_bank(val_banks[source.name], pair_root / f"{source.name}_val.npz")

        config = V5DStudentConfig(
            latent_dim=64,
            distillation_loss="local_kl",
            freeze_mode="shared_rna_backbone",
            teacher_target="Task29 exact K60 Full-Teacher SNF affinity, candidate-normalized",
            max_epochs=max_epochs,
            checkpoint_metric="teacher_recall_at_10",
            pair_bank_seed=args.seed,
            local_teacher_temperature=0.07,
            local_student_temperature=0.07,
        )
        model = RNAScaffoldBoundedResidualEncoder(backbone, 64, 128, 0.10)
        model, training = train_shared_student(
            sources,
            train_banks,
            val_banks,
            mean,
            std,
            config,
            args.seed,
            device,
            dataset_weights,
            model=model,
        )
        out = args.outdir / fold_name / f"seed_{args.seed}"
        out.mkdir(parents=True)
        history = training["history"]
        z_student_test = _raw_encode(model, target, "test", mean, std, device)
        z_rna_test = shared[target.name][target.test]
        labels_test = _labels(args.root, target, target.test)
        np.save(out / "z_student_test.npy", z_student_test)
        np.save(out / "z_rna_shared_test.npy", z_rna_test)
        best_topology = training.pop("best_topology_state_dict")
        best_loss = training.pop("best_loss_state_dict")
        run_config = {
            **config.to_dict(),
            "task_id": 30,
            "student_parameterization": "R_embed + 0.10*tanh(adapter(R_embed))",
            "rna_scaffold_dim": 64,
            "adapter": "64->128->64",
            "residual_bound": 0.10,
            "residual_final_layer_zero_initialized": True,
            "final_representation": "raw Z_student = R_embed + delta",
            "pair_bank_rule": "exact per-anchor Teacher top-60 neighbors",
            "candidate_count": 60,
            "candidate_target": "selected Full Teacher affinity row renormalized to sum 1",
            "teacher_embedding_reused_from": "Task28 full-Teacher topology cache",
        }
        checkpoint_base = {
            "model_state_dict": best_topology,
            "seed": args.seed,
            "config": run_config,
            "rna_mean": mean,
            "rna_std": std,
            "checkpoint_type": "best_by_topology",
            "manifest": {
                "task_id": 30,
                "source_commit": args.source_commit,
                "target": DATASET_NAMES[target_key],
                "sources": [DATASET_NAMES[key] for key in source_keys],
                "student_input": "RNA only",
                "teacher_target": "exact K60 Full Teacher SNF affinity, candidate-normalized",
                "labels_used_for_training_or_selection": False,
            },
        }
        torch.save(checkpoint_base, out / "best_by_topology.pt")
        torch.save({**checkpoint_base, "model_state_dict": best_loss, "checkpoint_type": "best_by_loss"}, out / "best_by_loss.pt")
        write_json(out / "config.json", run_config)
        write_json(out / "training_history.json", history)
        baseline26 = json.loads((args.task26_dir / fold_name / f"seed_{args.seed}" / "target_metrics.json").read_text(encoding="utf-8"))
        baseline28 = json.loads((args.task28_dir / fold_name / f"seed_{args.seed}" / "target_metrics.json").read_text(encoding="utf-8"))
        metrics = {
            "RNA": baseline26["RNA"],
            "Task26": baseline26["Task26"],
            "Task28": baseline28["Task28"],
            "Task30": evaluate_geometry(z_student_test, z_rna_test, labels_test, args.seed),
        }
        write_json(out / "target_metrics.json", metrics)
        source_validation = []
        for source in sources:
            z_val = encode_split(model, source, "val", mean, std, device)
            labels_val = _labels(args.root, source, source.val)
            topology = teacher_alignment_metrics(z_val, val_banks[source.name].teacher_embedding)
            source_validation.append({
                "dataset": source.name,
                "metrics": evaluate_geometry(z_val, z_val, labels_val, args.seed),
                "topology": topology,
                "candidate_count": int(val_banks[source.name].partner_index.shape[1]),
                "q_finite": bool(np.isfinite(val_banks[source.name].teacher_similarity).all()),
            })
        write_json(out / "source_validation_metrics.json", source_validation)
        comparison = {
            "Task26": _history_summary(args.task26_dir / fold_name / f"seed_{args.seed}" / "training_history.json"),
            "Task28": _history_summary(args.task28_dir / fold_name / f"seed_{args.seed}" / "training_history.json"),
            "Task30": _history_summary(out / "training_history.json"),
        }
        write_json(out / "source_mechanism_audit.json", {"comparison": comparison, "source_validation": source_validation, "checkpoint_rule": "teacher_recall_at_10 validation topology", "labels_used_for_selection": False})
        write_json(out / "manifest.json", {
            "task_id": 30,
            "parent_tasks": [26, 28, 29],
            "experiment": "D-mode structure-preserving K60 pair-bank Student",
            "scientific_change": "pair-bank representation only: exact per-anchor Teacher top-60 instead of 15 local + 15 random",
            "source_commit": args.source_commit,
            "seed": args.seed,
            "target": DATASET_NAMES[target_key],
            "source_datasets": [DATASET_NAMES[key] for key in source_keys],
            "teacher": "full source-specific SNF Teacher; reused cached G_T_train and G_SNF_verified validation geometry",
            "student_input": "RNA only",
            "student_parameterization": "Z = R_embed + delta; delta = 0.10*tanh(adapter(R_embed))",
            "rna_scaffold_frozen": True,
            "adapter": "64->128->64",
            "residual_bound": 0.10,
            "student_loss": "Local-KL",
            "local_teacher_temperature": 0.07,
            "local_student_temperature": 0.07,
            "pair_bank_rule": "exact per-anchor Teacher top-60 neighbors",
            "candidate_count": 60,
            "candidate_target": "candidate-normalized selected Full Teacher affinity",
            "checkpoint_metric": "teacher_recall_at_10",
            "labels_used_for_training": False,
            "labels_used_for_checkpoint_selection": False,
            "target_atac_used": False,
            "run_command": shlex.join([sys.executable, *sys.argv]),
        })
        fold_rows.append({"fold": fold_name, "target": DATASET_NAMES[target_key], "seed": args.seed, "metrics": metrics, "source_validation": source_validation})
        mechanism_rows.append({"fold": fold_name, "comparison": comparison})

    write_json(args.outdir / "TASK30_SOURCE_MECHANISM_AUDIT.json", mechanism_rows)
    write_json(args.outdir / "SUMMARY.json", {
        "task_id": 30,
        "parent_tasks": [26, 28, 29],
        "training_performed": True,
        "teacher_retrained": False,
        "student_trained": True,
        "snf_recomputed": False,
        "B_or_F_recomputed": False,
        "labels_used_for_training_or_selection": False,
        "pair_bank_rule": "exact per-anchor Teacher top-60 neighbors",
        "candidate_count": 60,
        "folds": fold_rows,
        "smoke": args.smoke,
    })


if __name__ == "__main__":
    main()
