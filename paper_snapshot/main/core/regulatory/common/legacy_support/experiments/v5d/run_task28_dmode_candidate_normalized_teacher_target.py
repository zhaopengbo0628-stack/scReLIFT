from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v5d.evaluate_teacher import evaluate_geometry, load_split_labels
from v5d.utils import write_json
from v5d_student.config import V5DStudentConfig
from v5d_student.data import RNAOnlySource
from v5d_student.pair_bank import PairBank, load_pair_bank, save_pair_bank
from v5d_student.student_encoder import RNAScaffoldBoundedResidualEncoder
from v5d_student.train_student import encode_split, train_shared_student

DATASET_NAMES = {"D1": "PBMC-MO-001", "D2": "PBMC-MO-002", "D3": "PBMC-MO-003", "D4": "PBMC-MO-004"}
FOLDS = {
    "target_D1": ("D1", ("D2", "D3", "D4")),
    "target_D2": ("D2", ("D1", "D3", "D4")),
    "target_D3": ("D3", ("D1", "D2", "D4")),
    "target_D4": ("D4", ("D1", "D2", "D3")),
}
LABEL_PATHS = {
    "D1": "DATA/PBMC-MO-001__10X__GranSorted__10k__Multiome/processed/label.csv",
    "D2": "DATA/PBMC-MO-002__10X__GranSorted__10k__Multiome/processed/label.csv",
    "D3": "DATA/PBMC-MO-003__10X__Unsorted__10k__Multiome/processed/label.csv",
    "D4": "DATA/PBMC-MO-004__10X__Unsorted__3k__Multiome/processed/label.csv",
}


def _candidate_normalized_bank(bank: PairBank) -> PairBank:
    targets = bank.teacher_similarity.astype(np.float32).copy()
    targets /= np.maximum(targets.sum(axis=1, keepdims=True), 1e-12)
    return PairBank(bank.partner_index.copy(), targets, bank.local_k, bank.teacher_embedding)


def _load_shared_rna(checkpoint: Path, fit_sources: list[RNAOnlySource], encode_sources: list[RNAOnlySource], v5_source: Path, device: torch.device):
    sys.path.insert(0, str(v5_source))
    from v5s.denoise import DenoisingAutoencoder

    means = [np.asarray(s.rna[s.train], dtype=np.float32).mean(axis=0) for s in fit_sources]
    variances = [np.asarray(s.rna[s.train], dtype=np.float32).var(axis=0) for s in fit_sources]
    mean = np.stack(means).mean(axis=0).astype(np.float32)
    std = np.sqrt(np.mean(np.stack(variances) + (np.stack(means) - mean) ** 2, axis=0)).astype(np.float32)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = DenoisingAutoencoder(int(saved["feature_count"]), 512, 64)
    model.load_state_dict(saved["model_state_dict"])
    model.to(device).eval()
    outputs = {}
    with torch.no_grad():
        for source in encode_sources:
            values = ((np.asarray(source.rna, dtype=np.float32) - mean) / (std + 1e-6)).astype(np.float32)
            outputs[source.name] = np.concatenate([model.encoder(torch.from_numpy(values[i:i + 512]).to(device)).cpu().numpy() for i in range(0, len(values), 512)]).astype(np.float32)
    return outputs, mean, std, model.encoder


def _labels(root: Path, source: RNAOnlySource, rows: np.ndarray) -> np.ndarray:
    found = load_split_labels(source.root, rows)
    if found is not None:
        return found
    key = next(k for k, v in DATASET_NAMES.items() if v == source.name)
    with (root / LABEL_PATHS[key]).open("r", encoding="utf-8", newline="") as handle:
        values = np.asarray([row["cell_type"] for row in csv.DictReader(handle)], dtype=str)
    return values[rows]


def _raw_encode(model, source, split, mean, std, device):
    rows = source.rows(split)
    values = ((np.asarray(source.rna[rows], dtype=np.float32) - mean) / (std + 1e-6)).astype(np.float32)
    with torch.no_grad():
        return np.concatenate([model.forward_raw(torch.from_numpy(values[i:i + 512]).to(device)).cpu().numpy() for i in range(0, len(values), 512)]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/u202522802045/share/zpb"))
    parser.add_argument("--source-quality-dir", type=Path, default=Path("/home/u202522802045/share/zpb/result_contain/pre_task14_dmode_source_quality_20260926"))
    parser.add_argument("--shared-rna-dir", type=Path, default=Path("/home/u202522802802045/share/zpb/result_contain/pre_task14_dmode_shared_rna_audit_20260926"))
    parser.add_argument("--task26-dir", type=Path, default=Path("/home/u202522802045/share/zpb/result_contain/26_dmode_rna_scaffold_bounded_residual_20260929"))
    parser.add_argument("--v5-source", type=Path, default=Path("/home/u202522802045/share/zpb/V5/Teacher-S10x20_Student-64D-KL/source"))
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20020130)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--fold", choices=tuple(FOLDS), action="append")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if "802802045" in str(args.shared_rna_dir):
        args.shared_rna_dir = Path("/home/u202522802045/share/zpb/result_contain/pre_task14_dmode_shared_rna_audit_20260926")
    device = torch.device(args.device)
    folds = (args.fold or list(FOLDS))[:1] if args.smoke else (args.fold or list(FOLDS))
    if args.smoke:
        args.max_epochs = 1
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.outdir}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold_name in folds:
        target_key, source_keys = FOLDS[fold_name]
        fold_dir = args.source_quality_dir / fold_name
        sources = [RNAOnlySource(fold_dir / "harmonized", DATASET_NAMES[k]) for k in source_keys]
        target = RNAOnlySource(fold_dir / "harmonized", DATASET_NAMES[target_key])
        weights = np.load(fold_dir / "dataset_weights.npz")["omega"].astype(np.float64)
        dataset_weights = {s.name: float(w) for s, w in zip(sources, weights)}
        shared, mean, std, backbone = _load_shared_rna(args.shared_rna_dir / fold_name / "shared_rna_best.pt", sources, [*sources, target], args.v5_source, device)
        train_banks, val_banks = {}, {}
        pair_root = args.outdir / fold_name / "pair_banks"
        pair_root.mkdir(parents=True)
        for source in sources:
            task26_pair_root = args.task26_dir / fold_name / "pair_banks"
            train_banks[source.name] = _candidate_normalized_bank(load_pair_bank(task26_pair_root / f"{source.name}_train.npz"))
            val_banks[source.name] = _candidate_normalized_bank(load_pair_bank(task26_pair_root / f"{source.name}_val.npz"))
            save_pair_bank(train_banks[source.name], pair_root / f"{source.name}_train.npz")
            save_pair_bank(val_banks[source.name], pair_root / f"{source.name}_val.npz")
        config = V5DStudentConfig(latent_dim=64, distillation_loss="local_kl", freeze_mode="shared_rna_backbone", teacher_target="candidate-normalized SNF(RNA_shared_DAE,F_D)", max_epochs=args.max_epochs, checkpoint_metric="teacher_recall_at_10", pair_bank_seed=args.seed)
        model = RNAScaffoldBoundedResidualEncoder(backbone, 64, 128, 0.10)
        model, training = train_shared_student(sources, train_banks, val_banks, mean, std, config, args.seed, device, dataset_weights, model=model)
        out = args.outdir / fold_name / f"seed_{args.seed}"
        out.mkdir()
        z_student = _raw_encode(model, target, "test", mean, std, device)
        z_rna = shared[target.name][target.test]
        labels = _labels(args.root, target, target.test)
        np.save(out / "z_student_test.npy", z_student)
        np.save(out / "z_rna_shared_test.npy", z_rna)
        best_topology = training.pop("best_topology_state_dict")
        best_loss = training.pop("best_loss_state_dict")
        run_config = {**config.to_dict(), "student_parameterization": "R_embed + 0.10*tanh(adapter(R_embed))", "rna_scaffold_dim": 64, "adapter": "64->128->64", "residual_bound": 0.10, "residual_final_layer_zero_initialized": True, "final_representation": "raw Z_student = R_embed + delta"}
        checkpoint = {"model_state_dict": best_topology, "seed": args.seed, "config": run_config, "rna_mean": mean, "rna_std": std, "checkpoint_type": "best_by_topology", "manifest": {"task_id": 28, "source_commit": args.source_commit, "target": DATASET_NAMES[target_key], "sources": [DATASET_NAMES[k] for k in source_keys], "student_input": "RNA only", "teacher_target": "candidate-normalized SNF(RNA_shared_DAE,F_D)", "labels_used_for_training_or_selection": False}}
        torch.save(checkpoint, out / "best_by_topology.pt")
        torch.save({**checkpoint, "model_state_dict": best_loss, "checkpoint_type": "best_by_loss"}, out / "best_by_loss.pt")
        write_json(out / "config.json", run_config)
        write_json(out / "training_history.json", training["history"])
        metrics = {"RNA": evaluate_geometry(z_rna, z_rna, labels, args.seed), "Task28": evaluate_geometry(z_student, z_rna, labels, args.seed)}
        write_json(out / "target_metrics.json", metrics)
        write_json(out / "manifest.json", {"task_id": 28, "parent_task": 26, "experiment": "D-mode candidate-normalized Teacher target", "scientific_change": "Normalize each saved 30-candidate SNF score vector to sum 1 before the unchanged Local-KL softmax", "source_commit": args.source_commit, "seed": args.seed, "target": DATASET_NAMES[target_key], "source_datasets": [DATASET_NAMES[k] for k in source_keys], "teacher": "candidate-normalized SNF(RNA_shared_DAE,F_D)", "student_input": "RNA only", "student_parameterization": "Z = R_embed + delta; delta = 0.10*tanh(adapter(R_embed))", "rna_scaffold_frozen": True, "adapter": "64->128->64", "residual_bound": 0.10, "residual_final_layer_zero_initialized": True, "student_loss": "Local-KL", "local_k": 15, "random_k": 15, "checkpoint_metric": "teacher_recall_at_10", "labels_used_for_training": False, "labels_used_for_checkpoint_selection": False, "target_atac_used": False, "task26_pair_bank_dir": str(args.task26_dir / fold_name / "pair_banks"), "shared_rna_checkpoint": str(args.shared_rna_dir / fold_name / "shared_rna_best.pt"), "run_command": shlex.join([sys.executable, *sys.argv])})
        rows.append({"fold": fold_name, "target": DATASET_NAMES[target_key], "seed": args.seed, "rna_ari": metrics["RNA"]["ari"], "rna_nmi": metrics["RNA"]["nmi"], "student_ari": metrics["Task28"]["ari"], "student_nmi": metrics["Task28"]["nmi"], "student_knn_purity_at_15": metrics["Task28"]["knn_purity_at_15"]})
    write_json(args.outdir / "SUMMARY.json", {"task_id": 28, "parent_task": 26, "scientific_change": "candidate-local sum normalization before unchanged Local-KL softmax", "source_commit": args.source_commit, "training_performed": True, "teacher_retrained": False, "student_trained": True, "snf_recomputed": False, "folds": rows, "smoke": args.smoke})


if __name__ == "__main__":
    main()
