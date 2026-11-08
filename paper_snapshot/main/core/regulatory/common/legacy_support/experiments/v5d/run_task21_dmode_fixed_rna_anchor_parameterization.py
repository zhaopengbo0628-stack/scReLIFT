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
from v5d.utils import stable_name_seed, write_json
from v5d_student.config import V5DStudentConfig
from v5d_student.data import RNAOnlySource
from v5d_student.pair_bank import PairBank, save_pair_bank
from v5d_student.student_encoder import RNAAnchoredStudentEncoder
from v5d_student.train_student import encode_split, train_shared_student


DATASET_NAMES = {
    "D1": "PBMC-MO-001",
    "D2": "PBMC-MO-002",
    "D3": "PBMC-MO-003",
    "D4": "PBMC-MO-004",
}
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


def _l2(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def _snf_q(rna: np.ndarray, field: np.ndarray) -> np.ndarray:
    import snf

    views = [
        snf.make_affinity(_l2(values.astype(np.float32)), metric="sqeuclidean", K=20, mu=0.5, normalize=True)
        for values in (rna, field)
    ]
    fused = np.asarray(snf.snf(*views, K=20, t=20, alpha=1.0), dtype=np.float32)
    np.fill_diagonal(fused, 0.0)
    return fused / np.maximum(fused.sum(axis=1, keepdims=True), 1e-12)


def _spectral_embedding(q: np.ndarray, dimensions: int = 128) -> np.ndarray:
    symmetric = (q + q.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    embedding = vectors[:, order] * np.sqrt(values)[None, :]
    embedding = embedding[:, : min(dimensions, embedding.shape[1])]
    if embedding.shape[1] < dimensions:
        embedding = np.pad(embedding, ((0, 0), (0, dimensions - embedding.shape[1])))
    return _l2(embedding.astype(np.float32))


def _build_snf_bank(q: np.ndarray, seed: int, neighbor_k: int, random_k: int) -> PairBank:
    cells = q.shape[0]
    local_k = min(neighbor_k, cells - 1)
    order = np.argsort(-q, axis=1, kind="stable")
    local = np.empty((cells, local_k), dtype=np.int64)
    for anchor in range(cells):
        local[anchor] = order[anchor][order[anchor] != anchor][:local_k]
    rng = np.random.default_rng(seed)
    random_partners = np.empty((cells, random_k), dtype=np.int64)
    universe = np.arange(cells, dtype=np.int64)
    for anchor in range(cells):
        excluded = np.concatenate(([anchor], local[anchor]))
        allowed = np.setdiff1d(universe, excluded, assume_unique=False)
        if allowed.size == 0:
            allowed = np.setdiff1d(universe, [anchor])
        random_partners[anchor] = rng.choice(allowed, random_k, replace=allowed.size < random_k)
    partners = np.concatenate((local, random_partners), axis=1)
    targets = q[np.arange(cells)[:, None], partners].astype(np.float32)
    return PairBank(partners, targets, local_k, _spectral_embedding(q))


def _load_final_labels(root: Path, source: RNAOnlySource, rows: np.ndarray) -> np.ndarray:
    labels = load_split_labels(source.root, rows)
    if labels is not None:
        return labels
    dataset_key = next(key for key, value in DATASET_NAMES.items() if value == source.name)
    label_path = root / LABEL_PATHS[dataset_key]
    with label_path.open("r", encoding="utf-8", newline="") as handle:
        labels = np.asarray([row["cell_type"] for row in csv.DictReader(handle)], dtype=str)
    if labels.size != source.rna.shape[0]:
        raise RuntimeError(f"{source.name}: fallback label count does not match harmonized cells")
    return labels[rows]
def _load_shared_rna(
    checkpoint: Path,
    fit_sources: list[RNAOnlySource],
    encode_sources: list[RNAOnlySource],
    v5_source: Path,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, torch.nn.Module]:
    if str(v5_source) not in sys.path:
        sys.path.insert(0, str(v5_source))
    from v5s.denoise import DenoisingAutoencoder

    means = [np.asarray(source.rna[source.train], dtype=np.float32).mean(axis=0) for source in fit_sources]
    variances = [np.asarray(source.rna[source.train], dtype=np.float32).var(axis=0) for source in fit_sources]
    mean = np.stack(means).mean(axis=0).astype(np.float32)
    variance = np.mean(np.stack(variances) + (np.stack(means) - mean) ** 2, axis=0)
    std = np.sqrt(variance).astype(np.float32)
    values = torch.load(checkpoint, map_location="cpu", weights_only=False)
    feature_count = int(values["feature_count"])
    state = values["model_state_dict"]
    model = DenoisingAutoencoder(feature_count, 512, 64)
    model.load_state_dict(state)
    model.to(device).eval()
    outputs = {}
    with torch.no_grad():
        for source in encode_sources:
            values = ((np.asarray(source.rna, dtype=np.float32) - mean) / (std + 1e-6)).astype(np.float32)
            pieces = []
            for start in range(0, values.shape[0], 512):
                pieces.append(model.encoder(torch.from_numpy(values[start:start + 512]).to(device)).cpu().numpy())
            outputs[source.name] = np.concatenate(pieces).astype(np.float32)
    return outputs, mean, std, model.encoder


def _mean_std(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = [key for key in rows[0] if key != "seed"]
    return {key: {"mean": float(np.mean([row[key] for row in rows])), "std": float(np.std([row[key] for row in rows], ddof=1)) if len(rows) > 1 else 0.0} for key in keys}


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task21 D-mode fixed RNA anchor residual parameterization")
    parser.add_argument("--root", type=Path, default=Path("/home/u202522802045/share/zpb"))
    parser.add_argument("--source-quality-dir", type=Path, default=Path("/home/u202522802045/share/zpb/result_contain/pre_task14_dmode_source_quality_20260926"))
    parser.add_argument("--shared-rna-dir", type=Path, default=Path("/home/u202522802045/share/zpb/result_contain/pre_task14_dmode_shared_rna_audit_20260926"))
    parser.add_argument("--v5-source", type=Path, default=Path("/home/u202522802045/share/zpb/V5/Teacher-S10x20_Student-64D-KL/source"))
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20020130)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--fold", choices=tuple(FOLDS), action="append")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    folds = args.fold or list(FOLDS)
    if args.smoke:
        folds = folds[:1]
        args.max_epochs = 1
    args.outdir.mkdir(parents=True, exist_ok=False)
    rows = []
    for fold_name in folds:
        target_key, source_keys = FOLDS[fold_name]
        fold_source_dir = args.source_quality_dir / fold_name
        harmonized = fold_source_dir / "harmonized"
        source_names = [DATASET_NAMES[key] for key in source_keys]
        target_name = DATASET_NAMES[target_key]
        sources = [RNAOnlySource(harmonized, name) for name in source_names]
        target = RNAOnlySource(harmonized, target_name)
        with np.load(fold_source_dir / "dataset_weights.npz") as values:
            omega = values["omega"].astype(np.float64)
        dataset_weights = {name: float(weight) for name, weight in zip(source_names, omega)}
        shared_dir = args.shared_rna_dir / fold_name
        shared_checkpoint = shared_dir / "shared_rna_best.pt"
        if not shared_checkpoint.exists():
            raise FileNotFoundError(shared_checkpoint)
        shared, mean, std, backbone = _load_shared_rna(
            shared_checkpoint, sources, [*sources, target], args.v5_source, device
        )
        train_banks, val_banks = {}, {}
        pair_root = args.outdir / fold_name / "pair_banks"
        pair_root.mkdir(parents=True)
        for source in sources:
            f_train = np.load(fold_source_dir / f"{source.name}_F_train.npy")
            f_val = np.load(fold_source_dir / f"{source.name}_F_val.npy")
            train_q = _snf_q(shared[source.name][source.train], f_train)
            val_q = _snf_q(shared[source.name][source.val], f_val)
            train_banks[source.name] = _build_snf_bank(train_q, stable_name_seed(args.seed, source.name), 15, 15)
            val_banks[source.name] = _build_snf_bank(val_q, stable_name_seed(args.seed + 10000, source.name), 15, 15)
            save_pair_bank(train_banks[source.name], pair_root / f"{source.name}_train.npz")
            save_pair_bank(val_banks[source.name], pair_root / f"{source.name}_val.npz")
        config = V5DStudentConfig(
            latent_dim=64,
            distillation_loss="local_kl",
            freeze_mode="shared_rna_backbone",
            teacher_target="SNF(RNA_shared_DAE,F_D)",
            max_epochs=args.max_epochs,
            checkpoint_metric="teacher_recall_at_10",
            pair_bank_seed=args.seed,
        )
        student = RNAAnchoredStudentEncoder(
            backbone,
            latent_dim=64,
            adapter_hidden_dim=128,
            max_correction=0.25,
            fixed_correction=0.10,
        )
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
            model=student,
        )
        fold_out = args.outdir / fold_name
        seed_out = fold_out / f"seed_{args.seed}"
        seed_out.mkdir()
        target_student = encode_split(model, target, "test", mean, std, device)
        target_rna = shared[target.name][target.test]
        labels = _load_final_labels(args.root, target, target.test)

        student_metrics = evaluate_geometry(target_student, target_rna, labels, args.seed)
        rna_metrics = evaluate_geometry(target_rna, target_rna, labels, args.seed)
        np.save(seed_out / "z_student_test.npy", target_student)
        np.save(seed_out / "z_rna_shared_test.npy", target_rna)
        best_topology = training.pop("best_topology_state_dict")
        best_loss = training.pop("best_loss_state_dict")
        correction_scale = float(model.correction_scale().detach().cpu())
        run_config = {
            **config.to_dict(),
            "student_parameterization": "normalize(z_RNA + alpha*normalize(adapter(z_RNA)))",
            "rna_backbone_dim": 64,
            "adapter_hidden_dim": 128,
            "max_correction": 0.25,
            "correction_mode": "fixed",
            "fixed_correction": 0.10,
            "final_correction_scale": correction_scale,
        }
        checkpoint_common = {"model_state_dict": best_topology, "seed": args.seed, "config": run_config, "rna_mean": mean, "rna_std": std, "checkpoint_type": "best_by_topology", "manifest": {"task_id": 21, "target": target_name, "sources": source_names, "dataset_weights": dataset_weights, "teacher_target": "SNF(RNA_shared_DAE,F_D)", "student_input": "RNA only", "rna_backbone_frozen": True, "correction_mode": "fixed", "fixed_correction": 0.10, "target_atac_used": False, "target_labels_used_for_training_or_selection": False}}
        torch.save(checkpoint_common, seed_out / "best_by_topology.pt")
        torch.save({**checkpoint_common, "model_state_dict": best_loss, "checkpoint_type": "best_by_loss"}, seed_out / "best_by_loss.pt")
        write_json(seed_out / "config.json", run_config)
        write_json(seed_out / "training_history.json", training["history"])
        manifest = {"task_id": 21, "parent_task": 20, "experiment": "D-mode frozen RNA anchor + fixed residual correction", "primary_bottleneck": "STUDENT_GLOBAL_GEOMETRY_COMPRESSION", "diagnostic_evidence": "Task19 recovered ARI versus Task17 but its learnable residual gate collapsed near 0.063; Task20 fixed alpha=0.25 overshot and degraded local/NMI metrics", "single_changed_variable": "Student residual gate: fixed alpha=0.25 -> fixed alpha=0.10", "unchanged_components": ["cross-dataset reliability screening", "B_D", "F_D", "source-specific SNF Teachers", "n_d*Q_d source weighting", "Local-KL", "candidate bank", "checkpoint rule", "held-out RNA-only inference"], "expected_mechanism": "retain a smaller nonzero cross-modal correction while the frozen RNA anchor preserves global partition geometry", "seed": args.seed, "target": target_name, "source_datasets": source_names, "dataset_weight_formula": "omega_d = n_d * Q_d / sum_k(n_k * Q_k)", "dataset_weights": dataset_weights, "teacher": "SNF(RNA_shared_DAE,F_D)", "snf_config": {"metric": "sqeuclidean", "K": 20, "mu": 0.5, "normalize": True, "t": 20, "alpha": 1.0}, "student_input": "RNA only", "student_parameterization": "normalize(z_RNA + 0.10*normalize(adapter(z_RNA)))", "rna_backbone_dim": 64, "rna_backbone_frozen": True, "adapter": "64->128->64", "correction_mode": "fixed", "fixed_correction": 0.10, "final_correction_scale": correction_scale, "student_loss": "Local-KL", "local_k": 15, "random_k": 15, "checkpoint_metric": "teacher_recall_at_10", "labels_used_for_training": False, "labels_used_for_checkpoint_selection": False, "target_atac_used": False, "target_labels_used_only_for_final_test": True, "source_quality_artifacts": str(fold_source_dir), "shared_rna_checkpoint": str(shared_checkpoint), "run_command": shlex.join([sys.executable, *sys.argv])}
        write_json(seed_out / "manifest.json", manifest)
        write_json(seed_out / "target_metrics.json", {"RNA_shared": rna_metrics, "Student": student_metrics})
        rows.append({"fold": fold_name, "target": target_name, "seed": args.seed, "rna_ari": rna_metrics["ari"], "rna_nmi": rna_metrics["nmi"], "student_ari": student_metrics["ari"], "student_nmi": student_metrics["nmi"], "student_knn_purity_at_15": student_metrics["knn_purity_at_15"]})
    write_json(args.outdir / "SUMMARY.json", {"task_id": 21, "parent_task": 20, "single_changed_variable": "Student residual gate: fixed alpha=0.25 -> fixed alpha=0.10", "training_performed": True, "teacher_retrained": False, "student_trained": True, "folds": rows, "mean_std": _mean_std([{key: value for key, value in row.items() if key not in {"fold", "target"}} for row in rows]), "target_atac_used": False, "target_labels_used_for_training_or_selection": False, "smoke": args.smoke})
    with (args.outdir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

