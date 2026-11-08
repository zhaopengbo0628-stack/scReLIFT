from __future__ import annotations

import argparse
import copy
import json
import math
import shlex
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_task26_dmode_rna_scaffold_bounded_residual import (
    DATASET_NAMES,
    FOLDS,
    _labels,
    _load_shared_rna,
    _raw_encode,
)
from v5d.evaluate_teacher import evaluate_geometry
from v5d.utils import write_json
from v5d_student.config import V5DStudentConfig
from v5d_student.data import RNAOnlySource, normalized_rows
from v5d_student.geometry_metrics import teacher_alignment_metrics
from v5d_student.pair_bank import PairBank, load_pair_bank
from v5d_student.student_encoder import RNAScaffoldVectorNormResidualEncoder


@dataclass(frozen=True)
class GlobalTarget:
    landmarks: np.ndarray
    values: np.ndarray
    mean: float
    std: float
    spectral: dict[str, float]


def _symmetrize(graph: np.ndarray) -> np.ndarray:
    values = (np.asarray(graph, dtype=np.float64) + np.asarray(graph, dtype=np.float64).T) / 2.0
    values = np.maximum(values, 0.0)
    np.fill_diagonal(values, 0.0)
    return values


def _spectral_coordinates(graph: np.ndarray, dimensions: int = 64) -> tuple[np.ndarray, np.ndarray]:
    values = _symmetrize(graph)
    degree = values.sum(axis=1)
    inv_degree = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    laplacian = np.eye(values.shape[0]) - inv_degree[:, None] * values * inv_degree[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    width = min(dimensions, max(1, values.shape[0] - 1))
    return eigenvalues, eigenvectors[:, 1 : 1 + width]


def _farthest_landmarks(coordinates: np.ndarray, count: int = 64) -> np.ndarray:
    width = min(count, coordinates.shape[0])
    selected = np.empty(width, dtype=np.int64)
    selected[0] = 0
    nearest = np.sum((coordinates - coordinates[0]) ** 2, axis=1)
    nearest[0] = -1.0
    for index in range(1, width):
        best = float(nearest.max())
        candidates = np.flatnonzero(np.isclose(nearest, best, rtol=1e-12, atol=1e-12))
        landmark = int(candidates[0])
        selected[index] = landmark
        distance = np.sum((coordinates - coordinates[landmark]) ** 2, axis=1)
        nearest = np.minimum(nearest, distance)
        nearest[selected[: index + 1]] = -1.0
    return selected


def _landmark_distances(coordinates: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    left = np.sum(coordinates * coordinates, axis=1, keepdims=True)
    right = np.sum(coordinates[landmarks] * coordinates[landmarks], axis=1)[None, :]
    return np.sqrt(np.maximum(left + right - 2.0 * coordinates @ coordinates[landmarks].T, 0.0)).astype(np.float32)


def _build_global_target(graph: np.ndarray, mean: float | None = None, std: float | None = None) -> tuple[GlobalTarget, float, float]:
    eigenvalues, coordinates = _spectral_coordinates(graph)
    landmarks = _farthest_landmarks(coordinates, 64)
    distances = _landmark_distances(coordinates, landmarks)
    fit_mean = float(distances.mean()) if mean is None else float(mean)
    fit_std = float(distances.std()) if std is None else float(std)
    fit_std = max(fit_std, 1e-6)
    nontrivial = eigenvalues[1:]
    low = eigenvalues[1 : 1 + coordinates.shape[1]]
    target = GlobalTarget(
        landmarks=landmarks,
        values=((distances - fit_mean) / fit_std).astype(np.float32),
        mean=fit_mean,
        std=fit_std,
        spectral={
            "nontrivial_rank": int(np.sum(nontrivial > 1e-8)),
            "near_zero_eigenvalues": int(np.sum(eigenvalues <= 1e-8)),
            "lambda2": float(eigenvalues[1]),
            "lambda65": float(eigenvalues[min(65, len(eigenvalues) - 1)]),
            "low64_lambda_sum_fraction": float(low.sum() / max(nontrivial.sum(), 1e-12)),
            "relation_std_before_normalization": float(distances.std()),
            "relation_effective_rank": float(_effective_rank(distances)),
        },
    )
    return target, fit_mean, fit_std


def _effective_rank(values: np.ndarray) -> float:
    singular_values = np.linalg.svd(values - values.mean(axis=0, keepdims=True), compute_uv=False)
    weights = singular_values * singular_values
    weights /= max(float(weights.sum()), 1e-30)
    weights = weights[weights > 0]
    return float(np.exp(-np.sum(weights * np.log(weights))))


def _project_topology_embeddings(train_bank: PairBank, val_bank: PairBank, target_dim: int = 64) -> tuple[PairBank, PairBank]:
    if train_bank.teacher_embedding is None or val_bank.teacher_embedding is None:
        raise ValueError("Task32 requires cached Teacher topology embeddings")
    train_embedding = np.asarray(train_bank.teacher_embedding, dtype=np.float32)
    val_embedding = np.asarray(val_bank.teacher_embedding, dtype=np.float32)
    if train_embedding.shape[1] > target_dim:
        reducer = PCA(n_components=target_dim, svd_solver="randomized", random_state=0)
        train_embedding = reducer.fit_transform(train_embedding)
        val_embedding = reducer.transform(val_embedding)
    if train_embedding.shape[1] < target_dim:
        width = target_dim - train_embedding.shape[1]
        train_embedding = np.pad(train_embedding, ((0, 0), (0, width)))
        val_embedding = np.pad(val_embedding, ((0, 0), (0, width)))
    train_embedding /= np.maximum(np.linalg.norm(train_embedding, axis=1, keepdims=True), 1e-8)
    val_embedding /= np.maximum(np.linalg.norm(val_embedding, axis=1, keepdims=True), 1e-8)
    return replace(train_bank, teacher_embedding=train_embedding.astype(np.float32)), replace(
        val_bank, teacher_embedding=val_embedding.astype(np.float32)
    )


def _bank_batch(bank: PairBank, anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = np.arange(bank.partner_index.shape[1], dtype=np.int64)
    anchor_matrix = np.broadcast_to(anchors[:, None], (anchors.size, columns.size))
    return (
        anchor_matrix.ravel(),
        bank.partner_index[anchor_matrix, columns].ravel(),
        bank.teacher_similarity[anchor_matrix, columns].ravel(),
    )


def _encode_rows(
    model: torch.nn.Module,
    source: RNAOnlySource,
    rows: np.ndarray,
    split: str,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    values = normalized_rows(source, source.rows(split)[rows], mean, std)
    return model.forward_raw(torch.from_numpy(values).to(device))


def _batch_losses(
    model: torch.nn.Module,
    source: RNAOnlySource,
    bank: PairBank,
    target: GlobalTarget,
    anchors: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    config: V5DStudentConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    left, right, local_target = _bank_batch(bank, anchors)
    all_rows = np.concatenate((left, right, target.landmarks))
    unique_rows, inverse = np.unique(all_rows, return_inverse=True)
    z = F.normalize(_encode_rows(model, source, unique_rows, "train", mean, std, device), dim=1)
    anchor_z = z[torch.from_numpy(inverse[: left.size]).to(device)]
    partner_z = z[torch.from_numpy(inverse[left.size : left.size + right.size]).to(device)]
    landmark_z = z[torch.from_numpy(inverse[-target.landmarks.size :]).to(device)]
    local_prediction = (anchor_z * partner_z).sum(dim=1)
    local_prediction = local_prediction.reshape(anchors.size, bank.partner_index.shape[1])
    local_target_tensor = torch.from_numpy(local_target.astype(np.float32)).to(device).reshape_as(local_prediction)
    teacher_probability = F.softmax(local_target_tensor / config.local_teacher_temperature, dim=1)
    student_log_probability = F.log_softmax(local_prediction / config.local_student_temperature, dim=1)
    local_loss = F.kl_div(student_log_probability, teacher_probability, reduction="batchmean")
    student_distances = torch.cdist(anchor_z, landmark_z)
    global_target = torch.from_numpy(target.values[left]).to(device)
    student_distances = (student_distances - target.mean) / target.std
    global_loss = F.mse_loss(student_distances, global_target)
    return local_loss, global_loss


@torch.no_grad()
def _validation_losses(model: torch.nn.Module, source: RNAOnlySource, bank: PairBank, target: GlobalTarget, mean: np.ndarray, std: np.ndarray, config: V5DStudentConfig, device: torch.device) -> tuple[float, float]:
    model.eval()
    rows = source.val
    values = normalized_rows(source, rows, mean, std)
    z = F.normalize(model.forward_raw(torch.from_numpy(values).to(device)), dim=1)
    anchors = torch.arange(bank.anchors, device=device)[:, None].expand_as(torch.from_numpy(bank.partner_index))
    partners = torch.from_numpy(bank.partner_index).to(device)
    teacher_similarity = torch.from_numpy(bank.teacher_similarity.astype(np.float32)).to(device)
    prediction = (z[anchors] * z[partners]).sum(dim=2)
    teacher_probability = F.softmax(teacher_similarity / config.local_teacher_temperature, dim=1)
    student_log_probability = F.log_softmax(prediction / config.local_student_temperature, dim=1)
    local_loss = float(F.kl_div(student_log_probability, teacher_probability, reduction="batchmean").cpu())
    landmark_z = z[torch.from_numpy(target.landmarks).to(device)]
    distances = (torch.cdist(z, landmark_z) - target.mean) / target.std
    global_loss = float(F.mse_loss(distances, torch.from_numpy(target.values).to(device)).cpu())
    return local_loss, global_loss


def _global_metric(z: np.ndarray, target: GlobalTarget) -> dict[str, float]:
    values = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
    distances = np.linalg.norm(values[:, None, :] - values[target.landmarks][None, :, :], axis=2)
    predicted = (distances - target.mean) / target.std
    correlation = np.corrcoef(predicted.ravel(), target.values.ravel())[0, 1]
    return {
        "normalized_distance_mse": float(np.mean((predicted - target.values) ** 2)),
        "normalized_distance_pearson": float(correlation) if np.isfinite(correlation) else float("nan"),
        "cells": int(z.shape[0]),
        "landmarks": int(target.landmarks.size),
    }


def _topology_score(embedding: np.ndarray, bank: PairBank) -> dict[str, float]:
    if bank.teacher_embedding is None:
        raise ValueError("Missing projected Teacher topology embedding")
    return teacher_alignment_metrics(embedding, bank.teacher_embedding)


def _train(
    sources: list[RNAOnlySource],
    train_banks: dict[str, PairBank],
    val_banks: dict[str, PairBank],
    train_targets: dict[str, GlobalTarget],
    val_targets: dict[str, GlobalTarget],
    mean: np.ndarray,
    std: np.ndarray,
    config: V5DStudentConfig,
    seed: int,
    device: torch.device,
    dataset_weights: dict[str, float],
    backbone: torch.nn.Module,
) -> tuple[torch.nn.Module, dict[str, object]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RNAScaffoldVectorNormResidualEncoder(backbone, 64, 128, 0.10).to(device)
    rng = np.random.default_rng(seed)
    initial_local = []
    initial_global = []
    with torch.no_grad():
        for source in sources:
            anchors = np.arange(min(config.batch_anchor, train_banks[source.name].anchors), dtype=np.int64)
            local, global_loss = _batch_losses(model, source, train_banks[source.name], train_targets[source.name], anchors, mean, std, config, device)
            initial_local.append(float(local.cpu()))
            initial_global.append(float(global_loss.cpu()))
    local_scale = max(float(np.average(initial_local, weights=[dataset_weights[s.name] for s in sources])), 1e-8)
    global_scale = max(float(np.average(initial_global, weights=[dataset_weights[s.name] for s in sources])), 1e-8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    steps = max(math.ceil(source.train.size / config.batch_anchor) for source in sources)
    best_topology_state = None
    best_topology = -float("inf")
    best_topology_epoch = 0
    best_total_state = None
    best_total = float("inf")
    history = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_local = []
        train_global = []
        for _ in range(steps):
            losses = []
            for source in sources:
                bank = train_banks[source.name]
                anchors = rng.integers(0, bank.anchors, size=min(config.batch_anchor, bank.anchors))
                local, global_loss = _batch_losses(model, source, bank, train_targets[source.name], anchors, mean, std, config, device)
                losses.append((local / local_scale + global_loss / global_scale) * dataset_weights[source.name])
                train_local.append(float(local.detach().cpu()))
                train_global.append(float(global_loss.detach().cpu()))
            loss = torch.stack(losses).sum() / sum(dataset_weights.values())
            if not torch.isfinite(loss):
                raise FloatingPointError("Task32 total loss is not finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        val_embeddings = {}
        val_local = []
        val_global = []
        topology = []
        topology_weights = []
        for source in sources:
            z_val = _encode_rows(model, source, np.arange(source.val.size, dtype=np.int64), "val", mean, std, device).detach().cpu().numpy()
            val_embeddings[source.name] = z_val
            local_loss, global_loss = _validation_losses(model, source, val_banks[source.name], val_targets[source.name], mean, std, config, device)
            val_local.append(local_loss)
            val_global.append(global_loss)
            topology.append(_topology_score(z_val, val_banks[source.name]))
            topology_weights.append(dataset_weights[source.name])
        val_local_mean = float(np.average(val_local, weights=topology_weights))
        val_global_mean = float(np.average(val_global, weights=topology_weights))
        topology_score = float(np.average([x["teacher_recall_at_10"] for x in topology], weights=topology_weights))
        total_score = val_local_mean / local_scale + val_global_mean / global_scale
        record = {
            "epoch": epoch,
            "train_local_kl": float(np.mean(train_local)),
            "train_global_landmark_mse": float(np.mean(train_global)),
            "val_local_kl": val_local_mean,
            "val_global_landmark_mse": val_global_mean,
            "val_total_normalized": total_score,
            "val_topology_score_teacher_recall_at_10": topology_score,
            "val_topology_by_dataset": {source.name: metrics for source, metrics in zip(sources, topology)},
        }
        history.append(record)
        print(record, flush=True)
        if total_score < best_total:
            best_total = total_score
            best_total_state = copy.deepcopy(model.state_dict())
        if topology_score > best_topology:
            best_topology = topology_score
            best_topology_epoch = epoch
            best_topology_state = copy.deepcopy(model.state_dict())
    if best_topology_state is None or best_total_state is None:
        raise RuntimeError("Task32 did not select checkpoints")
    model.load_state_dict(best_topology_state)
    return model.eval(), {
        "history": history,
        "best_topology_epoch": best_topology_epoch,
        "best_topology_score": best_topology,
        "best_total_normalized": best_total,
        "best_topology_state_dict": best_topology_state,
        "best_total_state_dict": best_total_state,
        "local_initial_scale": local_scale,
        "global_initial_scale": global_scale,
    }



@torch.no_grad()
def _residual_diagnostics(model: torch.nn.Module, source: RNAOnlySource, split: str, mean: np.ndarray, std: np.ndarray, device: torch.device) -> dict[str, float]:
    rows = source.rows(split)
    ratios = []
    cosines = []
    rel_bounds = []
    finite = True
    for start in range(0, rows.size, 2048):
        values = normalized_rows(source, rows[start : start + 2048], mean, std)
        tensor = torch.from_numpy(values).to(device)
        if hasattr(model, "decompose_raw"):
            scaffold, delta, z_raw = model.decompose_raw(tensor)
        else:
            scaffold = model.backbone(tensor)
            delta = model.bound * torch.tanh(model.adapter(scaffold))
            z_raw = scaffold + delta
        scaffold_norm = torch.linalg.vector_norm(scaffold, dim=1).clamp_min(1e-8)
        delta_norm = torch.linalg.vector_norm(delta, dim=1)
        z_norm = torch.linalg.vector_norm(z_raw, dim=1).clamp_min(1e-8)
        ratios.extend((delta_norm / scaffold_norm).detach().cpu().numpy().astype(float).tolist())
        cosines.extend(torch.sum(scaffold * z_raw, dim=1).div(scaffold_norm * z_norm).detach().cpu().numpy().astype(float).tolist())
        if hasattr(model, "rho"):
            rel_bounds.extend((delta_norm / (float(model.rho) * scaffold_norm)).detach().cpu().numpy().astype(float).tolist())
        finite = finite and bool(torch.isfinite(scaffold).all() and torch.isfinite(delta).all() and torch.isfinite(z_raw).all())
    ratio = np.asarray(ratios, dtype=np.float64)
    cosine = np.asarray(cosines, dtype=np.float64)
    rel = np.asarray(rel_bounds, dtype=np.float64) if rel_bounds else np.asarray([], dtype=np.float64)
    return {
        "cells": int(rows.size),
        "finite": bool(finite),
        "delta_over_scaffold_mean": float(ratio.mean()),
        "delta_over_scaffold_median": float(np.median(ratio)),
        "delta_over_scaffold_p90": float(np.percentile(ratio, 90)),
        "delta_over_scaffold_p95": float(np.percentile(ratio, 95)),
        "delta_over_scaffold_max": float(ratio.max()),
        "rna_z_cosine_mean": float(cosine.mean()),
        "rna_z_cosine_median": float(np.median(cosine)),
        "relative_bound_ratio_max": float(rel.max()) if rel.size else float("nan"),
        "near_bound_fraction_ge_0_90": float(np.mean(rel >= 0.90)) if rel.size else float("nan"),
        "bound_hit_fraction_ge_0_99": float(np.mean(rel >= 0.99)) if rel.size else float("nan"),
    }

def _save_target(path: Path, target: GlobalTarget) -> None:
    np.savez_compressed(path, landmarks=target.landmarks, values=target.values, mean=np.asarray(target.mean), std=np.asarray(target.std))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/u202522802045/share/zpb"))
    parser.add_argument("--source-quality-dir", type=Path, default=Path("/home/u202522802045/share/zpb/result_contain/pre_task14_dmode_source_quality_20260926"))
    parser.add_argument("--shared-rna-dir", type=Path, default=Path("/home/u202522802045/share/zpb/result_contain/pre_task14_dmode_shared_rna_audit_20260926"))
    parser.add_argument("--task11-dir", type=Path, default=Path("/home/u202522802045/share/zpb/result_contain/pre_task11_residual_audit_20260925"))
    parser.add_argument("--task23-dir", type=Path, default=Path("/home/u202522802045/share/zpb/result_contain/23_dmode_snf_correction_localization_audit_20260927"))
    parser.add_argument("--task26-dir", type=Path, default=Path("/home/u202522802045/share/zpb/result_contain/26_dmode_rna_scaffold_bounded_residual_20260929"))
    parser.add_argument("--task30-dir", type=Path, default=Path("/home/u202522802045/share/zpb/result_contain/30_dmode_structure_preserving_k60_pairbank_student_20260930"))
    parser.add_argument("--task31-dir", type=Path, default=Path("/home/u202522802045/share/zpb/result_contain/31_dmode_spectral_landmark_global_20260930"))
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
    rows = []
    config = V5DStudentConfig(
        latent_dim=64,
        distillation_loss="local_kl_plus_global_landmark",
        freeze_mode="shared_rna_backbone",
        teacher_target="Full Teacher spectral landmark-relational distance",
        max_epochs=max_epochs,
        checkpoint_metric="teacher_recall_at_10",
        pair_bank_seed=args.seed,
        local_teacher_temperature=0.07,
        local_student_temperature=0.07,
    )
    for fold_name in folds:
        target_key, source_keys = FOLDS[fold_name]
        fold_dir = args.source_quality_dir / fold_name
        sources = [RNAOnlySource(fold_dir / "harmonized", DATASET_NAMES[key]) for key in source_keys]
        target_source = RNAOnlySource(fold_dir / "harmonized", DATASET_NAMES[target_key])
        weights = np.load(fold_dir / "dataset_weights.npz")["omega"].astype(np.float64)
        dataset_weights = {source.name: float(weight) for source, weight in zip(sources, weights)}
        shared, mean, std, backbone = _load_shared_rna(
            args.shared_rna_dir / fold_name / "shared_rna_best.pt", sources, [*sources, target_source], args.v5_source, device
        )
        train_banks = {}
        val_banks = {}
        train_targets = {}
        val_targets = {}
        target_root = args.outdir / fold_name / "global_targets"
        target_root.mkdir(parents=True, exist_ok=True)
        for source in sources:
            train_bank = load_pair_bank(args.task26_dir / fold_name / "pair_banks" / f"{source.name}_train.npz")
            val_bank = load_pair_bank(args.task26_dir / fold_name / "pair_banks" / f"{source.name}_val.npz")
            train_bank, val_bank = _project_topology_embeddings(train_bank, val_bank, 64)
            train_banks[source.name] = train_bank
            val_banks[source.name] = val_bank
            train_graph = np.load(args.task11_dir / next(key for key, value in DATASET_NAMES.items() if value == source.name) / "seed_20020130" / "G_T_train.npy", mmap_mode="r")
            val_graph = np.load(args.task23_dir / "matched_geometry" / fold_name / source.name / "G_SNF_verified.npy", mmap_mode="r")
            train_target, fit_mean, fit_std = _build_global_target(train_graph)
            val_target, _, _ = _build_global_target(val_graph, fit_mean, fit_std)
            train_targets[source.name] = train_target
            val_targets[source.name] = val_target
            _save_target(target_root / f"{source.name}_train.npz", train_target)
            _save_target(target_root / f"{source.name}_val.npz", val_target)
        model, training = _train(sources, train_banks, val_banks, train_targets, val_targets, mean, std, config, args.seed, device, dataset_weights, backbone)
        out = args.outdir / fold_name / f"seed_{args.seed}"
        out.mkdir(parents=True, exist_ok=True)
        z_student_test = _raw_encode(model, target_source, "test", mean, std, device)
        z_rna_test = shared[target_source.name][target_source.test]
        labels_test = _labels(args.root, target_source, target_source.test)
        np.save(out / "z_student_test.npy", z_student_test)
        np.save(out / "z_rna_shared_test.npy", z_rna_test)
        baseline26 = json.loads((args.task26_dir / fold_name / f"seed_{args.seed}" / "target_metrics.json").read_text())
        baseline30 = json.loads((args.task30_dir / fold_name / f"seed_{args.seed}" / "target_metrics.json").read_text())
        baseline31 = json.loads((args.task31_dir / fold_name / f"seed_{args.seed}" / "target_metrics.json").read_text())
        target_metrics = {
            "RNA": baseline26["RNA"],
            "Task26": baseline26["Task26"],
            "Task30": baseline30["Task30"],
            "Task31": baseline31["Task31"],
            "Task32": evaluate_geometry(z_student_test, z_rna_test, labels_test, args.seed),
        }
        write_json(out / "target_metrics.json", target_metrics)
        source_validation = []
        for source in sources:
            z_val = _raw_encode(model, source, "val", mean, std, device)
            source_validation.append({
                "dataset": source.name,
                "topology": _topology_score(z_val, val_banks[source.name]),
                "global_teacher_fidelity": _global_metric(z_val, val_targets[source.name]),
                "labels_used": False,
                "test_used": False,
            })
        run_config = {
            **config.to_dict(),
            "task_id": 32,
            "student_parameterization": "R_embed + vector_norm_bound(adapter(R_embed), rho=0.10*||R_embed||)",
            "rna_scaffold_dim": 64,
            "adapter": "64->128->64",
            "residual_parameterization_change": "elementwise tanh bound -> smooth vector-norm relative bound",
            "rho": 0.10,
            "global_target": "normalized-Laplacian eigenvectors 2:65 -> 64 farthest-point landmarks -> landmark Euclidean distances",
            "global_target_normalization": "source-train-only scalar mean/std of train landmark distances",
            "loss": "Local-KL/local_initial_scale + global_landmark_MSE/global_initial_scale",
            "global_landmark_count": 64,
            "topology_embedding_projection": "train-fit PCA of cached 128D topology cache to 64D, validation transform only",
        }
        checkpoint = {
            "model_state_dict": training["best_topology_state_dict"],
            "seed": args.seed,
            "config": run_config,
            "rna_mean": mean,
            "rna_std": std,
            "checkpoint_type": "best_by_topology",
            "manifest": {
                "task_id": 32,
                "source_commit": args.source_commit,
                "target": DATASET_NAMES[target_key],
                "sources": [DATASET_NAMES[key] for key in source_keys],
                "student_input": "RNA only",
                "teacher_target": "spectral landmark-relational distance",
                "labels_used_for_training_or_selection": False,
            },
        }
        torch.save(checkpoint, out / "best_by_topology.pt")
        torch.save({**checkpoint, "model_state_dict": training["best_total_state_dict"], "checkpoint_type": "best_by_total_normalized_loss"}, out / "best_by_total_normalized_loss.pt")
        write_json(out / "config.json", run_config)
        write_json(out / "training_history.json", training["history"])
        write_json(out / "source_validation_metrics.json", source_validation)
        residual_diagnostics = {source.name: _residual_diagnostics(model, source, "val", mean, std, device) for source in sources}
        residual_diagnostics[DATASET_NAMES[target_key]] = _residual_diagnostics(model, target_source, "test", mean, std, device)
        write_json(out / "residual_diagnostics.json", residual_diagnostics)
        write_json(out / "global_target_stats.json", {source.name: train_targets[source.name].spectral for source in sources})
        write_json(out / "manifest.json", {
            "task_id": 32,
            "parent_tasks": [31],
            "experiment": "D-mode vector-norm-bounded residual Student",
            "source_commit": args.source_commit,
            "seed": args.seed,
            "target": DATASET_NAMES[target_key],
            "source_datasets": [DATASET_NAMES[key] for key in source_keys],
            "teacher_retrained": False,
            "student_input": "RNA only",
            "rna_scaffold_frozen": True,
            "adapter": "64->128->64",
            "residual_parameterization_change": "elementwise tanh bound -> smooth vector-norm relative bound",
            "rho": 0.10,
            "student_loss": "normalized Local-KL + normalized global landmark-relational MSE",
            "global_landmark_count": 64,
            "checkpoint_metric": "teacher_recall_at_10 validation topology",
            "labels_used_for_training": False,
            "labels_used_for_checkpoint_selection": False,
            "test_labels_used_only_for_final_metrics": True,
            "training_local_scale": training["local_initial_scale"],
            "training_global_scale": training["global_initial_scale"],
            "run_command": shlex.join([sys.executable, *sys.argv]),
        })
        rows.append({"fold": fold_name, "target": DATASET_NAMES[target_key], "seed": args.seed, "metrics": target_metrics, "source_validation": source_validation})
    write_json(args.outdir / "SUMMARY.json", {
        "task_id": 32,
        "parent_tasks": [31],
        "training_performed": True,
        "teacher_retrained": False,
        "snf_recomputed": False,
        "B_or_F_recomputed": False,
        "student_trained": True,
        "student_input": "RNA only",
        "labels_used_for_training_or_selection": False,
        "global_target": "64D normalized-Laplacian low-frequency Teacher coordinates with 64 farthest-point landmark distances",
        "folds": rows,
        "promotion_to_seed31": False,
    })


if __name__ == "__main__":
    main()
