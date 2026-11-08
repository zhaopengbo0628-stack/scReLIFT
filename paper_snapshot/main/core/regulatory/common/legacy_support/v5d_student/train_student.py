from __future__ import annotations

import copy
import math
from dataclasses import replace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.decomposition import PCA

from .config import V5DStudentConfig
from .data import RNAOnlySource, normalized_rows
from .pair_bank import PairBank, sample_pair_batch, sample_recoverability_pair_batch
from .student_encoder import RNAStudentEncoder
from v5s_student.geometry_metrics import teacher_alignment_metrics


def _tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(values).to(device)


def _model_output(model: nn.Module, values: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_raw"):
        return model.forward_raw(values)
    return model(values)


def _normalized_model_output(model: nn.Module, values: torch.Tensor) -> torch.Tensor:
    return F.normalize(_model_output(model, values), dim=1)


def _cross_modal_nt_xent(student: torch.Tensor, teacher: torch.Tensor, temperature: float) -> torch.Tensor:
    student = F.normalize(student, dim=1)
    teacher = F.normalize(teacher.detach(), dim=1)
    labels = torch.arange(student.shape[0], device=student.device)
    student_to_teacher = F.cross_entropy(student @ teacher.T / temperature, labels)
    teacher_to_student = F.cross_entropy(teacher @ student.T / temperature, labels)
    return (student_to_teacher + teacher_to_student) / 2.0


def _recoverability_geometry_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    local_count: int,
    boundary_count: int,
    config: V5DStudentConfig,
) -> torch.Tensor:
    width = local_count + boundary_count
    if prediction.numel() % width:
        raise ValueError("Recoverability geometry batch has an invalid pair width")
    errors = F.smooth_l1_loss(prediction, target, reduction="none").reshape(-1, width)
    weights = weight.reshape(-1, width)
    local_errors = errors[:, :local_count]
    local_weights = weights[:, :local_count]
    local_loss = (local_errors * local_weights).sum() / local_weights.sum().clamp_min(1e-8)
    if boundary_count == 0:
        return local_loss
    boundary_loss = errors[:, local_count:].mean()
    return config.local_geometry_weight * local_loss + config.boundary_geometry_weight * boundary_loss


def _local_kl_geometry_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    candidate_count: int,
    config: V5DStudentConfig,
) -> torch.Tensor:
    if prediction.numel() % candidate_count:
        raise ValueError("Local KL geometry batch has an invalid pair width")
    prediction = prediction.reshape(-1, candidate_count)
    target = target.reshape(-1, candidate_count)
    teacher_prob = F.softmax(target / config.local_teacher_temperature, dim=1)
    student_log_prob = F.log_softmax(prediction / config.local_student_temperature, dim=1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")


def _bank_pair_batch(bank: PairBank, anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = np.arange(bank.partner_index.shape[1], dtype=np.int64)
    anchor_matrix = np.broadcast_to(anchors[:, None], (anchors.size, columns.size))
    return (
        anchor_matrix.ravel(),
        bank.partner_index[anchor_matrix, columns].ravel(),
        bank.teacher_similarity[anchor_matrix, columns].ravel(),
    )


def _pair_cosine_loss(
    model: RNAStudentEncoder,
    source: RNAOnlySource,
    bank: PairBank,
    anchors: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    config: V5DStudentConfig,
    rng: np.random.Generator,
) -> torch.Tensor:
    if config.distillation_loss in {"geometry", "sampled_pair_mse"}:
        left, right, target = sample_pair_batch(
            bank, anchors, config.sampled_local_per_anchor, config.sampled_random_per_anchor, rng
        )
    else:
        left, right, target = _bank_pair_batch(bank, anchors)
    split_rows = source.train
    left_z = _normalized_model_output(model, _tensor(normalized_rows(source, split_rows[left], mean, std), device))
    right_z = _normalized_model_output(model, _tensor(normalized_rows(source, split_rows[right], mean, std), device))
    prediction = (left_z * right_z).sum(dim=1)
    return F.mse_loss(prediction, _tensor(target.astype(np.float32), device))


def _gram_mse_loss(
    model: RNAStudentEncoder,
    source: RNAOnlySource,
    bank: PairBank,
    anchors: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    if bank.teacher_embedding is None:
        raise ValueError("Gram MSE requires cached teacher embeddings")
    values = normalized_rows(source, source.train[anchors], mean, std)
    student = model(_tensor(values, device))
    teacher = _tensor(bank.teacher_embedding[anchors], device)
    return F.mse_loss(student @ student.T, teacher @ teacher.T)


def _validation_topology(
    embeddings: dict[str, np.ndarray],
    banks: dict[str, PairBank],
    checkpoint_metric: str,
    dataset_weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, dict[str, float]]]:
    by_dataset = {}
    scores = []
    weights = []
    for name, embedding in embeddings.items():
        if banks[name].teacher_embedding is None:
            raise ValueError("Topology checkpointing requires cached teacher embeddings")
        metrics = teacher_alignment_metrics(embedding, banks[name].teacher_embedding)
        by_dataset[name] = metrics
        if checkpoint_metric not in metrics:
            raise ValueError(f"Unsupported checkpoint metric: {checkpoint_metric}")
        scores.append(metrics[checkpoint_metric])
        weights.append(1.0 if dataset_weights is None else dataset_weights[name])
    return float(np.average(scores, weights=weights)), by_dataset


def _project_teacher_embeddings(
    train_banks: dict[str, PairBank],
    val_banks: dict[str, PairBank],
    target_dim: int,
) -> dict[str, int]:
    projected_dims = {}
    for name, train_bank in train_banks.items():
        if train_bank.teacher_embedding is None or val_banks[name].teacher_embedding is None:
            raise ValueError("Teacher embedding projection requires cached teacher embeddings")
        train_teacher = np.asarray(train_bank.teacher_embedding, dtype=np.float32)
        val_teacher = np.asarray(val_banks[name].teacher_embedding, dtype=np.float32)
        if train_teacher.shape[1] > target_dim:
            components = min(target_dim, train_teacher.shape[0] - 1, train_teacher.shape[1])
            reducer = PCA(n_components=components, svd_solver="randomized", random_state=0)
            train_projected = reducer.fit_transform(train_teacher)
            val_projected = reducer.transform(val_teacher)
        else:
            train_projected = train_teacher
            val_projected = val_teacher
        if train_projected.shape[1] < target_dim:
            width = target_dim - train_projected.shape[1]
            train_projected = np.pad(train_projected, ((0, 0), (0, width)))
            val_projected = np.pad(val_projected, ((0, 0), (0, width)))
        train_projected = train_projected / np.maximum(np.linalg.norm(train_projected, axis=1, keepdims=True), 1e-8)
        val_projected = val_projected / np.maximum(np.linalg.norm(val_projected, axis=1, keepdims=True), 1e-8)
        train_banks[name] = replace(train_bank, teacher_embedding=train_projected.astype(np.float32))
        val_banks[name] = replace(val_banks[name], teacher_embedding=val_projected.astype(np.float32))
        projected_dims[name] = int(train_teacher.shape[1])
    return projected_dims


@torch.no_grad()
def encode_split(
    model: RNAStudentEncoder,
    source: RNAOnlySource,
    split: str,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    rows = source.rows(split)
    outputs = []
    for start in range(0, rows.size, batch_size):
        values = normalized_rows(source, rows[start:start + batch_size], mean, std)
        outputs.append(_model_output(model, _tensor(values, device)).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


@torch.no_grad()
def validation_loss(model: RNAStudentEncoder, embeddings: dict[str, np.ndarray], banks: dict[str, PairBank], config: V5DStudentConfig, device: torch.device, dataset_weights: dict[str, float] | None = None) -> tuple[float, dict[str, float]]:
    model.eval()
    losses = {}
    for name, bank in banks.items():
        z = _tensor(embeddings[name], device)
        if config.distillation_loss == "cross_modal_nt_xent":
            if bank.teacher_embedding is None:
                raise ValueError("Contrastive distillation requires cached teacher embeddings")
            teacher = _tensor(bank.teacher_embedding, device)
            losses[name] = float(_cross_modal_nt_xent(z, teacher, config.contrastive_temperature).cpu())
        elif config.distillation_loss == "recoverability_geometry":
            if bank.pair_weight is None or bank.teacher_embedding is None:
                raise ValueError("Recoverability geometry requires pair weights and teacher embeddings")
            anchors = torch.arange(bank.anchors, device=device)[:, None].expand_as(_tensor(bank.partner_index, device))
            partners = _tensor(bank.partner_index, device)
            target = _tensor(bank.teacher_similarity, device)
            weight = _tensor(bank.pair_weight, device)
            z = F.normalize(z, dim=1)
            prediction = (z[anchors] * z[partners]).sum(dim=2).reshape(-1)
            boundary_count = bank.partner_index.shape[1] - bank.local_k
            geometry = _recoverability_geometry_loss(
                prediction, target.reshape(-1), weight.reshape(-1), bank.local_k, boundary_count, config
            )
            teacher = _tensor(bank.teacher_embedding, device)
            anchor_loss = 1.0 - (z * teacher).sum(dim=1).mean()
            losses[name] = float(((1.0 - config.anchor_geometry_weight) * geometry + config.anchor_geometry_weight * anchor_loss).cpu())
        elif config.distillation_loss == "local_kl":
            anchors = torch.arange(bank.anchors, device=device)[:, None].expand_as(_tensor(bank.partner_index, device))
            partners = _tensor(bank.partner_index, device)
            target = _tensor(bank.teacher_similarity, device)
            z = F.normalize(z, dim=1)
            prediction = (z[anchors] * z[partners]).sum(dim=2).reshape(-1)
            losses[name] = float(_local_kl_geometry_loss(
                prediction, target.reshape(-1), bank.partner_index.shape[1], config
            ).cpu())
        elif config.distillation_loss == "gram_mse":
            if bank.teacher_embedding is None:
                raise ValueError("Gram MSE requires cached teacher embeddings")
            teacher = _tensor(bank.teacher_embedding, device)
            losses[name] = float(F.mse_loss(z @ z.T, teacher @ teacher.T).cpu())
        else:
            anchors = torch.arange(bank.anchors, device=device)[:, None].expand_as(_tensor(bank.partner_index, device))
            partners = _tensor(bank.partner_index, device)
            target = _tensor(bank.teacher_similarity, device)
            losses[name] = float(F.mse_loss((z[anchors] * z[partners]).sum(dim=2), target).cpu())
    if dataset_weights is None:
        return float(np.mean(list(losses.values()))), losses
    weighted_sum = sum(losses[name] * dataset_weights[name] for name in losses.keys())
    weight_total = sum(dataset_weights[name] for name in losses.keys())
    return float(weighted_sum / weight_total), losses


def train_shared_student(
    sources: list[RNAOnlySource],
    train_banks: dict[str, PairBank],
    val_banks: dict[str, PairBank],
    mean: np.ndarray,
    std: np.ndarray,
    config: V5DStudentConfig,
    seed: int,
    device: torch.device,
    dataset_weights: dict[str, float] | None = None,
    model: nn.Module | None = None,
) -> tuple[nn.Module, dict]:
    if config.distillation_loss not in {
        "pair_mse",
        "sampled_pair_mse",
        "geometry",
        "gram_mse",
        "cross_modal_nt_xent",
        "recoverability_geometry",
        "local_kl",
    }:
        raise ValueError(f"Unsupported distillation loss: {config.distillation_loss}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    gene_counts = {source.rna.shape[1] for source in sources}
    if len(gene_counts) != 1:
        raise ValueError("All V5-D sources must use the same RNA gene space")
    for source in sources:
        if train_banks[source.name].anchors != source.train.size or val_banks[source.name].anchors != source.val.size:
            raise ValueError(f"{source.name}: pair bank does not match its split")
    if model is None:
        model = RNAStudentEncoder(
            sources[0].rna.shape[1],
            config.hidden_dim_1,
            config.hidden_dim_2,
            config.latent_dim,
            config.dropout,
        )
    model = model.to(device)
    teacher_projection_dims = {}
    if config.distillation_loss in {"cross_modal_nt_xent", "recoverability_geometry"}:
        teacher_projection_dims = _project_teacher_embeddings(train_banks, val_banks, config.latent_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    rng = np.random.default_rng(seed)
    steps = max(math.ceil(source.train.size / config.batch_anchor) for source in sources)
    best_loss_state, best_loss, best_loss_epoch, wait = None, float("inf"), 0, 0
    best_topology_state, best_topology_score, best_topology_epoch = None, -float("inf"), 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_total = 0.0
        for _ in range(steps):
            dataset_losses = []
            for source in sources:
                bank = train_banks[source.name]
                anchors = rng.integers(0, bank.anchors, size=min(config.batch_anchor, bank.anchors))
                if config.distillation_loss == "cross_modal_nt_xent":
                    if bank.teacher_embedding is None:
                        raise ValueError("Contrastive distillation requires cached teacher embeddings")
                    values = normalized_rows(source, source.train[anchors], mean, std)
                    teacher = _tensor(bank.teacher_embedding[anchors], device)
                    view_a = model(_tensor(values, device))
                    view_b = model(_tensor(values, device))
                    dataset_losses.append(
                        (_cross_modal_nt_xent(view_a, teacher, config.contrastive_temperature)
                         + _cross_modal_nt_xent(view_b, teacher, config.contrastive_temperature)) / 2.0
                    )
                elif config.distillation_loss == "recoverability_geometry":
                    if bank.pair_weight is None or bank.teacher_embedding is None:
                        raise ValueError("Recoverability geometry requires pair weights and teacher embeddings")
                    left, right, target, weight = sample_recoverability_pair_batch(
                        bank, anchors, config.sampled_local_per_anchor, config.sampled_random_per_anchor, rng
                    )
                    split_rows = source.train
                    left_rna = normalized_rows(source, split_rows[left], mean, std)
                    right_rna = normalized_rows(source, split_rows[right], mean, std)
                    left_z = model(_tensor(left_rna, device))
                    right_z = model(_tensor(right_rna, device))
                    prediction = (left_z * right_z).sum(dim=1)
                    geometry = _recoverability_geometry_loss(
                        prediction,
                        _tensor(target.astype(np.float32), device),
                        _tensor(weight.astype(np.float32), device),
                        config.sampled_local_per_anchor,
                        config.sampled_random_per_anchor,
                        config,
                    )
                    values = normalized_rows(source, source.train[anchors], mean, std)
                    anchor_z = model(_tensor(values, device))
                    anchor_target = _tensor(bank.teacher_embedding[anchors], device)
                    anchor_loss = 1.0 - (anchor_z * anchor_target).sum(dim=1).mean()
                    dataset_losses.append(
                        (1.0 - config.anchor_geometry_weight) * geometry
                        + config.anchor_geometry_weight * anchor_loss
                    )
                elif config.distillation_loss == "local_kl":
                    left, right, target = _bank_pair_batch(bank, anchors)
                    split_rows = source.train
                    left_rna = normalized_rows(source, split_rows[left], mean, std)
                    right_rna = normalized_rows(source, split_rows[right], mean, std)
                    prediction = (_normalized_model_output(model, _tensor(left_rna, device)) * _normalized_model_output(model, _tensor(right_rna, device))).sum(dim=1)
                    dataset_losses.append(_local_kl_geometry_loss(
                        prediction,
                        _tensor(target.astype(np.float32), device),
                        bank.partner_index.shape[1],
                        config,
                    ))
                elif config.distillation_loss == "gram_mse":
                    dataset_losses.append(_gram_mse_loss(model, source, bank, anchors, mean, std, device))
                else:
                    dataset_losses.append(
                        _pair_cosine_loss(model, source, bank, anchors, mean, std, device, config, rng)
                    )
            if dataset_weights is None:
                loss = torch.stack(dataset_losses).mean()
            else:
                weighted_losses = [dataset_losses[i] * dataset_weights[sources[i].name] for i in range(len(sources))]
                weight_total = sum(dataset_weights[sources[i].name] for i in range(len(sources)))
                loss = torch.stack(weighted_losses).sum() / weight_total
            if not torch.isfinite(loss):
                raise FloatingPointError("Student geometry loss is not finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            train_total += float(loss.detach().cpu())
        val_embeddings = {source.name: encode_split(model, source, "val", mean, std, device) for source in sources}
        val_loss, val_by_dataset = validation_loss(model, val_embeddings, val_banks, config, device, dataset_weights)
        val_topology_score, val_topology_by_dataset = _validation_topology(
            val_embeddings, val_banks, config.checkpoint_metric, dataset_weights
        )
        if not np.isfinite(val_loss):
            raise FloatingPointError("Student validation geometry loss is not finite")
        if not np.isfinite(val_topology_score):
            raise FloatingPointError("Student validation topology score is not finite")
        record = {
            "epoch": epoch,
            "train_loss": train_total / steps,
            "val_loss_macro": val_loss,
            "val_loss_by_dataset": val_by_dataset,
            "val_topology_score_macro": val_topology_score,
            "val_topology_by_dataset": val_topology_by_dataset,
        }
        history.append(record)
        print(record, flush=True)
        if val_loss < best_loss - 1e-8:
            best_loss_state, best_loss, best_loss_epoch, wait = copy.deepcopy(model.state_dict()), val_loss, epoch, 0
        else:
            wait += 1
        if val_topology_score > best_topology_score + 1e-8:
            best_topology_state = copy.deepcopy(model.state_dict())
            best_topology_score = val_topology_score
            best_topology_epoch = epoch
        if wait >= config.patience:
            break
    if best_loss_state is None or best_topology_state is None:
        raise RuntimeError("No Student checkpoint was selected")
    model.load_state_dict(best_topology_state)
    return model.eval(), {
        "best_epoch": best_topology_epoch,
        "best_loss_epoch": best_loss_epoch,
        "best_topology_epoch": best_topology_epoch,
        "best_val_loss_macro": best_loss,
        "best_val_geometry_mse_macro": best_loss,
        "best_val_topology_score_macro": best_topology_score,
        "checkpoint_metric": config.checkpoint_metric,
        "selected_checkpoint": "best_by_topology",
        "best_loss_state_dict": best_loss_state,
        "best_topology_state_dict": best_topology_state,
        "teacher_projection_input_dims": teacher_projection_dims,
        "teacher_projection_output_dim": config.latent_dim if config.distillation_loss in {"cross_modal_nt_xent", "recoverability_geometry"} else None,
        "distillation_loss": config.distillation_loss,
        "history": history,
    }
