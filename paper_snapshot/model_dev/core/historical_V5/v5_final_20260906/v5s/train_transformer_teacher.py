from __future__ import annotations

import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse

from .config import V5SConfig
from .regulatory_transformer import RegulatoryTransformer


def _loss_epoch(
    model: RegulatoryTransformer,
    atac: sparse.csr_matrix,
    indices: np.ndarray,
    row_permutation: np.ndarray,
    config: V5SConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    rng: np.random.Generator,
    include_consistency: bool | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    include_consistency = training if include_consistency is None else include_consistency
    model.train(training)
    total_pred = total_cons = total = 0.0
    seen = 0
    order = rng.permutation(indices) if training else indices
    for start in range(0, order.size, config.batch_size):
        rows = order[start : start + config.batch_size]
        x = torch.from_numpy(atac[row_permutation[rows]].toarray().astype(np.float32)).to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        pred_loss = x.new_tensor(0.0)
        cons_loss = x.new_tensor(0.0)
        for gene_start in range(0, model.num_genes, config.gene_chunk_size):
            gene_stop = min(gene_start + config.gene_chunk_size, model.num_genes)
            clean = model(x, gene_start, gene_stop)
            if include_consistency:
                masked_z, reconstruction, mask = model(
                    x, gene_start, gene_stop, masked=True, mask_rate=config.mask_rate, return_reconstruction=True
                )
                local_mask = mask & model.candidate_mask[gene_start:gene_stop].unsqueeze(0)
                if local_mask.any():
                    original = x[:, model.candidate_index[gene_start:gene_stop]]
                    pred_loss = pred_loss + F.smooth_l1_loss(reconstruction[local_mask], original[local_mask])
                cons_loss = cons_loss + F.mse_loss(masked_z, clean.detach())
        chunks = (model.num_genes + config.gene_chunk_size - 1) // config.gene_chunk_size
        pred_loss = pred_loss / chunks
        cons_loss = cons_loss / chunks if include_consistency else pred_loss.new_tensor(0.0)
        loss = pred_loss + config.consistency_weight * cons_loss
        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        count = rows.size
        total_pred += float(pred_loss.detach().cpu()) * count
        total_cons += float(cons_loss.detach().cpu()) * count
        total += float(loss.detach().cpu()) * count
        seen += count
    return {"loss": total / seen, "reconstruction_loss": total_pred / seen, "consistency_loss": total_cons / seen}


def train_transformer_teacher(
    model: RegulatoryTransformer,
    atac: sparse.csr_matrix,
    train_index: np.ndarray,
    val_index: np.ndarray,
    row_permutation: np.ndarray,
    config: V5SConfig,
    seed: int,
    outdir: Path,
    device: torch.device,
) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_rng = np.random.default_rng(seed)
    val_rng = np.random.default_rng(seed + 1)
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    wait = 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        started = time.perf_counter()
        train_metrics = _loss_epoch(model, atac, train_index, row_permutation, config, device, optimizer, train_rng)
        with torch.no_grad():
            val_metrics = _loss_epoch(model, atac, val_index, row_permutation, config, device, None, val_rng, True)
        record = {"epoch": epoch, "seconds": time.perf_counter() - started}
        record.update({f"train_{key}": value for key, value in train_metrics.items()})
        record.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(record)
        print(record, flush=True)
        if val_metrics["loss"] < best_loss - 1e-8:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            wait = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            wait += 1
        if wait >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("Transformer did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    checkpoint = {
        "model_state": model.state_dict(),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "config": config.to_dict(),
    }
    torch.save(checkpoint, outdir / "best_teacher.pt")
    (outdir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"best_epoch": best_epoch, "best_val_loss": best_loss, "history": history}


@torch.no_grad()
def encode_transformer(
    model: RegulatoryTransformer,
    atac: sparse.csr_matrix,
    rows: np.ndarray,
    config: V5SConfig,
    device: torch.device,
    row_permutation: np.ndarray | None = None,
) -> np.ndarray:
    model.eval()
    permutation = np.arange(atac.shape[0], dtype=np.int64) if row_permutation is None else row_permutation
    outputs = []
    for start in range(0, rows.size, config.batch_size):
        batch_rows = rows[start : start + config.batch_size]
        x = torch.from_numpy(atac[permutation[batch_rows]].toarray().astype(np.float32)).to(device)
        chunks = [model(x, gene_start, min(gene_start + config.gene_chunk_size, model.num_genes)).cpu().numpy() for gene_start in range(0, model.num_genes, config.gene_chunk_size)]
        outputs.append(np.concatenate(chunks, axis=1).astype(np.float32))
    return np.concatenate(outputs, axis=0)
