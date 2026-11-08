from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class PairBank:
    partner_index: np.ndarray
    teacher_similarity: np.ndarray
    local_k: int
    teacher_embedding: np.ndarray | None = None
    pair_weight: np.ndarray | None = None
    anchor_recoverability: np.ndarray | None = None

    @property
    def anchors(self) -> int:
        return int(self.partner_index.shape[0])


def _normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def build_pair_bank(teacher: np.ndarray, neighbor_k: int, random_k: int, seed: int) -> PairBank:
    teacher = _normalize(np.asarray(teacher, dtype=np.float32))
    cells = teacher.shape[0]
    if cells < 2:
        raise ValueError("A pair bank requires at least two cells")
    local_k = min(neighbor_k, cells - 1)
    query_k = min(local_k + 1, cells)
    raw_neighbors = NearestNeighbors(n_neighbors=query_k, metric="cosine").fit(teacher).kneighbors(teacher, return_distance=False)
    local = np.empty((cells, local_k), dtype=np.int64)
    for anchor in range(cells):
        local[anchor] = [value for value in raw_neighbors[anchor] if value != anchor][:local_k]
    rng = np.random.default_rng(seed)
    random_partners = np.empty((cells, random_k), dtype=np.int64)
    universe = np.arange(cells, dtype=np.int64)
    for anchor in range(cells):
        allowed = np.setdiff1d(universe, np.concatenate(([anchor], local[anchor])), assume_unique=False)
        if allowed.size == 0:
            allowed = np.setdiff1d(universe, [anchor])
        random_partners[anchor] = rng.choice(allowed, size=random_k, replace=allowed.size < random_k)
    partners = np.concatenate((local, random_partners), axis=1)
    targets = np.sum(teacher[:, None, :] * teacher[partners], axis=2).astype(np.float32)
    return PairBank(partners, targets, local_k, teacher)


def build_recoverability_pair_bank(
    teacher: np.ndarray,
    rna: np.ndarray,
    neighbor_k: int,
    boundary_k: int,
    correspondence_floor: float,
) -> PairBank:
    teacher = _normalize(np.asarray(teacher, dtype=np.float32))
    rna = _normalize(np.asarray(rna, dtype=np.float32))
    if teacher.shape[0] != rna.shape[0]:
        raise ValueError("Teacher and RNA must contain the same cells")
    if not 0.0 <= correspondence_floor <= 1.0:
        raise ValueError("correspondence_floor must be in [0, 1]")
    cells = teacher.shape[0]
    if cells < 2:
        raise ValueError("A pair bank requires at least two cells")
    local_k = min(neighbor_k, cells - 1)
    boundary_k = min(boundary_k, cells - 1 - local_k)
    query_k = min(local_k + boundary_k + 1, cells)
    teacher_neighbors = NearestNeighbors(n_neighbors=query_k, metric="cosine").fit(teacher).kneighbors(teacher, return_distance=False)
    local = np.empty((cells, local_k), dtype=np.int64)
    boundary = np.empty((cells, boundary_k), dtype=np.int64)
    pair_weight = np.ones((cells, local_k + boundary_k), dtype=np.float32)
    anchor_recoverability = np.empty(cells, dtype=np.float32)
    for anchor in range(cells):
        ordered = teacher_neighbors[anchor][teacher_neighbors[anchor] != anchor]
        local[anchor] = ordered[:local_k]
        boundary[anchor] = ordered[local_k:local_k + boundary_k]
        candidate_partners = np.concatenate((local[anchor], boundary[anchor]))
        rna_similarity = rna[anchor] @ rna[candidate_partners].T
        center = float(np.median(rna_similarity))
        scale = max(float(np.std(rna_similarity)), 1e-3)
        local_recoverability = 1.0 / (1.0 + np.exp(-(rna_similarity[:local_k] - center) / scale))
        anchor_recoverability[anchor] = float(local_recoverability.mean()) if local_k else 0.0
        pair_weight[anchor, :local_k] = correspondence_floor + (1.0 - correspondence_floor) * local_recoverability
    partners = np.concatenate((local, boundary), axis=1)
    targets = np.sum(teacher[:, None, :] * teacher[partners], axis=2).astype(np.float32)
    return PairBank(partners, targets, local_k, teacher, pair_weight, anchor_recoverability)


def save_pair_bank(bank: PairBank, path: Path) -> None:
    pair_type = np.zeros_like(bank.partner_index, dtype=np.uint8)
    pair_type[:, bank.local_k:] = 1
    values = {
        "anchor_index": np.arange(bank.anchors, dtype=np.int64),
        "partner_index": bank.partner_index,
        "teacher_similarity": bank.teacher_similarity,
        "pair_type": pair_type,
        "local_k": np.asarray(bank.local_k, dtype=np.int64),
    }
    if bank.teacher_embedding is not None:
        values["teacher_embedding"] = bank.teacher_embedding
    if bank.pair_weight is not None:
        values["pair_weight"] = bank.pair_weight
    if bank.anchor_recoverability is not None:
        values["anchor_recoverability"] = bank.anchor_recoverability
    np.savez_compressed(path, **values)


def load_pair_bank(path: Path) -> PairBank:
    with np.load(path) as values:
        teacher_embedding = values["teacher_embedding"] if "teacher_embedding" in values else None
        pair_weight = values["pair_weight"] if "pair_weight" in values else None
        anchor_recoverability = values["anchor_recoverability"] if "anchor_recoverability" in values else None
        return PairBank(values["partner_index"], values["teacher_similarity"], int(values["local_k"]), teacher_embedding, pair_weight, anchor_recoverability)


def sample_pair_batch(
    bank: PairBank,
    anchors: np.ndarray,
    local_count: int,
    random_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_columns = rng.integers(0, bank.local_k, size=(anchors.size, local_count))
    random_width = bank.partner_index.shape[1] - bank.local_k
    random_columns = bank.local_k + rng.integers(0, random_width, size=(anchors.size, random_count))
    columns = np.concatenate((local_columns, random_columns), axis=1)
    anchor_matrix = np.broadcast_to(anchors[:, None], columns.shape)
    return anchor_matrix.ravel(), bank.partner_index[anchor_matrix, columns].ravel(), bank.teacher_similarity[anchor_matrix, columns].ravel()


def sample_recoverability_pair_batch(
    bank: PairBank,
    anchors: np.ndarray,
    local_count: int,
    boundary_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if bank.pair_weight is None:
        raise ValueError("Recoverability geometry requires pair weights")
    boundary_width = bank.partner_index.shape[1] - bank.local_k
    if boundary_width == 0:
        raise ValueError("Recoverability geometry requires boundary pairs")
    local_columns = rng.integers(0, bank.local_k, size=(anchors.size, local_count))
    boundary_columns = bank.local_k + rng.integers(0, boundary_width, size=(anchors.size, boundary_count))
    columns = np.concatenate((local_columns, boundary_columns), axis=1)
    anchor_matrix = np.broadcast_to(anchors[:, None], columns.shape)
    return (
        anchor_matrix.ravel(),
        bank.partner_index[anchor_matrix, columns].ravel(),
        bank.teacher_similarity[anchor_matrix, columns].ravel(),
        bank.pair_weight[anchor_matrix, columns].ravel(),
    )
