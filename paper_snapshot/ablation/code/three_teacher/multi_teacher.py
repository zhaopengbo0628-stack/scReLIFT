"""Teacher-strategy primitives; the full shared B2 architecture is unchanged."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class TeacherSelection:
    selected_source: str
    equal_weight_validation_loss: dict[str, float]


def select_best_single_teacher(validation_losses: Mapping[str, Mapping[str, float]]) -> TeacherSelection:
    if not validation_losses or any(not values for values in validation_losses.values()):
        raise ValueError("each source-specific Teacher needs validation loss on every source")
    scores = {name: sum(float(v) for v in losses.values()) / len(losses)
              for name, losses in validation_losses.items()}
    if not all(torch.isfinite(torch.tensor(v)) for v in scores.values()):
        raise ValueError("validation losses must be finite")
    return TeacherSelection(min(scores, key=lambda k: (scores[k], k)), scores)


def select_student_target(teacher_source: str, all_sources: tuple[str, ...]) -> tuple[str, ...]:
    if teacher_source not in all_sources or len(set(all_sources)) != 3:
        raise ValueError("expected one selected source Teacher and exactly three sources")
    # Each candidate Student sees all three sources; selection reuses its matching run.
    return all_sources


def validate_teacher_matrix(matrix: Mapping[str, Mapping[str, float]], sources: tuple[str, ...]) -> None:
    if set(matrix) != set(sources) or any(set(row) != set(sources) for row in matrix.values()):
        raise ValueError("teacher validation matrix must contain every Teacher x source-val cell")
