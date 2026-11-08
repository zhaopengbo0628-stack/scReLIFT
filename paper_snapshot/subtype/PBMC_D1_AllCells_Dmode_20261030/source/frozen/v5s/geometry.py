from __future__ import annotations

import numpy as np


def _normalized(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def cosine_similarity(values: np.ndarray) -> np.ndarray:
    return _normalized(values) @ _normalized(values).T
