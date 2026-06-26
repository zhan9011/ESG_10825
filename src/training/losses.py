from __future__ import annotations

import math

import numpy as np


def sqrt_balanced_class_weights(counts: np.ndarray, max_weight: float) -> np.ndarray:
    """Compute capped square-root inverse-frequency class weights."""
    total = int(counts.sum())
    classes = len(counts)
    return np.asarray(
        [min(max_weight, math.sqrt(total / (classes * max(1, int(count)))))
         for count in counts],
        dtype=np.float32,
    )
