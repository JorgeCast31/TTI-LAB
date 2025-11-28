"""core.probability module"""
from __future__ import annotations
import numpy as np
from typing import Tuple

def normalize(p: np.ndarray, axis: int = None, eps: float = 1e-12) -> np.ndarray:
    """Normalize array to sum to 1."""
    p = np.array(p, dtype=float)
    if axis is None:
        s = p.sum()
        if s == 0:
            return np.full_like(p, fill_value=1.0 / p.size)
        return p / (s + eps)
    s = p.sum(axis=axis, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return p / (s + eps)

def is_probability_vector(p: np.ndarray, tol: float = 1e-8) -> bool:
    """Check if p is a valid probability vector."""
    p = np.asarray(p)
    return (p >= -tol).all() and abs(p.sum() - 1.0) <= tol
