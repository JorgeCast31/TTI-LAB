"""core.operators module"""
from __future__ import annotations
import numpy as np

def apply_transition(p: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply transition matrix T to probability vector p."""
    p = np.asarray(p, dtype=float)
    T = np.asarray(T, dtype=float)
    if T.ndim != 2:
        raise ValueError("T must be a 2D matrix")
    return T.dot(p)
