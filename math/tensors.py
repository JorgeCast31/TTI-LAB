"""
math.tensors
============

Utilidades para manipulación de tensores informacionales (p. ej. unfolding).
"""

from __future__ import annotations
import numpy as np
from typing import Tuple

def unfold_tensor(t: np.ndarray, mode: int) -> np.ndarray:
    """
    Matriciza (unfold) un tensor `t` según el modo dado.

    Parameters
    ----------
    t : np.ndarray
    mode : int
        Modo sobre el que unfolds (0..t.ndim-1)

    Returns
    -------
    np.ndarray
        Matriz unfold de shape (t.shape[mode], -1)
    """
    return np.reshape(np.moveaxis(t, mode, 0), (t.shape[mode], -1))

def fold_tensor(mat: np.ndarray, mode: int, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Reconstruye un tensor a partir de su matricización `mat` y `shape`.
    """
    full = np.moveaxis(np.reshape(mat, (shape[mode],) + tuple(s for i,s in enumerate(shape) if i!=mode)), 0, mode)
    return full