"""core.information module"""
from __future__ import annotations
import numpy as np

def entropy(p: np.ndarray, base: float = 2.0) -> float:
    """Shannon entropy of probability vector."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return -np.sum(p * np.log(p) / np.log(base))

def kl_divergence(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
    """KL divergence D(p || q)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0
    return np.sum(p[mask] * (np.log(p[mask]/q[mask]) / np.log(base)))

def mutual_information(joint: np.ndarray, base: float = 2.0) -> float:
    """Mutual information between X and Y."""
    joint = np.asarray(joint, dtype=float)
    total = joint.sum()
    if total == 0:
        return 0.0
    joint = joint / total
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    mi = 0.0
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            pxy = joint[i,j]
            if pxy > 0:
                mi += pxy * (np.log(pxy / (px[i]*py[j])) / np.log(base))
    return mi
