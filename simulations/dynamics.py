"""simulations.dynamics module"""
from __future__ import annotations
import numpy as np
from core.operators import apply_transition
from core.probability import normalize

def simulate_markov_chain(p0: np.ndarray, T: np.ndarray, steps: int = 10, normalize_each_step: bool = True) -> np.ndarray:
    """Simulate a discrete Markov chain."""
    p = np.asarray(p0, dtype=float)
    n = p.size
    traj = np.zeros((steps+1, n), dtype=float)
    traj[0] = normalize(p)
    for t in range(1, steps+1):
        p = apply_transition(p, T)
        if normalize_each_step:
            p = normalize(p)
        traj[t] = p
    return traj
