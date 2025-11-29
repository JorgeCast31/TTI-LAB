"""
simulations.dynamics
====================

Funciones para simular dinámicas informacionales (ej. cadenas de Markov).
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Optional
from core.operators import apply_transition
from core.probability import normalize

def simulate_markov_chain(p0: np.ndarray, T: np.ndarray, steps: int = 10, normalize_each_step: bool = True) -> np.ndarray:
    """
    Simula una cadena de Markov discreta.

    Parameters
    ----------
    p0 : np.ndarray
        Distribución inicial (n,)
    T : np.ndarray
        Matriz de transición (n, n)
    steps : int
        Número de pasos a simular
    normalize_each_step : bool
        Si True, normaliza p en cada paso (útil si T no es estocástica perfecta)

    Returns
    -------
    np.ndarray
        Array con forma (steps+1, n) con las distribuciones en cada paso.

    Examples
    --------
    >>> import numpy as np
    >>> from simulations.dynamics import simulate_markov_chain
    >>> p0 = np.array([1.0, 0.0])
    >>> T = np.array([[0.9, 0.2],[0.1,0.8]])
    >>> traj = simulate_markov_chain(p0, T, steps=3)
    >>> traj.shape
    (4, 2)
    """
    p = np.asarray(p0, dtype=float)
    n = p.size
    traj = np.zeros((steps+1, n), dtype=float)
    traj[0] = normalize(p)
    for t in range(1, steps+1):
        # apply_transition ahora usa la convención fila: p @ T
        p = apply_transition(p, T)
        if normalize_each_step:
            p = normalize(p)
        traj[t] = p
    return traj