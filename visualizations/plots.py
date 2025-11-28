"""
visualizations.plots
====================

Funciones comunes de visualización para distribuciones, trayectorias
y matrices de transición en el laboratorio TTI.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence, Optional


__all__ = [
    "plot_distribution",
    "plot_trajectory",
    "plot_transition_matrix",
    "plot_entropy",
]


# ============================================================
# 1. Distribución instantánea p(t)
# ============================================================


def plot_distribution(
    p: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Distribution",
):
    """
    Bar plot de una distribución de probabilidad.

    Parameters
    ----------
    p : np.ndarray
        Vector de probabilidad.
    labels : list of str, optional
        Etiquetas de cada estado.
    ax : matplotlib Axes, optional
        Ejes donde dibujar.
    title : str
        Título.

    Returns
    -------
    ax : matplotlib Axes
    """
    p = np.asarray(p, dtype=float)

    if ax is None:
        fig, ax = plt.subplots()

    x = np.arange(len(p))

    if labels is None:
        labels = [f"s{i}" for i in range(len(p))]

    ax.bar(x, p, tick_label=labels)

    ax.set_title(title)
    ax.set_xlabel("State")
    ax.set_ylabel("Probability")
    ax.grid(alpha=0.2)

    return ax


# ============================================================
# 2. Trayectoria temporal p(t)
# ============================================================


def plot_trajectory(
    traj: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Trajectory",
):
    """
    Grafica la evolución temporal de una distribución p(t).

    Parameters
    ----------
    traj : np.ndarray
        Array de shape (T, n_states).
    ax : matplotlib Axes, optional
        Ejes donde dibujar.
    title : str
        Título.

    Returns
    -------
    ax : matplotlib Axes
    """
    traj = np.asarray(traj, dtype=float)

    if ax is None:
        fig, ax = plt.subplots()

    T, n_states = traj.shape

    for i in range(n_states):
        ax.plot(
            np.arange(T),
            traj[:, i],
            label=f"state_{i}",
            linewidth=2,
        )

    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Probability")
    ax.legend()
    ax.grid(alpha=0.3)

    return ax


# ============================================================
# 3. Matriz de transición
# ============================================================


def plot_transition_matrix(
    T: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Transition Matrix",
):
    """
    Muestra un mapa de calor de la matriz de transición T.

    Parameters
    ----------
    T : np.ndarray
        Matriz de transición (n x n).
    ax : matplotlib Axes, optional
        Ejes donde dibujar.
    title : str
        Título.

    Returns
    -------
    ax : matplotlib Axes
    """
    T = np.asarray(T, dtype=float)

    if ax is None:
        fig, ax = plt.subplots()

    im = ax.imshow(T, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="P(i → j)")

    ax.set_title(title)
    ax.set_xlabel("Destination state")
    ax.set_ylabel("Origin state")
    ax.grid(False)

    return ax


# ============================================================
# 4. Entropía temporal H(t)
# ============================================================


def plot_entropy(
    H: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Entropy Over Time",
):
    """
    Grafica la entropía H(t) en función del tiempo.

    Parameters
    ----------
    H : np.ndarray
        Vector con entropías por paso temporal.
    ax : matplotlib Axes, optional
        Ejes donde dibujar.
    title : str

    Returns
    -------
    ax : matplotlib Axes
    """
    H = np.asarray(H, dtype=float)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(np.arange(len(H)), H, marker="o", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Entropy H(t)")
    ax.grid(alpha=0.3)

    return ax
