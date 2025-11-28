"""Visualization utilities for TTI Lab.

This package exposes plotting helpers implemented in `visualizations.plots`.
Avoid importing the functions at package-import time to prevent import-time
cycles; users can still import the helpers from `visualizations.plots` or
access them via the package after the submodule is imported.
"""

__all__ = [
    "plot_distribution",
    "plot_trajectory",
    "plot_transition_matrix",
    "plot_entropy",
]