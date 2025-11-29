from __future__ import annotations
import numpy as np
from typing import Callable
from ....core.fields import DiscreteField2D

def step_ca(field: DiscreteField2D, rule_fn: Callable[[np.ndarray], np.ndarray]) -> DiscreteField2D:
    """
    Ejecuta un paso de autómata celular:
    - `field`: entrada como DiscreteField2D.
    - `rule_fn`: función que recibe `grid` (np.ndarray) y devuelve `new_grid` (np.ndarray)
      con la misma forma.

    Devuelve un nuevo `DiscreteField2D` con el grid actualizado.
    """
    new_grid = rule_fn(field.grid)
    # Asegurar que el resultado sea ndarray y tenga la forma correcta
    if not isinstance(new_grid, np.ndarray):
        new_grid = np.array(new_grid)
    if new_grid.shape != field.grid.shape:
        raise ValueError("rule_fn must return an array with the same shape as the input grid")
    return DiscreteField2D(grid=np.copy(new_grid))