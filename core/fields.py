from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Tuple

@dataclass
class DiscreteField2D:
    """
    Representa un campo discreto 2D simple.
    - grid: ndarray de forma (H, W) con valores float.
    """
    grid: np.ndarray

    @classmethod
    def from_shape(cls, shape: Tuple[int, int], init_value: float = 0.0) -> "DiscreteField2D":
        """
        Crea un campo 2D con la forma dada y un valor inicial.
        """
        grid = np.full(shape, fill_value=init_value, dtype=float)
        return cls(grid=grid)

    def copy(self) -> "DiscreteField2D":
        """
        Devuelve una copia independiente del campo.
        """
        return DiscreteField2D(grid=np.copy(self.grid))