import numpy as np

from simulations import simulate_markov_chain
from visualizations.plots import (
    plot_entropy,
    plot_trajectory,
    plot_transition_matrix,
)
import matplotlib.pyplot as plt


def main():
    # 1. Matriz de transición (2 estados)
    T = np.array([
        [0.9, 0.1],  # desde estado 0
        [0.2, 0.8],  # desde estado 1
    ])

    # 2. Estado inicial (todo en estado 0)
    p0 = np.array([1.0, 0.0])

    # 3. Simulamos 20 pasos
    traj, H = simulate_markov_chain(p0, T, n_steps=20)

    # 4. Visualizamos
    plot_transition_matrix(T)
    plot_trajectory(traj)
    plot_entropy(H)

    # Mostrar las figuras (necesario cuando se ejecuta como script)
    try:
        plt.show()
    except Exception:
        # Si no hay backend gráfico disponible, no hacemos nada
        pass


if __name__ == "__main__":
    main()
