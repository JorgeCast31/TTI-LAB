"""simulations package"""
import numpy as np
from core.probability import normalize
from core.information import entropy
from core.operators import apply_transition

def simulate_markov_chain(p0, T, n_steps):
    """Simulate Markov chain and return trajectories and entropies."""
    p = normalize(p0)
    traj = [p]
    H = [entropy(p)]
    for _ in range(n_steps):
        p = apply_transition(p, T)
        traj.append(p)
        H.append(entropy(p))
    return np.vstack(traj), np.array(H)
