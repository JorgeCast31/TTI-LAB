import numpy as np
from simulations.dynamics import simulate_markov_chain
from core.operators import apply_transition

def test_apply_transition_simple():
    p = np.array([1.0, 0.0])
    T = np.array([[0.9, 0.2], [0.1, 0.8]])
    p2 = apply_transition(p, T)
    assert p2.shape == p.shape

def test_simulate_markov_chain():
    p0 = np.array([1.0, 0.0])
    T = np.array([[0.9, 0.2], [0.1, 0.8]])
    traj = simulate_markov_chain(p0, T, steps=5)
    assert traj.shape == (6, 2)
