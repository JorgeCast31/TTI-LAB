import numpy as np
from core.probability import normalize, is_probability_vector

def test_normalize_vector():
    v = np.array([2.0, 2.0])
    nv = normalize(v)
    assert nv.shape == v.shape
    assert abs(nv.sum() - 1.0) < 1e-8

def test_is_probability_vector():
    assert is_probability_vector(np.array([0.5, 0.5]))
    assert not is_probability_vector(np.array([1.0, -0.1]))
