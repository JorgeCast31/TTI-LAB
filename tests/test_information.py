import pytest
import numpy as np
from core.information import entropy, kl_divergence, mutual_information


def test_entropy_kl_mutual_information_basic():
    p = np.array([0.5, 0.5])
    q = np.array([0.75, 0.25])

    # Entropy of a fair coin is 1 bit (base=2)
    H = entropy(p)
    assert pytest.approx(H, rel=1e-6) == 1.0

    # KL divergence D(p||q) should be positive
    D = kl_divergence(p, q)
    assert D > 0.0

    # Independent joint -> mutual information approx 0
    joint = np.array([[0.25, 0.25], [0.25, 0.25]])
    I_indep = mutual_information(joint)
    assert pytest.approx(I_indep, abs=1e-9) == 0.0

    # Dependent joint -> mutual information > 0
    joint_dep = np.array([[0.4, 0.1], [0.1, 0.4]])
    I_dep = mutual_information(joint_dep)
    assert I_dep > 0.0
import numpy as np
from core.information import entropy, kl_divergence, mutual_information

p = np.array([0.5, 0.5])
q = np.array([0.75, 0.25])

print("H(p) =", entropy(p))               # ~1.0 bit
print("D(p||q) =", kl_divergence(p, q))   # ~0.207 bits

joint = np.array([
    [0.25, 0.25],
    [0.25, 0.25],  # X e Y independientes
])
print("I(X;Y) indep =", mutual_information(joint))  # ~0.0

joint_dep = np.array([
    [0.4, 0.1],
    [0.1, 0.4],  # X e Y correlacionados
])
print("I(X;Y) dep =", mutual_information(joint_dep))  # > 0
