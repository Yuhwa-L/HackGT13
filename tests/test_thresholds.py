import numpy as np

from trust.thresholds import decide, loosest_threshold


def test_decide():
    assert decide(0.99, 0.5, 0.9) == "trust"
    assert decide(0.7, 0.5, 0.9) == "caution"
    assert decide(0.1, 0.5, 0.9) == "reject"


def test_loosest_threshold():
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    c = np.array([0, 0, 1, 1, 1])
    assert loosest_threshold(p, c, 0.0) == 0.3
    assert loosest_threshold(np.array([0.5]), np.array([0]), 0.0) is None
