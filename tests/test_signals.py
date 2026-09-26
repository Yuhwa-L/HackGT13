import numpy as np
import pytest

from trust.signals import knn_distance, softmax_stats, tta_signals


def test_softmax_stats_uniform_and_peaked():
    logits = np.array([[0.0] * 10, [10.0] + [0.0] * 9])
    s = softmax_stats(logits)
    assert s["raw_confidence"][0] == pytest.approx(0.1)
    assert s["entropy"][0] == pytest.approx(np.log(10))
    assert s["raw_confidence"][1] > 0.99
    assert s["margin"][1] == pytest.approx(s["raw_confidence"][1] - s["top2_prob"][1])


def test_tta_agree():
    orig = np.array([[5.0, 0, 0]])
    tta = np.array([[[5.0, 0, 0], [0, 5.0, 0], [5.0, 0, 0]]])
    s = tta_signals(orig, tta)
    assert s["tta_agree"][0] == pytest.approx(2 / 3)


def test_knn_distance_self_is_zero():
    pytest.importorskip("torch")
    bank = np.random.default_rng(0).normal(size=(100, 16)).astype("float32")
    d = knn_distance(bank[:5], bank, k=1)
    assert np.allclose(d, 0, atol=1e-5)
