import numpy as np
import pytest

from trust import metrics


def test_ece_perfectly_calibrated():
    conf = np.array([0.25] * 4 + [0.75] * 4)
    correct = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    assert metrics.ece(conf, correct, n_bins=2) == pytest.approx(0.0)


def test_auroc_perfect_ranking():
    score = np.array([0.9, 0.8, 0.2, 0.1])
    correct = np.array([1, 1, 0, 0])
    assert metrics.auroc_failure(score, correct) == pytest.approx(1.0)


def test_error_at_coverage():
    score = np.array([0.9, 0.8, 0.2, 0.1])
    correct = np.array([1, 1, 0, 0])
    assert metrics.error_at_coverage(score, correct, 0.5) == pytest.approx(0.0)
    assert metrics.error_at_coverage(score, correct, 1.0) == pytest.approx(0.5)


def test_within_group_ignores_single_class_groups():
    score = np.array([0.9, 0.1, 0.5, 0.4])
    correct = np.array([1, 0, 1, 1])
    groups = np.array(["a", "a", "b", "b"])
    assert metrics.auroc_within_group(score, correct, groups) == pytest.approx(1.0)


def test_brier_and_nll_perfect():
    assert metrics.brier(np.array([1.0, 0.0]), np.array([1, 0])) == pytest.approx(0.0)
    assert metrics.nll(np.array([1.0, 0.0]), np.array([1, 0])) == pytest.approx(0.0, abs=1e-6)
