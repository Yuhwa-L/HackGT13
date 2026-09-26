import numpy as np
import pytest

from trust import calibrate, temperature_scaling


def test_overconfident_logits_get_T_gt_1():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, 2000)
    logits = rng.normal(size=(2000, 10))
    logits[np.arange(2000), labels] += 1.0
    logits *= 5.0  # overconfident
    T = temperature_scaling.fit_temperature(logits, labels)
    assert T > 1.0


def test_isotonic_output_in_unit_interval():
    rng = np.random.default_rng(0)
    s = rng.uniform(size=500)
    c = (rng.uniform(size=500) < s).astype(int)
    iso = calibrate.fit_isotonic(s, c)
    p = calibrate.apply(iso, np.array([-1.0, 0.5, 2.0]))
    assert ((p >= 0) & (p <= 1)).all()
