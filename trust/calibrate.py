"""Owner: C. Isotonic calibration on val -> p_correct. Saved as data/models/isotonic_<fold>.json {"x": [...], "y": [...]};
apply with np.interp (identical to sklearn's IsotonicRegression.predict with out_of_bounds="clip").
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_isotonic(score, correct, sample_weight=None):
    """Monotone map from the failure model's uncalibrated P(correct) to a calibrated p_correct, as JSON-ready lists."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(score, correct, sample_weight=sample_weight)
    return {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}


def apply_isotonic(cal, score):
    return np.interp(score, cal["x"], cal["y"])
