"""Isotonic calibration -> p_correct. Owner: C."""
from __future__ import annotations


def fit_isotonic(scores, correct):
    """sklearn IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1) fit on (scores, correct). TODO(C)."""
    raise NotImplementedError("TODO(C): fit_isotonic")


def apply(calibrator, scores):
    """-> p_correct in [0,1]. TODO(C)."""
    raise NotImplementedError("TODO(C): apply")
