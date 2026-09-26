"""Evaluation metrics. Owner: C.

Convention for every function: `score` = higher means more likely correct; `correct` in {0,1}.
"""
from __future__ import annotations


def ece(conf, correct, n_bins: int = 15) -> float:
    """Expected calibration error, equal-width bins. TODO(C)."""
    raise NotImplementedError("TODO(C): ece")


def reliability_bins(conf, correct, n_bins: int = 15) -> list[dict]:
    """List of {bin_lo, bin_hi, conf, acc, count} (schemas.ReliabilityBin). TODO(C)."""
    raise NotImplementedError("TODO(C): reliability_bins")


def risk_coverage_curve(score, correct) -> list[dict]:
    """Sort by score desc; list of {coverage, risk}. TODO(C)."""
    raise NotImplementedError("TODO(C): risk_coverage_curve")


def aurc(score, correct) -> float:
    """Area under the risk-coverage curve. TODO(C)."""
    raise NotImplementedError("TODO(C): aurc")


def error_at_coverage(score, correct, coverage: float) -> float:
    """Error rate among the top `coverage` fraction by score. TODO(C)."""
    raise NotImplementedError("TODO(C): error_at_coverage")


def auroc_failure(score, correct) -> float:
    """AUROC for detecting failures: roc_auc_score(1 - correct, -score). TODO(C)."""
    raise NotImplementedError("TODO(C): auroc_failure")


def auroc_within_group(score, correct, group_keys) -> float:
    """Mean auroc_failure over groups (corruption x severity) containing both classes;
    single-class groups ignored. TODO(C).
    """
    raise NotImplementedError("TODO(C): auroc_within_group")


def brier(p, correct) -> float:
    """TODO(C)."""
    raise NotImplementedError("TODO(C): brier")


def nll(p, correct, eps: float = 1e-12) -> float:
    """Binary NLL of p as P(correct). TODO(C)."""
    raise NotImplementedError("TODO(C): nll")


def bootstrap_ci(metric_fn, df, group_col: str = "base_image_id", n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% CI by resampling BASE IMAGES (group_col), not rows. metric_fn(df_resampled) -> float. TODO(C)."""
    raise NotImplementedError("TODO(C): bootstrap_ci")
