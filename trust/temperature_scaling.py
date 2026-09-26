"""Temperature scaling baseline. Owner: C.

Two baselines use this: temp_scaled_clean (fit on clean val) and temp_scaled_corrupted (fit on
corrupted val).
"""
from __future__ import annotations


def fit_temperature(logits, labels, max_iter: int = 100) -> float:
    """LBFGS on NLL over log T (keeps T > 0). Returns T. TODO(C)."""
    raise NotImplementedError("TODO(C): fit_temperature")


def apply(logits, T: float):
    """Softmax(logits / T) -> (N,C) probs. TODO(C)."""
    raise NotImplementedError("TODO(C): apply")
