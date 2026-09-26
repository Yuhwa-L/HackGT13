"""Trust gate: decision -> allowed assistant actions. SOURCE OF TRUTH, enforced in code (not by the LLM).
Pure functions, fully tested (tests/shop/test_gate.py). Owner: A.
"""
from __future__ import annotations

from typing import Optional

ALLOWED: dict[str, set[str]] = {
    "trust":   {"recommend", "add_to_cart", "checkout"},
    "caution": {"ask_disambiguation", "recommend"},     # checkout only after user confirms a class
    "reject":  {"ask_retake"},
}


def effective_decision(decision: str, confirmed_class: Optional[str], candidate_classes: list[str]) -> str:
    """CAUTION + confirmed_class among candidates -> "trust" (for that class). Otherwise unchanged.
    TODO(A): decide whether confirmed_class must be in candidates (recommended: yes).
    """
    raise NotImplementedError("TODO(A): effective_decision")


def allowed_actions(decision: str) -> set[str]:
    """TODO(A)."""
    raise NotImplementedError("TODO(A): allowed_actions")


def can_checkout(decision: str, confirmed_class: Optional[str], candidate_classes: list[str]) -> bool:
    """TODO(A)."""
    raise NotImplementedError("TODO(A): can_checkout")


def enforce(response: dict, decision: str) -> dict:
    """Post-process an assistant response so it never exceeds the gate, regardless of LLM output:
    strip products / set cart_allowed=False when forbidden, overwrite allowed_actions. TODO(A).
    """
    raise NotImplementedError("TODO(A): enforce")
