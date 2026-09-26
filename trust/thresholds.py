"""TRUST / CAUTION / REJECT thresholds, fit on the cal split ONLY. Owner: C.

Pipeline step 6 (CLI) -> data/thresholds.json (schema: common.schemas.Thresholds).
"""
from __future__ import annotations

from typing import Optional

from common import cli

STEP, OWNER = "thresholds", "C"


def loosest_threshold(p, correct, target_error: float) -> Optional[float]:
    """Smallest tau such that error among {p >= tau} <= target_error, with >= 1 accepted sample.
    Returns None if unattainable. TODO(C).
    """
    raise NotImplementedError("TODO(C): loosest_threshold")


def decide(p: float, tau_reject: float, tau_trust: float) -> str:
    """p >= tau_trust -> "trust"; p >= tau_reject -> "caution"; else "reject". TODO(C)."""
    raise NotImplementedError("TODO(C): decide")


def broken_promise(raw_conf_clean_cal, correct_clean_cal, raw_conf_heldout, correct_heldout, target_error: float) -> dict:
    """Build the same loosest_threshold rule on raw_confidence using clean cal rows, then report its
    actual error on the held-out family test rows. Returns schemas.BrokenPromise fields. TODO(C).
    """
    raise NotImplementedError("TODO(C): broken_promise")


def main() -> None:
    args, cfg = cli.parse(cli.base_parser(__doc__))
    # TODO(C): load features + fold's calibrated model, compute p_correct on cal split,
    # tau_reject = loosest_threshold(..., reject_target_error), tau_trust = ... trust_target_error,
    # write via io.write_json(..., io.data_dir() / io.THRESHOLDS).
    cli.not_implemented(STEP, OWNER)


if __name__ == "__main__":
    main()
