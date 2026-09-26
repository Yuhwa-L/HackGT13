"""Pipeline step 7 -> data/evaluation.json (schema: common.schemas.Evaluation). Owner: C.

Every method in schemas.METHODS x every LOFO fold, plus natural_shift.cifar10_1, with bootstrap CIs
(trust/metrics.bootstrap_ci over base images). by_severity, reliability, risk_coverage, broken_promise
per fold.
"""
from __future__ import annotations

from common import cli

STEP, OWNER = "evaluate", "C"


def evaluate_fold(features_df, fold: str, cfg) -> dict:
    """Return a schemas.FoldResult-shaped dict. TODO(C)."""
    raise NotImplementedError


def main() -> None:
    args, cfg = cli.parse(cli.base_parser(__doc__))
    cli.not_implemented(STEP, OWNER)


if __name__ == "__main__":
    main()
