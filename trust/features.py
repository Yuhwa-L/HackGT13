"""Pipeline step 4 -> data/features.csv (schema: common.schemas.FEATURES_COLUMNS). Owner: B.

Joins prediction_runs + tta_logits + embeddings/reference bank signals.
"""
from __future__ import annotations

from common import cli

STEP, OWNER = "features", "B"

# What the failure model may train on. corruption/family/severity/split/base_image_id are
# ANALYSIS-ONLY and must never appear here (enforced by tests/test_features_contract.py).
# TODO(B): add trust_score if implemented.
MODEL_FEATURES: list[str] = [
    "raw_confidence",
    "entropy",
    "margin",
    "tta_agree",
    "tta_pconf",
    "tta_std",
    "knn_dist",
    "maha_pred",
]


def build_features(cfg, limit: int | None):
    """Return a DataFrame matching FEATURES_COLUMNS; failure = 1 - correct. TODO(B)."""
    raise NotImplementedError


def main() -> None:
    args, cfg = cli.parse(cli.base_parser(__doc__))
    try:
        df = build_features(cfg, args.limit)
    except NotImplementedError:
        cli.not_implemented(STEP, OWNER)
    from common import io

    io.write_table(df, io.data_dir() / io.FEATURES)


if __name__ == "__main__":
    main()
