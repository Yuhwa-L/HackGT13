"""Pipeline step 5: per-fold LR baseline + XGBoost failure model -> data/models/. Owner: C.

- Train on each LOFO fold's train_mask (benchmark/make_splits.lofo_folds), features = MODEL_FEATURES.
- sample_weight = config.trust.clean_row_weight for clean rows, 1.0 otherwise.
- XGBoost predicts `failure`; p_correct = isotonic(1 - P(failure)) fit on the val split.
- Save per fold: data/models/<fold>/{logreg.joblib, xgb.ubj, isotonic.joblib}
"""
from __future__ import annotations

from common import cli

STEP, OWNER = "train_failure_model", "C"


def train_fold(features_df, fold: str, cfg):
    """TODO(C): hyperparameters, early stopping on val, save models for this fold."""
    raise NotImplementedError


def main() -> None:
    args, cfg = cli.parse(cli.base_parser(__doc__))
    cli.not_implemented(STEP, OWNER)


if __name__ == "__main__":
    main()
