"""Base-image split assignment + leave-one-family-out folds. Owner: B.

The split unit is base_image_id: every corrupted version of an image shares its base image's split.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def assign_splits(
    base_ids: Sequence[str],
    labels: Sequence[int],
    fractions: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """Stratified-by-class, deterministic assignment base_image_id -> split name.

    Fractions from config.splits.fractions. CIFAR-10.1 base ids ("c101_*") are forced to "test"
    by the caller (make_manifest), not here.
    TODO(B): implement.
    """
    raise NotImplementedError("TODO(B): assign_splits")


def lofo_folds(df: pd.DataFrame, families: Sequence[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """For each held-out family F: (train_mask, eval_mask) over df rows.

    train_mask = split == "train" AND family in {"clean"} U (families - {F})
    eval_mask  = split == "test"  AND family == F
    Clean test rows and cifar10_1 are reported separately (not in any eval_mask).
    TODO(B): implement.
    """
    raise NotImplementedError("TODO(B): lofo_folds")
