"""Test-time augmentation views. Owner: A.

Operate on normalized tensors (B, C, H, W); output shape is always unchanged.
View names match config.tta.views: hflip, shift_pos2, shift_neg2.
The shop wrapper reuses these at 224 px, so do not hardcode 32.
"""
from __future__ import annotations


def hflip(x):
    """Horizontal flip. hflip(hflip(x)) == x. TODO(A)."""
    raise NotImplementedError("TODO(A): hflip")


def shift(x, px: int):
    """Horizontal shift by `px` columns (positive = right) with reflect padding; same output size.

    shift(shift(x, 2), -2) must equal x on interior columns. TODO(A).
    """
    raise NotImplementedError("TODO(A): shift")


def apply_view(x, name: str):
    """Dispatch a config.tta.views name ("hflip" | "shift_pos2" | "shift_neg2") to the op above. TODO(A)."""
    raise NotImplementedError("TODO(A): apply_view")
