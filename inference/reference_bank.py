"""Pipeline step 2 -> data/reference_bank.npy (M,512) float32 + data/reference_labels.npy. Owner: A.

Embeddings of CIFAR-10 TRAIN images. Used only for kNN / Mahalanobis signals; train images are
never trust-layer rows.
"""
from __future__ import annotations

from common import cli

STEP, OWNER = "reference_bank", "A"


def build_reference_bank(cfg, device: str, limit: int | None):
    """Return (bank (M,512) float32, labels (M,) int64). TODO(A): batch size, full 50k vs subsample."""
    raise NotImplementedError


def main() -> None:
    args, cfg = cli.parse(cli.base_parser(__doc__))
    if args.mock:
        print(f"[{STEP}] --mock: reference bank comes from `make mock`; nothing to do.")
        return
    cli.not_implemented(STEP, OWNER)


if __name__ == "__main__":
    main()
