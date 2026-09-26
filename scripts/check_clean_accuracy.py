"""Fail loudly if clean CIFAR-10 test accuracy < config.model.expected_clean_acc_min. Owner: A.

Needs CIFAR-10 in data/raw/ (benchmark/download.sh) and downloads the checkpoint on first run.
"""
from __future__ import annotations

from common import cli

STEP, OWNER = "check_clean_accuracy", "A"


def main() -> None:
    args, cfg = cli.parse(cli.base_parser(__doc__))
    # TODO(A): load_model(get_device()), run clean CIFAR-10 test, compute accuracy,
    # raise SystemExit(f"...") if acc < cfg.model.expected_clean_acc_min.
    cli.not_implemented(STEP, OWNER)


if __name__ == "__main__":
    main()
