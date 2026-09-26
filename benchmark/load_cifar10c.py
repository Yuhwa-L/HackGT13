"""CIFAR-10-C loader + index math. Owner: B.

Each <type>.npy is (50000, 32, 32, 3) uint8: 5 severities stacked, 10000 test images each.
Row i -> base image i % 10000, severity i // 10000 + 1.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

N_BASE = 10_000


def row_to_base_and_severity(i: int) -> tuple[int, int]:
    """Row index in a CIFAR-10-C array -> (base image index, severity in 1..5). TODO(B)."""
    raise NotImplementedError("TODO(B): row_to_base_and_severity")


def base_and_severity_to_row(base: int, severity: int) -> int:
    """Inverse of row_to_base_and_severity. TODO(B)."""
    raise NotImplementedError("TODO(B): base_and_severity_to_row")


def iter_cifar10c(
    raw_dir: Path,
    corruption: str,
    base_ids: Iterable[int],
    severities: Iterable[int],
) -> Iterator[tuple[int, int, np.ndarray, int]]:
    """Yield (base_index, severity, image_uint8 (32,32,3), label) for requested rows only.

    Read <raw_dir>/CIFAR-10-C/<corruption>.npy and labels.npy with np.load(mmap_mode="r").
    TODO(B): implement.
    """
    raise NotImplementedError("TODO(B): iter_cifar10c")
