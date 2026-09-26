"""CIFAR-10 test + train (reference bank source) + CIFAR-10.1 loaders. Owner: B.

All return uint8 images (N, 32, 32, 3) and int64 labels (N,). Normalization happens in inference/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def load_cifar10(raw_dir: Path, train: bool) -> tuple[np.ndarray, np.ndarray]:
    """CIFAR-10 via torchvision (download=False; run benchmark/download.sh first).

    train=True is ONLY used to build the reference bank; train images are never trust-layer rows.
    TODO(B): implement.
    """
    raise NotImplementedError("TODO(B): load_cifar10")


def load_cifar10_1(raw_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """cifar10.1_v6_data.npy / cifar10.1_v6_labels.npy. TODO(B): implement."""
    raise NotImplementedError("TODO(B): load_cifar10_1")
