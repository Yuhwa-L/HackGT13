"""Owner: B. Load CIFAR-10 (test set = base images; train set = embedding reference bank only) and CIFAR-10.1 v6 from data/raw/."""
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
C10_DIR = RAW / "cifar-10-batches-py"


def _frozen(*arrays):
    # Loaders are cached, so callers share one array: make it read-only instead of copying.
    for a in arrays:
        a.flags.writeable = False
    return arrays


def _load_batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    images = np.ascontiguousarray(d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
    return images, np.asarray(d[b"labels"], dtype=np.int64)


@lru_cache(maxsize=None)
def class_names():
    with open(C10_DIR / "batches.meta", "rb") as f:
        return tuple(n.decode() for n in pickle.load(f, encoding="bytes")[b"label_names"])


@lru_cache(maxsize=None)
def load_cifar10_test():
    """(10000, 32, 32, 3) uint8 images, (10000,) int64 labels."""
    return _frozen(*_load_batch(C10_DIR / "test_batch"))


@lru_cache(maxsize=None)
def load_cifar10_train():
    """(50000, 32, 32, 3) uint8 images, (50000,) int64 labels. Reference bank only, never trust-layer rows."""
    batches = [_load_batch(C10_DIR / f"data_batch_{k}") for k in range(1, 6)]
    return _frozen(np.concatenate([b[0] for b in batches]), np.concatenate([b[1] for b in batches]))


@lru_cache(maxsize=None)
def load_cifar10_1():
    """CIFAR-10.1 v6: (2000, 32, 32, 3) uint8 images, (2000,) int64 labels."""
    images = np.load(RAW / "cifar10.1_v6_data.npy")
    labels = np.load(RAW / "cifar10.1_v6_labels.npy").astype(np.int64)
    return _frozen(images, labels)
