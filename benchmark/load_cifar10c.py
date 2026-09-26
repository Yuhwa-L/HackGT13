"""Owner: B. Load the 8 MVP CIFAR-10-C types from data/raw/CIFAR-10-C/.
Each .npy is (50000, 32, 32, 3) uint8: row i = CIFAR-10 test image i % 10000 at severity i // 10000 + 1.
Also load_images(manifest): the one place that turns manifest rows into pixels (used by A).
"""
from functools import lru_cache

import numpy as np

from benchmark.load_cifar import RAW, load_cifar10_1, load_cifar10_test

C10C_DIR = RAW / "CIFAR-10-C"
N_TEST = 10000
SEVERITIES = (1, 2, 3, 4, 5)
CORRUPTIONS = {  # corruption -> family (2 per family)
    "gaussian_noise": "noise", "impulse_noise": "noise",
    "defocus_blur": "blur", "motion_blur": "blur",
    "fog": "weather", "brightness": "weather",
    "contrast": "digital", "jpeg_compression": "digital",
}


def c10c_index(test_index, severity):
    """Row in a CIFAR-10-C .npy for CIFAR-10 test image `test_index` at `severity` (1-5)."""
    return (np.asarray(severity) - 1) * N_TEST + np.asarray(test_index)


@lru_cache(maxsize=None)
def load_cifar10c(corruption):
    """(50000, 32, 32, 3) uint8, memory-mapped read-only so the 8 files are not all held in RAM."""
    if corruption not in CORRUPTIONS:
        raise ValueError(f"unknown corruption {corruption!r}; MVP types: {list(CORRUPTIONS)}")
    return np.load(C10C_DIR / f"{corruption}.npy", mmap_mode="r")


@lru_cache(maxsize=None)
def load_cifar10c_labels():
    return np.load(C10C_DIR / "labels.npy").astype(np.int64)


def _source(dataset, corruption):
    if dataset == "cifar10_test":
        return load_cifar10_test()[0]
    if dataset == "cifar10c":
        return load_cifar10c(corruption)
    if dataset == "cifar10_1":
        return load_cifar10_1()[0]
    raise ValueError(f"unknown dataset {dataset!r}")


def load_images(manifest):
    """Pixels for manifest rows, in the same order: (len(manifest), 32, 32, 3) uint8.

    Needs only the `dataset`, `corruption` and `image_index` columns, so any slice of manifest.csv works.
    """
    out = np.empty((len(manifest), 32, 32, 3), dtype=np.uint8)
    image_index = manifest["image_index"].to_numpy()
    for (dataset, corruption), pos in manifest.groupby(["dataset", "corruption"], sort=False).indices.items():
        rows = image_index[pos]
        order = np.argsort(rows, kind="stable")  # sorted reads are much faster on a memmap
        out[pos[order]] = _source(dataset, corruption)[rows[order]]
    return out
