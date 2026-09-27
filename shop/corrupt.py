"""The 8 MVP corruptions on 224 x 224 product photos, via the imagecorruptions package (Hendrycks & Dietterich's code).
Seeded per (photo, corruption, severity) so every version is reproducible, which the demo cache relies on.
"""
import hashlib
import warnings

import numpy as np

if not hasattr(np, "float_"):  # imagecorruptions' fog still uses np.float_, which NumPy 2 removed
    np.float_ = np.float64

import imagecorruptions.corruptions as _ic  # noqa: E402
from imagecorruptions import corrupt as _corrupt  # noqa: E402


def corrupt(img, corruption, severity, key=""):
    """img: (224, 224, 3) uint8. corruption 'clean' or severity 0 returns img unchanged."""
    if corruption == "clean" or severity == 0:
        return img
    seed = int.from_bytes(hashlib.sha256(f"{key}|{corruption}|{severity}".encode()).digest()[:4], "little")
    state = np.random.get_state()
    np.random.seed(seed)
    # impulse_noise goes through skimage's random_noise, which has its own generator and ignores np.random: seed it too
    sk_noise = _ic.sk.util.random_noise
    _ic.sk.util.random_noise = lambda *a, **k: sk_noise(*a, rng=np.random.default_rng(seed), **k)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(_corrupt(img, corruption_name=corruption, severity=int(severity)), dtype=np.uint8)
    finally:
        np.random.set_state(state)
        _ic.sk.util.random_noise = sk_noise
