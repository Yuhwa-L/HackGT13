"""Mock artifact generators (plan.md Section 8). Entry point for `make mock`.

Rules for every generator:
  - every JSON artifact has "is_mock": true; every table has an is_mock column set to True
  - values are plausible-looking but obviously synthetic; NEVER real CIFAR images
  - runs in seconds, no downloads
  - writes through common/io.py (so outputs are validated against common/schemas.py)

Shape targets (so the charts have the right *shape*):
  - ~40 base images x (clean + 8 corruption types x 5 severities), valid split invariants
  - accuracy declines with severity; raw confidence declines much less; p_correct tracks accuracy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import schemas
from common.config import Config, load_config

N_MOCK_BASE_IMAGES = 40
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def mock_manifest(cfg: Config, rng: np.random.Generator, n_base: int = N_MOCK_BASE_IMAGES) -> pd.DataFrame:
    """Rows for clean + every MVP corruption x severity, plus a few cifar10_1 rows (split=test).

    TODO(D): implement. Must pass schemas.check_manifest_invariants.
    """
    raise NotImplementedError("TODO(D): mock_manifest")


def mock_prediction_runs(manifest: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """PREDICTION_RUNS_COLUMNS; accuracy falls with severity faster than raw_confidence. TODO(D)."""
    raise NotImplementedError("TODO(D): mock_prediction_runs")


def mock_arrays(manifest: pd.DataFrame, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """embeddings (N,512), embeddings_ids (N,), tta_logits (N,3,10), small reference bank + labels. TODO(D)."""
    raise NotImplementedError("TODO(D): mock_arrays")


def mock_features(runs: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """FEATURES_COLUMNS derived loosely from runs. TODO(D)."""
    raise NotImplementedError("TODO(D): mock_features")


def mock_thresholds(cfg: Config) -> schemas.Thresholds:
    """TODO(D)."""
    raise NotImplementedError("TODO(D): mock_thresholds")


def mock_evaluation(cfg: Config, rng: np.random.Generator) -> schemas.Evaluation:
    """Every family fold x every method in schemas.METHODS, plus natural_shift.cifar10_1. TODO(D)."""
    raise NotImplementedError("TODO(D): mock_evaluation")


def placeholder_png(text: str, path: Path, rng: np.random.Generator, size: int = 256) -> None:
    """Colored noise with `text` (class name) drawn on it, via PIL. TODO(D)."""
    raise NotImplementedError("TODO(D): placeholder_png")


def mock_demo_cache(manifest: pd.DataFrame, cfg: Config, rng: np.random.Generator) -> schemas.DemoCache:
    """PredictResponse per (base_image_id, corruption, severity) + placeholder PNGs in data/demo_images/. TODO(D)."""
    raise NotImplementedError("TODO(D): mock_demo_cache")


def write_all_core(cfg: Config, out_dir: Path) -> list[Path]:
    """Generate and write every core artifact (Section 5) into out_dir. TODO(D): wire the above together."""
    raise NotImplementedError("TODO(D): write_all_core")


def write_all_shop(out_dir: Path) -> list[Path]:
    """Mock shop artifacts (Section 10.6): manifest, shop_cache, catalog, placeholder images.

    Must only be called if shop/ exists; import shop helpers lazily so core works without shop/.
    TODO(A): implement (can reuse the core generators with dataset="imagenet_products").
    """
    raise NotImplementedError("TODO(A): write_all_shop")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--core-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    data = cfg.resolve(cfg.paths.data)
    written = write_all_core(cfg, data)
    if not args.core_only and (Path(__file__).resolve().parent.parent / "shop").is_dir():
        written += write_all_shop(data / "shop")
    for p in written:
        print(f"[mock] wrote {p}")


if __name__ == "__main__":
    main()
