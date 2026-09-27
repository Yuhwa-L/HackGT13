"""The shop benchmark: which photos, which split, which versions. No torch or XGBoost, so both pipeline stages
(shop.pipeline_model and shop.run_pipeline) and the cache builder can import it."""
import hashlib

import numpy as np
import pandas as pd

from shop.classes import product_indices
from shop.config import (BANK_VARIANT, BENCHMARK_VARIANTS, CORRUPTIONS, PRODUCT_CLASSES, RAW, SEED, SEVERITIES,
                         SPLITS)

N = len(PRODUCT_CLASSES)
VERSIONS = [("clean", 0)] + [(c, s) for c in CORRUPTIONS for s in SEVERITIES]
BANK_MIN = 3  # bank photos per class, at least
ARRAY_NAMES = ["logits", "tta_logits", "embeddings", "bank_embeddings"]  # model-stage outputs, .npy


def list_photos():
    """Benchmark photos and the disjoint reference bank, deduplicated by content (ImageNetV2 variants overlap)."""
    cls_of = {str(i): k for k, i in enumerate(product_indices())}
    rows, seen = [], set()
    for variant in [*BENCHMARK_VARIANTS, BANK_VARIANT]:  # benchmark first: a shared photo stays a benchmark photo
        for p in sorted((RAW / variant).glob("*/*")):
            h = hashlib.md5(p.read_bytes()).hexdigest()
            if h not in seen:
                seen.add(h)
                rows.append(dict(path=str(p), md5=h, label=cls_of[p.parent.name],
                                 role="bank" if variant == BANK_VARIANT else "benchmark"))
    df = pd.DataFrame(rows)
    # Every class needs bank photos for kNN / Mahalanobis. Top up thin classes from the benchmark (disjoint by construction).
    for c in range(N):
        short = BANK_MIN - (df[(df.role == "bank") & (df.label == c)]).shape[0]
        if short > 0:
            df.loc[df[(df.role == "benchmark") & (df.label == c)].sort_values("md5").index[:short], "role"] = "bank"
    return df



def assign_splits(labels, rng):
    """Same scheme as benchmark/make_splits.py: evenly spaced random keys per class, cut at the split fractions."""
    key = np.empty(len(labels))
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        key[idx] = (np.arange(len(idx)) + rng.random()) / len(idx)
    order = np.argsort(key, kind="stable")
    cuts = np.round(np.cumsum(list(SPLITS.values())) * len(labels)).astype(int)
    split = np.empty(len(labels), dtype=object)
    for name, lo, hi in zip(SPLITS, np.r_[0, cuts[:-1]], cuts):
        split[order[lo:hi]] = name
    return split



def build_manifest(photos):
    bench = photos[photos.role == "benchmark"].reset_index(drop=True)
    split = assign_splits(bench.label.to_numpy(), np.random.default_rng(SEED))
    rows = []
    for (_, p), sp in zip(bench.iterrows(), split):
        base = f"prod_{p.md5[:10]}"
        for corr, sev in VERSIONS:
            rows.append(dict(sample_id=f"{base}_{corr}_s{sev}", base_image_id=base, path=p.path,
                             dataset="imagenet_products" if corr == "clean" else "imagenet_products_c",
                             true_label=p.label, true_class=PRODUCT_CLASSES[p.label], corruption=corr,
                             family=CORRUPTIONS.get(corr, "clean"), severity=sev, split=sp))
    m = pd.DataFrame(rows)
    assert m.sample_id.is_unique and (m.groupby("base_image_id").split.nunique() == 1).all()
    return m
