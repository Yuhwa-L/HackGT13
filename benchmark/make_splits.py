"""Owner: B. Build data/manifest.csv, one row per evaluated image: sample_id, base_image_id, dataset, image_index,
true_label, true_class, corruption, family, severity, split.
Split by base_image_id, never by row: train 50 / val 15 / cal 15 / test 20, stratified by class, fixed seed.
CIFAR-10.1 rows: dataset=cifar10_1, corruption=natural, family=natural, split=test.

Run: python -m benchmark.make_splits [--n-base 10000]   (default 10000 = the frozen 410k-row benchmark the committed results
use; 2000 = CPU-sized, different splits; <= 50 = 1 PM tiny pipeline)
"""
import argparse

import numpy as np
import pandas as pd

from benchmark.load_cifar import RAW, class_names, load_cifar10_1, load_cifar10_test
from benchmark.load_cifar10c import C10C_DIR, CORRUPTIONS, SEVERITIES, c10c_index, load_cifar10c_labels

SEED = 0
N_BASE = 10000  # frozen benchmark size: data/evaluation.json, scores.parquet and demo_cache.json come from it
TINY_MAX = 50  # at or below this many base images, CIFAR-10.1 is subsampled too
SPLITS = {"train": 0.50, "val": 0.15, "cal": 0.15, "test": 0.20}
OUT = RAW.parent / "manifest.csv"
COLUMNS = ["sample_id", "base_image_id", "dataset", "image_index", "true_label", "true_class",
           "corruption", "family", "severity", "split"]
N_CLASSES = 10


def pick_stratified(labels, n, rng):
    """Indices of n images, as equal per class as possible (the first n % 10 classes get one extra), sorted."""
    counts = np.full(N_CLASSES, n // N_CLASSES) + (np.arange(N_CLASSES) < n % N_CLASSES)
    picked = [rng.choice(np.flatnonzero(labels == c), size=k, replace=False) for c, k in enumerate(counts)]
    return np.sort(np.concatenate(picked))


def assign_splits(labels, rng):
    """Split name per base image. Global counts are exact; each class is within two images of its share.

    Each class's images get evenly spaced keys in [0, 1) in random order (with a random per-class offset),
    then all images are sorted by key and cut at the split fractions, so the classes interleave evenly.
    """
    key = np.empty(len(labels))
    for c in np.unique(labels):   # any label set (trust.fit_any reuses this); CIFAR's 0..9 give the same draws as before
        idx = rng.permutation(np.flatnonzero(labels == c))
        key[idx] = (np.arange(len(idx)) + rng.random()) / len(idx)
    order = np.argsort(key, kind="stable")
    cuts = np.round(np.cumsum(list(SPLITS.values())) * len(labels)).astype(int)
    split = np.empty(len(labels), dtype=object)
    for name, lo, hi in zip(SPLITS, np.r_[0, cuts[:-1]], cuts):
        split[order[lo:hi]] = name
    return split


def build_manifest(n_base=N_BASE, seed=SEED):
    rng = np.random.default_rng(seed)
    names = np.array(class_names())
    _, test_labels = load_cifar10_test()
    _, c101_labels = load_cifar10_1()

    base = pick_stratified(test_labels, n_base, rng)
    labels = test_labels[base]
    split = assign_splits(labels, rng)
    base_id = np.char.mod("c10_%05d", base)

    blocks = [pd.DataFrame({
        "sample_id": np.char.mod("c10_clean_%05d_s0", base), "base_image_id": base_id, "dataset": "cifar10_test",
        "image_index": base, "true_label": labels, "corruption": "clean", "family": "clean", "severity": 0,
        "split": split})]
    for corruption, family in CORRUPTIONS.items():
        for s in SEVERITIES:
            blocks.append(pd.DataFrame({
                "sample_id": [f"c10c_{corruption}_{i:05d}_s{s}" for i in base], "base_image_id": base_id,
                "dataset": "cifar10c", "image_index": c10c_index(base, s), "true_label": labels,
                "corruption": corruption, "family": family, "severity": s, "split": split}))

    n_c101 = max(N_CLASSES, n_base // 5) if n_base <= TINY_MAX else len(c101_labels)
    c101 = pick_stratified(c101_labels, n_c101, rng) if n_c101 < len(c101_labels) else np.arange(len(c101_labels))
    blocks.append(pd.DataFrame({
        "sample_id": np.char.mod("c101_natural_%05d_s0", c101), "base_image_id": np.char.mod("c101_%05d", c101),
        "dataset": "cifar10_1", "image_index": c101, "true_label": c101_labels[c101], "corruption": "natural",
        "family": "natural", "severity": 0, "split": "test"}))

    m = pd.concat(blocks, ignore_index=True)
    m["true_class"] = names[m["true_label"].to_numpy()]
    return m[COLUMNS]


def check_manifest(m, n_base):
    assert m["sample_id"].is_unique, "duplicate sample_id"
    assert m[COLUMNS].notna().all().all(), "missing values"
    assert (m.groupby("base_image_id")["split"].nunique() == 1).all(), "a base image is in two splits"

    c10 = m[m["dataset"] != "cifar10_1"]
    per_base = c10.groupby("base_image_id").size()
    assert len(per_base) == n_base and (per_base == 1 + len(CORRUPTIONS) * len(SEVERITIES)).all(), "rows per base image"
    assert (c10.groupby("base_image_id")["true_label"].nunique() == 1).all(), "label differs within a base image"

    c101 = m[m["dataset"] == "cifar10_1"]
    assert (c101["split"] == "test").all() and (c101["family"] == "natural").all(), "CIFAR-10.1 must be natural/test"
    assert set(m.loc[m["split"] != "test", "family"]) <= {"clean", *CORRUPTIONS.values()}, "natural rows outside test"

    # Split shares per base image, overall and per class.
    bases = c10.drop_duplicates("base_image_id")
    for name, frac in SPLITS.items():
        assert abs((bases["split"] == name).sum() - frac * n_base) <= 1, f"{name} share off"
    per_class = pd.crosstab(bases["true_label"], bases["split"])
    class_size = per_class.sum(axis=1)
    for name, frac in SPLITS.items():
        assert (abs(per_class.get(name, 0) - frac * class_size) < 2).all(), f"{name} not stratified by class"

    # Pixel-level ground truth: the image_index points at the right label in each source file.
    _, test_labels = load_cifar10_test()
    clean = m[m["dataset"] == "cifar10_test"]
    assert (test_labels[clean["image_index"]] == clean["true_label"]).all(), "clean labels"
    _, c101_labels = load_cifar10_1()
    assert (c101_labels[c101["image_index"]] == c101["true_label"]).all(), "CIFAR-10.1 labels"
    cc = m[m["dataset"] == "cifar10c"]
    assert (cc["image_index"] % 10000 == cc["base_image_id"].str[4:].astype(int)).all(), "CIFAR-10-C row -> base image"
    assert (cc["image_index"] // 10000 + 1 == cc["severity"]).all(), "CIFAR-10-C row -> severity"
    if (C10C_DIR / "labels.npy").exists():
        assert (load_cifar10c_labels()[cc["image_index"]] == cc["true_label"]).all(), "CIFAR-10-C labels"
    else:
        print(f"warning: {C10C_DIR / 'labels.npy'} missing, skipped the CIFAR-10-C label check")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-base", type=int, default=N_BASE, help="CIFAR-10 test base images (max 10000)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    if not N_CLASSES <= args.n_base <= 10000:
        ap.error("--n-base must be between 10 and 10000")

    m = build_manifest(args.n_base, args.seed)
    check_manifest(m, args.n_base)
    m.to_csv(args.out, index=False)

    n_c101 = (m["dataset"] == "cifar10_1").sum()
    print(f"wrote {args.out}: {len(m):,} rows = {args.n_base:,} CIFAR-10 base images x 41 + {n_c101:,} CIFAR-10.1 (seed {args.seed})")
    print(pd.crosstab(m["family"], m["split"], margins=True).to_string())


if __name__ == "__main__":
    main()
