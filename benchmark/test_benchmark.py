"""Owner: B. Self-checks for the loaders, load_images and the manifest. Needs data/raw/ (REQUIREMENTS.md §2.3);
tests whose dataset is missing print a skip instead of failing.
Run from the repo root: python -m benchmark.test_benchmark (pytest also works).
"""
import pickle
import sys

import numpy as np
import pandas as pd

from benchmark.load_cifar import C10_DIR, RAW, class_names, load_cifar10_1, load_cifar10_test, load_cifar10_train
from benchmark.load_cifar10c import C10C_DIR, CORRUPTIONS, load_images
from benchmark.make_splits import SPLITS, build_manifest, check_manifest

HAVE_C10 = (C10_DIR / "test_batch").exists() and (RAW / "cifar10.1_v6_data.npy").exists()
HAVE_C10C = all((C10C_DIR / f"{c}.npy").exists() for c in [*CORRUPTIONS, "labels"])


class Skip(Exception):
    pass


def _skip(name, what):
    msg = f"{name}: {what} not in data/raw"
    if "pytest" in sys.modules:
        sys.modules["pytest"].skip(msg)
    raise Skip(msg)


def test_loaders_match_raw_decode():
    if not HAVE_C10:
        _skip("test_loaders_match_raw_decode", "CIFAR-10 / CIFAR-10.1")
    with open(C10_DIR / "test_batch", "rb") as f:
        d = pickle.load(f, encoding="bytes")
    x, y = load_cifar10_test()
    assert x.shape == (10000, 32, 32, 3) and x.dtype == np.uint8 and y.shape == (10000,)
    raw = d[b"data"][123]  # 1024 R, then 1024 G, then 1024 B, each row-major
    assert (x[123, 5, 7] == [raw[5 * 32 + 7], raw[1024 + 5 * 32 + 7], raw[2048 + 5 * 32 + 7]]).all(), "pixel layout"
    assert (y == np.array(d[b"labels"])).all()
    assert len(class_names()) == 10 and class_names()[3] == "cat"
    xt, yt = load_cifar10_train()
    assert xt.shape == (50000, 32, 32, 3) and np.bincount(yt).tolist() == [5000] * 10
    x1, y1 = load_cifar10_1()
    assert x1.shape == (2000, 32, 32, 3) and x1.dtype == np.uint8 and np.bincount(y1).tolist() == [200] * 10
    try:
        x[0, 0, 0, 0] = 1
    except ValueError:
        pass
    else:
        raise AssertionError("cached loader arrays must be read-only")


def test_manifest_sizes_and_checks():
    if not HAVE_C10:
        _skip("test_manifest_sizes_and_checks", "CIFAR-10 / CIFAR-10.1")
    for n in (10, 40, 50, 51, 2000, 10000):
        m = build_manifest(n)
        check_manifest(m, n)  # its own asserts: leakage, row counts, stratification, labels
        n_c101 = (m.dataset == "cifar10_1").sum()
        assert len(m) == 41 * n + n_c101
        assert n_c101 == (2000 if n > 50 else max(10, n // 5))


def test_manifest_deterministic():
    if not HAVE_C10:
        _skip("test_manifest_deterministic", "CIFAR-10 / CIFAR-10.1")
    a = build_manifest(2000)
    assert a.equals(build_manifest(2000)), "same seed must give the same manifest"
    assert not a.equals(build_manifest(2000, seed=1)), "seed is ignored"


def test_full_manifest_no_leakage_exact_stratification():
    if not HAVE_C10:
        _skip("test_full_manifest_no_leakage_exact_stratification", "CIFAR-10 / CIFAR-10.1")
    m = build_manifest(10000)
    assert m.groupby("base_image_id").split.agg(set).map(len).eq(1).all(), "a base image is in two splits"
    clean = m[m.dataset == "cifar10_test"].set_index("base_image_id")
    cc = m[m.dataset == "cifar10c"]
    assert (cc.split.to_numpy() == clean.loc[cc.base_image_id, "split"].to_numpy()).all()
    assert (cc.true_label.to_numpy() == clean.loc[cc.base_image_id, "true_label"].to_numpy()).all()
    per_class = pd.crosstab(clean.true_label, clean.split)
    for name, frac in SPLITS.items():
        assert (per_class[name] == frac * 1000).all(), f"{name}: not exactly {frac:.0%} of every class"
    assert (m[m.dataset == "cifar10_1"].split == "test").all()


def test_sample_id_format():
    if not HAVE_C10:
        _skip("test_sample_id_format", "CIFAR-10 / CIFAR-10.1")
    m = build_manifest(2000)
    assert m.sample_id.str.fullmatch(
        r"c10_clean_\d{5}_s0|c10c_[a-z_]+_\d{5}_s[1-5]|c101_natural_\d{5}_s0").all()
    assert (m.sample_id.str.extract(r"_(\d{5})_s")[0].to_numpy() == m.base_image_id.str[-5:].to_numpy()).all()
    cc = m[m.dataset == "cifar10c"]
    assert (cc.sample_id == "c10c_" + cc.corruption + "_" + cc.base_image_id.str[4:] + "_s"
            + cc.severity.astype(str)).all()


def test_load_images_clean_and_cifar10_1():
    if not HAVE_C10:
        _skip("test_load_images_clean_and_cifar10_1", "CIFAR-10 / CIFAR-10.1")
    m = build_manifest(2000)
    sub = m[m.dataset != "cifar10c"].sample(500, random_state=0)
    x, x1 = load_cifar10_test()[0], load_cifar10_1()[0]
    for (_, r), im in zip(sub.iterrows(), load_images(sub)):
        assert (im == (x if r.dataset == "cifar10_test" else x1)[r.image_index]).all()
    assert load_images(sub.iloc[:0]).shape == (0, 32, 32, 3)


def test_load_images_cifar10c_shuffled():
    if not (HAVE_C10 and HAVE_C10C):
        _skip("test_load_images_cifar10c_shuffled", "CIFAR-10-C")
    m = build_manifest(10000)
    check_manifest(m, 10000)  # now also checks CIFAR-10-C labels.npy for all 400k corrupted rows
    shuf = m.sample(3000, random_state=0)  # all three datasets mixed, in random order
    x, x1 = load_cifar10_test()[0], load_cifar10_1()[0]
    for (_, r), im in zip(shuf.iterrows(), load_images(shuf)):
        if r.dataset == "cifar10c":
            ref = np.load(C10C_DIR / f"{r.corruption}.npy", mmap_mode="r")[r.image_index]
        else:
            ref = (x if r.dataset == "cifar10_test" else x1)[r.image_index]
        assert (im == ref).all()


def test_severity_moves_away_from_clean():
    if not (HAVE_C10 and HAVE_C10C):
        _skip("test_severity_moves_away_from_clean", "CIFAR-10-C")
    m = build_manifest(2000)
    clean = m[m.dataset == "cifar10_test"].iloc[:200].sort_values("base_image_id")
    ref = load_images(clean).astype(float)
    for c in CORRUPTIONS:
        mse = []
        for s in range(1, 6):
            rows = m[(m.corruption == c) & (m.severity == s) & m.base_image_id.isin(clean.base_image_id)]
            mse.append(((load_images(rows.sort_values("base_image_id")).astype(float) - ref) ** 2).mean())
        # Not strictly monotone: CIFAR-10-C motion_blur s4 sits slightly closer to clean than s3.
        assert mse[0] < mse[2] < mse[4], f"{c}: {np.round(mse)}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("ok", name)
            except Skip as e:
                print("skip", e)
