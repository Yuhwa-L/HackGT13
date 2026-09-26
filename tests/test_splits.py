import numpy as np
import pandas as pd

from benchmark.make_splits import assign_splits, lofo_folds

FRACS = {"train": 0.50, "val": 0.15, "cal": 0.15, "test": 0.20}
FAMILIES = ["noise", "blur", "weather", "digital"]


def _ids(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return [f"c10_{i:05d}" for i in range(n)], rng.integers(0, 10, n).tolist()


def test_deterministic_and_fractions():
    ids, labels = _ids()
    a = assign_splits(ids, labels, FRACS, seed=1337)
    assert a == assign_splits(ids, labels, FRACS, seed=1337)
    assert set(a) == set(ids)
    counts = pd.Series(a).value_counts(normalize=True)
    for split, frac in FRACS.items():
        assert abs(counts[split] - frac) <= 0.01


def test_lofo_heldout_never_in_train():
    ids, labels = _ids(200)
    split = assign_splits(ids, labels, FRACS, seed=0)
    rows = [{"base_image_id": b, "split": split[b], "family": fam}
            for b in ids for fam in ["clean"] + FAMILIES]
    df = pd.DataFrame(rows)
    folds = lofo_folds(df, FAMILIES)
    for fam, (train_mask, eval_mask) in folds.items():
        assert not (df[train_mask]["family"] == fam).any()
        assert (df[train_mask]["split"] == "train").all()
        assert (df[eval_mask]["family"] == fam).all()
        assert (df[eval_mask]["split"] == "test").all()
