"""Owner: B. Self-checks for trust/signals.py and trust/features.py against scipy / sklearn references and hand-worked
cases. The end-to-end tests build mock A outputs (trust/mock_runs.py), which needs CIFAR-10 in data/raw/.
Run from the repo root: python -m trust.test_features (pytest also works).
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.special import softmax as sp_softmax
from scipy.stats import entropy as sp_entropy
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from benchmark.load_cifar import C10_DIR, RAW
from trust import signals
from trust.features import COLUMNS, auroc, build_features, load_inputs
from trust.mock_runs import write_mock

HAVE_C10 = (C10_DIR / "test_batch").exists() and (RAW / "cifar10.1_v6_data.npy").exists()


class Skip(Exception):
    pass


def _need_cifar(name):
    if HAVE_C10:
        return
    msg = f"{name}: CIFAR-10 / CIFAR-10.1 not in data/raw (needed to build the mock manifest)"
    if "pytest" in sys.modules:
        sys.modules["pytest"].skip(msg)
    raise Skip(msg)


def test_softmax_stats_match_scipy():
    rng = np.random.default_rng(0)
    L = rng.normal(size=(500, 10)) * 5
    L[0] = [1000, 999, 0, 0, 0, 0, 0, 0, 0, 0]  # would overflow a naive exp
    st = signals.softmax_stats(L)
    P = sp_softmax(L, axis=1)
    assert np.allclose(st["raw_confidence"], P.max(1)) and np.allclose(st["entropy"], sp_entropy(P, axis=1))
    Ps = np.sort(P, 1)
    assert np.allclose(st["margin"], Ps[:, -1] - Ps[:, -2])
    assert (st["pred"] == L.argmax(1)).all() and np.isfinite(st["entropy"]).all()
    assert np.isclose(st["raw_confidence"][0], 1 / (1 + np.exp(-1)))


def test_tta_stats_by_hand():
    logits = np.array([[2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    views = np.zeros((1, 3, 10))
    views[0, 0, 0], views[0, 1, 0], views[0, 2, 3] = 2, 1, 5  # views 1-2 keep class 0, view 3 flips to class 3
    ts = signals.tta_stats(logits, views)
    pk = [sp_softmax(logits[0])[0]] + [sp_softmax(v)[0] for v in views[0]]
    assert np.isclose(ts["tta_agree"][0], 2 / 3)
    assert np.isclose(ts["tta_pconf"][0], np.mean(pk[1:]))  # views only
    assert np.isclose(ts["tta_std"][0], np.std(pk))         # original + views
    L = np.random.default_rng(1).normal(size=(50, 10))
    same = signals.tta_stats(L, np.repeat(L[:, None], 3, 1))  # views identical to the original
    assert (same["tta_agree"] == 1).all() and np.allclose(same["tta_pconf"], sp_softmax(L, 1).max(1))
    assert np.allclose(same["tta_std"], 0)


def test_knn_dist_matches_sklearn():
    rng = np.random.default_rng(2)
    q, bank = rng.normal(size=(1234, 64)), rng.normal(size=(3000, 64))
    ref = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="brute").fit(bank).kneighbors(q)[0][:, -1]
    assert np.allclose(signals.knn_dist(q, bank, chunk=100), ref, atol=1e-5)  # chunk does not divide N
    assert np.allclose(signals.knn_dist(bank[:50], bank, k=1), 0, atol=1e-5)


def test_trust_score_matches_brute_force():
    rng = np.random.default_rng(3)
    bank = rng.normal(size=(2000, 32))
    bl = rng.permutation(np.repeat(np.arange(10), 200))  # unsorted labels
    q = np.vstack([rng.normal(size=(300, 32)), bank[:3]])
    pr = rng.integers(0, 10, len(q))
    pr[-3:] = bl[:3]  # the last 3 queries are bank points, predicted as their own class
    got = signals.bank_signals(q, pr, bank, bl, chunk=64)

    def unit(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    D = np.linalg.norm(unit(q)[:, None] - unit(bank)[None], axis=2)
    per_class = np.stack([D[:, bl == c].min(1) for c in range(10)], 1)
    d_pred = per_class[np.arange(len(q)), pr]
    other = np.where(np.arange(10)[None] == pr[:, None], np.inf, per_class).min(1)
    assert np.allclose(got["trust_score"][:-3], other[:-3] / d_pred[:-3], rtol=1e-4)
    assert (got["trust_score"][-3:] > 100).all() and np.isfinite(got["trust_score"]).all()  # exact match: huge, finite
    assert np.allclose(got["knn_dist"], signals.knn_dist(q, bank, chunk=64))
    assert (signals.bank_signals(q, per_class.argmin(1), bank, bl)["trust_score"] >= 1 - 1e-6).all()


def test_mahalanobis_matches_scipy():
    rng = np.random.default_rng(4)
    lab = np.repeat(np.arange(10), 300)
    X = rng.normal(size=(3000, 20)) @ rng.normal(size=(20, 20)) + np.repeat(rng.normal(size=(10, 20)) * 3, 300, 0)
    means, prec = signals.fit_mahalanobis(X, lab, ridge=0)
    assert np.allclose(means[4], X[lab == 4].mean(0))
    C = sum(np.cov(X[lab == c].T, bias=True) * 300 for c in range(10)) / 3000  # shared (pooled) covariance
    assert np.allclose(prec, np.linalg.inv(C), rtol=1e-6, atol=1e-8)
    e, pr = rng.normal(size=(77, 20)), rng.integers(0, 10, 77)
    ref = [mahalanobis(e[i], means[pr[i]], np.linalg.inv(C)) for i in range(77)]
    assert np.allclose(signals.maha_pred(e, pr, means, prec, chunk=10), ref)
    _, prec2 = signals.fit_mahalanobis(rng.normal(size=(50, 100)), np.repeat(np.arange(10), 5))  # rank-deficient
    assert np.isfinite(prec2).all()


def test_auroc_matches_sklearn():
    rng = np.random.default_rng(5)
    r, y = rng.integers(0, 5, 1000).astype(float), rng.integers(0, 2, 1000)  # heavy ties
    assert np.isclose(auroc(r, y), roc_auc_score(y, r)) and np.isnan(auroc(r, np.zeros(1000)))


def _mock_dir():
    d = Path(tempfile.mkdtemp())
    write_mock(d, n_base=30)
    return d


def test_features_join_by_sample_id():
    _need_cifar("test_features_join_by_sample_id")
    d = _mock_dir()
    manifest, runs, tta, emb, bank, bl = load_inputs(d)
    assert not (runs.sample_id.to_numpy() == manifest.sample_id.to_numpy()).all(), "mock rows should be shuffled"
    f = build_features(manifest, runs, tta, emb, bank, bl)
    assert list(f.columns) == COLUMNS and (f.sample_id == manifest.sample_id).all()
    fi = f.set_index("sample_id")
    for sid in manifest.sample_id.sample(40, random_state=1):  # recompute each from its own parquet / .npy row
        i = runs.index[runs.sample_id == sid][0]
        lg = runs.loc[i, [f"logit_{c}" for c in range(10)]].to_numpy(float)[None]
        row = fi.loc[sid]
        assert np.isclose(row.raw_confidence, sp_softmax(lg).max())
        assert np.isclose(row.tta_pconf, signals.tta_stats(lg, tta[i:i + 1])["tta_pconf"][0])
        assert np.isclose(row.knn_dist, signals.knn_dist(emb[i:i + 1], bank)[0])
        assert row.failure == 1 - runs.loc[i, "correct"]


def test_bool_correct_accepted():
    _need_cifar("test_bool_correct_accepted")
    d = _mock_dir()
    r = pd.read_parquet(d / "prediction_runs.parquet")
    r["correct"] = r["correct"].astype(bool)
    r.to_parquet(d / "prediction_runs.parquet")
    f = build_features(*load_inputs(d))
    assert f.failure.dtype == np.int64 and set(f.failure) <= {0, 1}


def test_contract_violations_are_caught():
    _need_cifar("test_contract_violations_are_caught")
    base = _mock_dir()

    def edit_runs(fn):
        def mutate(d):
            r = pd.read_parquet(d / "prediction_runs.parquet")
            fn(r)
            r.to_parquet(d / "prediction_runs.parquet")
        return mutate

    def short_tta(d):
        np.save(d / "tta_logits.npy", np.load(d / "tta_logits.npy")[:-1])

    def drop_row(r):
        r.drop(index=0, inplace=True)

    def flip_pred(r):
        r.loc[0, "pred_label"] = (r.loc[0, "pred_label"] + 1) % 10

    def flip_correct(r):
        r.loc[0, "correct"] = 1 - r.loc[0, "correct"]

    def dup_id(r):
        r.loc[1, "sample_id"] = r.loc[0, "sample_id"]

    cases = {"missing row": edit_runs(drop_row), "short tta_logits": short_tta,
             "pred_label != argmax": edit_runs(flip_pred), "wrong correct": edit_runs(flip_correct),
             "duplicate sample_id": edit_runs(dup_id)}
    for message, mutate in cases.items():
        d = Path(tempfile.mkdtemp())
        shutil.copytree(base, d, dirs_exist_ok=True)
        mutate(d)
        try:
            build_features(*load_inputs(d))
        except AssertionError:
            pass
        else:
            raise AssertionError(f"contract violation not caught: {message}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("ok", name)
            except Skip as e:
                print("skip", e)
