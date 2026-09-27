"""Owner: C. Self-checks for the trust layer. Run from the repo root: python -m trust.test_trust (pytest also works)."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from trust import fit_any
from trust.calibrate import apply_isotonic, fit_isotonic
from trust.evaluate import (auroc, auroc_within, aurc, ece, err_at_coverage, paired_bootstrap, rc_curve, reliability,
                            summarize)
from trust.run_trust import FAMILIES, SCORE_COLS, fold_masks, load, run, select
from trust.temperature_scaling import fit_temperature, scaled_confidence, softmax
from trust.thresholds import decide, realized, tau_for
from trust.train_failure_model import CLEAN_SHARE, FEATURES, GROUPS, MIN_SHAP, clean_weights


def test_metrics_known_values():
    c, ok = np.array([.9, .8, .7, .1]), np.array([1, 1, 1, 0])
    assert np.isclose(aurc(c, ok), 0.0625)                             # risks 0, 0, 0, 1/4 at coverage 1/4 .. 1
    assert np.isclose(aurc(-c, ok), (1 + 1 / 2 + 1 / 3 + 1 / 4) / 4)   # worst order
    assert np.isclose(aurc(np.ones(4), ok), 0.25)                      # all tied: one group, risk = overall error
    assert aurc(c, np.ones(4)) == 0
    assert auroc(c, ok) == 1 and auroc(-c, ok) == 0 and auroc(np.ones(4), ok) == 0.5
    assert auroc(c, np.ones(4)) is None
    assert err_at_coverage(c, [1, 1, 0, 0], .5) == 0 and np.isclose(err_at_coverage(c, [1, 1, 0, 0], .75), 1 / 3)
    rc = rc_curve(c, ok, points=4)
    assert rc[-1] == [1.0, 0.25] and rc[0] == [0.25, 0.0]
    assert auroc_within(c, ok, ["a", "a", "a", "b"]) is None            # both groups single-class
    assert auroc_within(c, ok, ["a", "b", "a", "b"]) == 1.0            # group b has both classes


def test_weighted_metrics_match_row_duplication_and_sklearn():
    rng = np.random.default_rng(1)
    c = rng.integers(0, 20, 500) / 20                                   # many ties
    ok = (rng.random(500) < c).astype(int)
    w = rng.integers(0, 4, 500).astype(float)                           # bootstrap-style counts, including zeros
    dup = np.repeat(np.arange(500), w.astype(int))
    assert np.isclose(aurc(c, ok, w), aurc(c[dup], ok[dup]))
    assert np.isclose(auroc(c, ok, w), auroc(c[dup], ok[dup]))
    assert np.isclose(auroc(c, ok, w), roc_auc_score(ok, c, sample_weight=w))
    assert np.isclose(auroc(c, ok), roc_auc_score(ok, c))


def test_calibration_metrics():
    ok = np.array([1, 0, 1, 1])
    assert ece(ok.astype(float), ok) == 0                              # confidence equals correctness
    prob, ok = np.full(10, .7), np.array([1] * 7 + [0] * 3)
    assert np.isclose(ece(prob, ok), 0) and np.isclose(ece(np.full(10, .9), ok), .2)
    rel = reliability(prob, ok)
    assert len(rel) == 1 and rel[0]["n"] == 10 and np.isclose(rel[0]["acc"], .7) and np.isclose(rel[0]["conf"], .7)
    assert ece(np.ones(3), np.ones(3)) == 0                            # prob exactly 1.0 lands in the last bin
    s = summarize(prob, prob, ok, ["g"] * 10)
    assert set(s) == {"aurc", "auroc", "auroc_within", "ece", "mean_conf", "err_at_20", "err_at_50"}


def test_paired_bootstrap():
    rng = np.random.default_rng(2)
    ok = (rng.random(2000) < .7).astype(int)
    base = np.repeat(np.arange(400), 5)                                 # 5 rows per base image
    good, noise = ok + rng.normal(0, .3, 2000), rng.random(2000)
    b = paired_bootstrap(good, noise, ok, base, n=300)
    assert b["aurc_diff"] < 0 and b["aurc_diff_ci"][1] < 0             # better ranking -> lower AURC, CI excludes 0
    assert b["auroc_diff"] > 0 and b["auroc_diff_ci"][0] > 0
    same = paired_bootstrap(noise, noise, ok, base, n=50)
    assert same["aurc_diff_ci"] == [0.0, 0.0] and same["auroc_diff_ci"] == [0.0, 0.0]
    # resampling unit is the base image: copying every row 5x under the same base id must not shrink the CI
    # (a row-level bootstrap would shrink it by ~sqrt(5))
    one = np.arange(400) * 5                                            # one row per base image
    width = lambda r: r["auroc_diff_ci"][1] - r["auroc_diff_ci"][0]
    w1 = width(paired_bootstrap(good[one], noise[one], ok[one], base[one], n=400))
    rep = np.repeat(one, 5)
    w5 = width(paired_bootstrap(good[rep], noise[rep], ok[rep], base[rep], n=400))
    assert .8 < w5 / w1 < 1.25, (w1, w5)


def test_thresholds():
    c, ok = np.array([.9, .8, .7, .6]), np.array([1, 1, 0, 1])
    assert tau_for(c, ok, .25) == .6                                   # all 4 accepted: error 1/4
    assert tau_for(c, ok, .2) == .8 and tau_for(c, ok, 0) == .8        # top 2 accepted: error 0
    assert tau_for(c, np.zeros(4), .1) == np.inf                       # unreachable -> accept nothing
    ties, ok3 = np.array([.9, .9, .5]), np.array([1, 0, 1])
    assert tau_for(ties, ok3, .4) == .5                                # never cuts inside the .9 tie (error 1/2)
    assert tau_for(ties, ok3, .3) == np.inf
    assert list(decide([.1, .5, .8, .95], .5, .9)) == ["reject", "caution", "caution", "trust"]
    assert list(decide([.1, .99], .5, np.inf)) == ["reject", "caution"]
    r = realized(c, ok, .75, .05)
    assert r["coverage"] == .5 and r["error"] == 0 and realized(c, ok, np.inf, .01)["error"] is None


def test_temperature_scaling():
    rng = np.random.default_rng(3)
    z = rng.normal(0, 3, (20000, 10))
    labels = (softmax(z / 2.0).cumsum(1) < rng.random((20000, 1))).sum(1)   # labels drawn at true T = 2
    T = fit_temperature(z, labels)
    assert abs(T - 2.0) < 0.1, T
    assert np.allclose(scaled_confidence(z, 1.0), softmax(z).max(1))
    assert (scaled_confidence(z, T) <= scaled_confidence(z, 1.0) + 1e-12).all()   # T > 1 only softens


def test_isotonic_json_matches_sklearn():
    rng = np.random.default_rng(4)
    s = rng.random(5000)
    c = (rng.random(5000) < s ** 2).astype(int)
    w = rng.integers(1, 4, 5000)
    cal = fit_isotonic(s, c, w)
    ref = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s, c, sample_weight=w)
    q = np.r_[rng.random(1000), -1.0, 2.0]                              # includes out-of-range scores
    assert np.allclose(apply_isotonic(cal, q), ref.predict(q))
    assert apply_isotonic(json.loads(json.dumps(cal)), .5) == apply_isotonic(cal, .5)
    # regression: fit on float32 scores (what XGBoost returns), sklearn's predict splits flat levels into values
    # ~1e-7 apart, which let tau_for cut inside a level (prototype: a "5%" rule that ran at 5.5% on cal).
    # apply_isotonic must keep every score in a flat segment exactly tied.
    s32 = s.astype(np.float32)
    cal32 = fit_isotonic(s32, c)
    x, yv = np.array(cal32["x"]), np.array(cal32["y"])
    flat = np.nonzero(yv[1:] == yv[:-1])[0]
    assert len(flat) > 10
    for i in flat:
        assert len(np.unique(apply_isotonic(cal32, s32[(s32 >= x[i]) & (s32 <= x[i + 1])]))) == 1


def test_clean_weights():
    is_clean = np.r_[np.ones(30, bool), np.zeros(930, bool)]
    w = clean_weights(is_clean)
    assert np.isclose(w[is_clean].sum() / w.sum(), CLEAN_SHARE) and (w[~is_clean] == 1).all()
    assert (clean_weights(np.zeros(5, bool)) == 1).all() and (clean_weights(np.ones(5, bool)) == 1).all()


FAMILY_OF = {"gaussian_noise": "noise", "impulse_noise": "noise", "defocus_blur": "blur", "motion_blur": "blur",
             "fog": "weather", "brightness": "weather", "contrast": "digital", "jpeg_compression": "digital"}
HARDNESS = {"clean": 0, "noise": .9, "blur": .6, "weather": .3, "digital": .5, "natural": .3}


def write_mock(d, n_base=200, n_c101=100, seed=0):
    """Contract-shaped manifest.csv / prediction_runs.parquet / features.csv from a synthetic classifier."""
    rng, d = np.random.default_rng(seed), Path(d)
    split_of = np.array(["train"] * 10 + ["val"] * 3 + ["cal"] * 3 + ["test"] * 4)[rng.permutation(np.arange(n_base) % 20)]
    labels = rng.integers(0, 10, n_base)
    conds = [("clean", "clean", 0)] + [(c, f, s) for c, f in FAMILY_OF.items() for s in range(1, 6)]
    rows = [(f"c10_{i:05d}_{c}_s{s}", f"c10_{i:05d}", "cifar10_test" if s == 0 else "cifar10c", labels[i], c, f, s, split_of[i])
            for i in range(n_base) for c, f, s in conds]
    rows += [(f"c101_{i:05d}", f"c101_{i:05d}", "cifar10_1", rng.integers(10), "natural", "natural", 0, "test") for i in range(n_c101)]
    man = pd.DataFrame(rows, columns=["sample_id", "base_image_id", "dataset", "true_label", "corruption", "family", "severity", "split"])
    man["true_class"] = "class_" + man.true_label.astype(str)
    n, y = len(man), man.true_label.to_numpy()
    hard = man.family.map(HARDNESS).to_numpy() * np.maximum(man.severity.to_numpy(), 1)
    z = rng.normal(0, 1, (n, 10))
    z[np.arange(n), y] += 4 - 1.2 * hard + rng.normal(0, 1.5, n)
    pred, p = z.argmax(1), softmax(z)
    ps = np.sort(p, 1)
    pv = softmax(z[:, None, :] + rng.normal(0, 1, (n, 3, 10)) * (.3 + .4 * hard)[:, None, None])   # 3 TTA views
    pc = pv[np.arange(n)[:, None], np.arange(3)[None, :], pred[:, None]]
    man.to_csv(d / "manifest.csv", index=False)
    pr = pd.DataFrame({"sample_id": man.sample_id, "pred_label": pred, "correct": (pred == y).astype(int),
                       **{f"logit_{k}": z[:, k].astype(np.float32) for k in range(10)}})
    pr.to_parquet(d / "prediction_runs.parquet", index=False)
    pd.DataFrame(dict(sample_id=man.sample_id, raw_confidence=ps[:, -1], entropy=-(p * np.log(p)).sum(1), margin=ps[:, -1] - ps[:, -2],
                      tta_agree=(pv.argmax(2) == pred[:, None]).mean(1), tta_pconf=pc.mean(1),
                      tta_std=np.c_[p[np.arange(n), pred], pc].std(1), knn_dist=.02 + .01 * hard + rng.gamma(2, .005, n),
                      maha_pred=300 + 80 * hard + rng.gamma(2, 40, n), failure=(pred != y).astype(int))
                 ).to_csv(d / "features.csv", index=False)
    return man


def _strict_json(path):   # rejects NaN / Infinity, which JavaScript's JSON.parse cannot read
    return json.loads(path.read_text(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"{c} in {path.name}")))


def test_pipeline_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        d, man = Path(tmp), write_mock(tmp)
        run(tmp)
        ev, th = _strict_json(d / "evaluation.json"), _strict_json(d / "thresholds.json")
        assert ev["headline_fold"] == "blur" and set(ev["folds"]) == set(FAMILIES) and th["fold"] == "blur"
        methods = {"raw_confidence", "temp_clean", "temp_corrupted", "tta_only", "lr_softmax_only", "lr_all", "trust_layer"}
        for fo in ev["folds"].values():
            assert set(fo["heldout_test"]["methods"]) == methods
            assert fo["heldout_test"]["methods"]["trust_layer"]["auroc"] > .6          # the synthetic signal is learnable
            t = fo["thresholds"]
            assert t["tau_trust"] is None or t["tau_reject"] <= t["tau_trust"]
            assert set(fo["bootstrap_heldout"]) == {"trust_minus_raw_confidence", "trust_minus_temp_corrupted", "trust_minus_tta_only"}
        assert [r["severity"] for r in ev["folds"]["blur"]["headline_chart"]] == [0, 1, 2, 3, 4, 5]
        assert ev["cifar10_1"]["n"] == 100 and "bootstrap" in ev["cifar10_1"]

        sc = pd.read_parquet(d / "scores.parquet")
        n_test = man.base_image_id[(man.split == "test") & (man.dataset != "cifar10_1")].nunique()
        assert list(sc.columns) == SCORE_COLS and sc.sample_id.is_unique and len(sc) == n_test * 41 + 100
        assert sc.p_correct.between(0, 1).all() and set(sc.decision) <= {"trust", "caution", "reject"}
        corrupted = sc.family.isin(FAMILIES)
        assert (sc.fold[corrupted] == sc.family[corrupted]).all() and (sc.fold[~corrupted] == "blur").all()
        blur = sc[sc.fold == "blur"]
        assert (decide(blur.p_correct, th["tau_reject"], th["tau_trust"] or np.inf) == blur.decision).all()
        rs = sc.reasons.map(json.loads)
        assert (rs[sc.decision == "trust"].map(len) == 0).all() and rs.map(len).sum() > 0
        for r in rs:
            assert all(x["signal"] in GROUPS and x["shap"] >= MIN_SHAP for x in r)
            assert [x["shap"] for x in r] == sorted((x["shap"] for x in r), reverse=True)

        feats = pd.read_csv(d / "features.csv").set_index("sample_id")    # saved models reproduce the stored p_correct
        p_of = {}
        for F in FAMILIES:
            booster = xgb.Booster()
            booster.load_model(d / "models" / f"xgb_{F}.json")
            cal = json.loads((d / "models" / f"isotonic_{F}.json").read_text())
            p_of[F] = lambda ids, b=booster, c=cal: apply_isotonic(c, 1 - b.predict(xgb.DMatrix(feats.loc[ids, FEATURES])))
            part = sc[sc.fold == F]
            assert np.allclose(p_of[F](part.sample_id), part.p_correct, atol=1e-6)

        # data-use contract: temperatures come from val rows only, thresholds from cal rows only
        df = load(tmp)
        m, y, Z = fold_masks(df, "blur"), df.correct.to_numpy(), df[[f"logit_{k}" for k in range(10)]].to_numpy(np.float64)
        val_clean = (df.split == "val").to_numpy() & (df.family == "clean").to_numpy()
        assert np.isclose(fit_temperature(Z[val_clean], df.true_label[val_clean]), th["T_clean"])
        assert np.isclose(fit_temperature(Z[m["val"]], df.true_label[m["val"]]), th["T_corrupted"])
        p_cal = p_of["blur"](df.sample_id[m["cal"]])
        assert np.isclose(tau_for(p_cal, y[m["cal"]], .05), th["tau_reject"])
        assert np.isclose(tau_for(p_cal, y[m["cal"]], .01), th["tau_trust"] if th["tau_trust"] is not None else np.inf)


def test_fold_masks_no_leakage():
    with tempfile.TemporaryDirectory() as tmp:
        write_mock(tmp, n_base=60, n_c101=10)
        df = load(tmp)
        c101, fam, base = (df.dataset == "cifar10_1").to_numpy(), df.family.to_numpy(), df.base_image_id.to_numpy()
        for F in FAMILIES:
            m = fold_masks(df, F)
            fit = m["train"] | m["val"] | m["cal"]
            assert not (fit & (fam == F)).any() and not (fit & c101).any()           # held-out family / CIFAR-10.1 never fit
            assert m["train"].any() and m["val"].any() and m["cal"].any() and m["heldout_test"].any()
            assert not set(base[m["train"]]) & set(base[m["val"] | m["cal"] | m["heldout_test"] | m["clean_test"]])
            assert (df.split[m["heldout_test"]] == "test").all() and (fam[m["heldout_test"]] == F).all()
            assert set(fam[m["seen_test"]]) == set(FAMILIES) - {F}


def test_select_writes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        write_mock(tmp, n_base=60, n_c101=10)
        before = sorted(Path(tmp).rglob("*"))
        select(tmp)
        assert sorted(Path(tmp).rglob("*")) == before


def test_contract_violations_are_caught():
    def edit(name, fn):
        def corrupt(d):
            path = d / name
            df = fn(pd.read_parquet(path) if name.endswith(".parquet") else pd.read_csv(path))
            df.to_parquet(path, index=False) if name.endswith(".parquet") else df.to_csv(path, index=False)
        return corrupt

    def flip(col, row=0):
        def fn(df):
            df.loc[row, col] = 1 - df.loc[row, col]
            return df
        return fn

    def leak(m):   # move one row of base image c10_00000 to a different split
        m.loc[0, "split"] = "val" if m.loc[0, "split"] != "val" else "cal"
        return m

    cases = [(None, None),
             (edit("manifest.csv", leak), "more than one split"),
             (edit("features.csv", lambda f: f.iloc[1:]), "sample_ids differ"),
             (edit("features.csv", lambda f: f.drop(columns="knn_dist")), "missing columns"),
             (edit("features.csv", flip("failure")), "failure != 1 - correct"),
             (edit("prediction_runs.parquet", flip("correct")), "correct != (pred_label == true_label)"),
             (edit("features.csv", lambda f: f.assign(raw_confidence=f.raw_confidence.sample(frac=1, random_state=0).to_numpy())),
              "misaligned"),
             (edit("manifest.csv", lambda m: m.assign(dataset=m.dataset.replace("cifar10_1", "cifar10.1"))), "unknown dataset")]
    for corrupt, message in cases:
        with tempfile.TemporaryDirectory() as tmp:
            write_mock(tmp, n_base=40, n_c101=20)
            if corrupt is None:
                load(tmp)                                                 # the untouched mock passes
                continue
            corrupt(Path(tmp))
            try:
                load(tmp)
            except AssertionError as e:
                assert message in str(e), (message, str(e))
            else:
                raise AssertionError(f"contract violation not caught: {message}")


def write_any(d, K=3, n_items=500, views=0, dim=0, split=False, seed=0):
    """trust.fit_any's input format from a synthetic classifier: each item clean + 2 harder conditions. Harder rows get a
    weaker class signal, less stable TTA views and embeddings farther from the reference."""
    rng, d = np.random.default_rng(seed), Path(d)
    item, c = np.repeat(np.arange(n_items), 3), np.tile(np.arange(3), n_items)
    n, y = len(item), rng.integers(0, K, n_items)[item]
    hard = c + rng.gamma(2, .3, n)
    z = rng.normal(0, 1, (n, K))
    z[np.arange(n), y] += 3.5 - 1.2 * hard
    df = pd.DataFrame({"sample_id": [f"s{i}" for i in range(n)], "group_id": [f"item{i}" for i in item], "label": y,
                       **{f"logit_{k}": z[:, k] for k in range(K)}, "condition": np.array(["clean", "fog", "glare"])[c]})
    if split:
        df["split"] = rng.choice(["train", "val", "cal", "test"], n_items, p=[.5, .15, .15, .2])[item]
    df.to_csv(d / "predictions.csv", index=False)
    if views:
        np.save(d / "tta_logits.npy", z[:, None] + rng.normal(0, 1, (n, views, K)) * (.3 + .6 * hard)[:, None, None])
    if dim:
        protos, ref_y = rng.normal(0, 2, (K, dim)), np.arange(40 * K) % K
        np.save(d / "embeddings.npy", protos[y] + rng.normal(0, 1, (n, dim)) * (1 + hard)[:, None])
        np.save(d / "reference_embeddings.npy", protos[ref_y] + rng.normal(0, 1, (len(ref_y), dim)))
        np.save(d / "reference_labels.npy", ref_y)
        (d / "classes.txt").write_text("".join(f"class {k}\n" for k in range(K)))
    return df


def _layer_matches_fit(df, rep, layer, te, tta=None, emb=None):
    """score() on the test rows, from the saved layer, reproduces fit()'s test numbers."""
    res = fit_any.score(layer, df.loc[te, [c for c in df if c.startswith("logit_")]], tta, emb)
    ok, t = (res.pred == df.label[te].to_numpy()).to_numpy(), rep["test"]
    assert np.isclose(ok.mean(), t["accuracy"]) and np.isclose(ece(res.p_correct, ok), t["ece"]["trust_layer"])
    for d in fit_any.DECISIONS:
        assert np.isclose((res.decision == d).mean(), t["decisions"]["trust_layer"][d]["share"])
    rs = res.reasons.map(json.loads)
    assert (rs[res.decision == "trust"].map(len) == 0).all() and rs.map(len).sum() > 0
    return res, rs


def test_fit_any_k3_confidence_only():
    with tempfile.TemporaryDirectory() as tmp:
        d, out = Path(tmp), Path(tmp) / "layer"
        df = write_any(d)
        rep = fit_any.fit(d, out)
        assert {p.name for p in out.iterdir()} == {"xgb.json", "isotonic.json", "trust_config.json", "reference.npz", "report.json"}
        cfg = _strict_json(out / "trust_config.json")
        assert cfg["features"] == ["raw_confidence", "entropy", "margin"] and cfg["n_classes"] == 3 and cfg["n_views"] is None
        assert cfg["split_seed"] == 0 and cfg["class_names"] is None and cfg["targets"] == {"reject": .05, "trust": .01}
        assert cfg["tau_reject"] <= (cfg["tau_trust"] or 1)
        split = fit_any.split_by_group(df, cfg["split_seed"])   # no split column: whole items, 50/15/15/20 of them
        assert (pd.Series(split).groupby(df.group_id).nunique() == 1).all()
        assert rep["rows"] == {s: int((split == s).sum()) for s in fit_any.SPLITS}
        assert all(abs(rep["rows"][s] / len(df) - f) < .02 for s, f in fit_any.SPLITS.items())
        assert any("rows in cal" in w for w in rep["warnings"])                  # thin data warns instead of failing
        assert rep["test"]["auroc"]["trust_layer"] > .7 and list(rep["test_by_condition"]) == ["clean", "fog", "glare"]
        _, rs = _layer_matches_fit(df, rep, fit_any.load_layer(out), split == "test")
        assert {r["signal"] for x in rs for r in x} == {"confidence"}


def test_fit_any_all_signals():
    with tempfile.TemporaryDirectory() as tmp:
        d, out = Path(tmp), Path(tmp) / "layer"
        df = write_any(d, K=4, views=4, dim=8, split=True)
        rep = fit_any.fit(d, out)
        cfg = _strict_json(out / "trust_config.json")
        assert cfg["features"] == FEATURES and cfg["n_classes"] == 4 and cfg["n_views"] == 4 and cfg["split_seed"] is None
        assert cfg["class_names"] == ["class 0", "class 1", "class 2", "class 3"]
        te = (df.split == "test").to_numpy()
        layer = fit_any.load_layer(out)
        res, rs = _layer_matches_fit(df, rep, layer, te, np.load(d / "tta_logits.npy")[te], np.load(d / "embeddings.npy")[te])
        assert (res.pred_class == "class " + res.pred.astype(str)).all()
        text = " ".join(r["text"] for x in rs for r in x)
        assert "of 4 small input changes" in text and "the reference examples" in text                # generic wording
        assert "flips/shifts" not in text and "training images" not in text
        try:
            fit_any.score(layer, df.loc[te, [f"logit_{k}" for k in range(4)]])
        except fit_any.InputError as e:
            assert "tta_logits" in str(e)
        else:
            raise AssertionError("scoring without the TTA logits the layer needs must fail")
        run_cli = lambda *a: subprocess.run([sys.executable, "-m", "trust.fit_any", *map(str, a)], capture_output=True, text=True)
        cli = run_cli("--score", out, "--data", d, "--out", d / "scores.csv")
        scored = pd.read_csv(d / "scores.csv")
        assert cli.returncode == 0 and list(scored.sample_id) == list(df.sample_id) and scored.p_correct.between(0, 1).all()


def test_fit_any_rejects_bad_inputs():
    csv = lambda fn: lambda d: fn(pd.read_csv(d / "predictions.csv")).to_csv(d / "predictions.csv", index=False)
    npy = lambda name, fn: lambda d: np.save(d / name, fn(np.load(d / name)))
    cases = [(csv(lambda f: f.drop(columns="label")), "missing the column(s) label"),
             (csv(lambda f: f.drop(columns="logit_1")), "no gaps; found logit_0, logit_2"),
             (csv(lambda f: f.drop(columns=["logit_1", "logit_2"])), "at least 2 classes"),
             (csv(lambda f: f.assign(sample_id="same")), "sample_id must be unique"),
             (csv(lambda f: f.assign(label=f.label.where(f.index != 5, 3))), "line 7 has '3'"),
             (csv(lambda f: f.assign(logit_0=f.logit_0.where(f.index != 3, np.nan))), "first on line 5"),
             (csv(lambda f: f.assign(split=np.where(f.index == 0, "test", "train"))), "more than one split"),
             (csv(lambda f: f.assign(split="validation")), "split must be one of train, val, cal, test"),
             (csv(lambda f: f.assign(split=np.where(f.index < 60, "train", "test"))), "the val split is empty"),
             (npy("tta_logits.npy", lambda a: a[:-1]), "tta_logits.npy must be N x V x K = 120 x V x 3"),
             (npy("embeddings.npy", lambda a: a[:, :2]), "reference_embeddings.npy must be M x 2"),
             (npy("reference_labels.npy", lambda a: np.minimum(a, 1)), "must include every class; missing 2"),
             (lambda d: (d / "reference_labels.npy").unlink(), "needs reference_embeddings.npy and reference_labels.npy")]
    for edit, message in cases:
        with tempfile.TemporaryDirectory() as tmp:
            write_any(tmp, n_items=40, views=2, dim=4)
            edit(Path(tmp))
            try:
                fit_any.fit(tmp, Path(tmp) / "layer")
            except fit_any.InputError as e:
                assert message in str(e), (message, str(e))
            else:
                raise AssertionError(f"bad input not caught: {message}")
            if message.startswith("missing"):   # the CLI prints the message, not a traceback
                cli = subprocess.run([sys.executable, "-m", "trust.fit_any", "--data", tmp, "--out", f"{tmp}/layer"],
                                     capture_output=True, text=True)
                assert cli.returncode == 1 and cli.stderr.startswith("error: ") and "Traceback" not in cli.stderr


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
