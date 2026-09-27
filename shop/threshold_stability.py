"""Why the shop's thresholds are cross-fitted: across 10 random reshuffles of the 380 non-test photos (test photos fixed),
compare the old fixed split (train / 71 val / 71 cal photos) with 5-fold cross-fitting over all 380. Reports the mean
and std of both thresholds and of the test-set decision shares and errors. Needs the pipeline outputs in data/shop/.

Run: python -m shop.threshold_stability   (~30 s)
"""
import numpy as np, pandas as pd
from trust.calibrate import apply_isotonic, fit_isotonic
from trust.thresholds import decide, tau_for
from trust.train_failure_model import FEATURES, clean_weights, fit_xgb, p_correct_score
s = pd.read_parquet("data/shop/scores.parquet"); f = pd.read_csv("data/shop/features.csv")
d = s[["sample_id", "base_image_id", "split", "family", "true_label", "correct"]].merge(f, on="sample_id")
X, y = d[FEATURES], d.correct.to_numpy(); clean = (d.family == "clean").to_numpy(); te = (d.split == "test").to_numpy()
bases = d[~te].drop_duplicates("base_image_id")[["base_image_id", "true_label"]].reset_index(drop=True)
def strat_folds(labels, k, rng):
    f = np.empty(len(labels), int)
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c)); f[idx] = np.arange(len(idx)) % k
    return f
def summarize(p, taus):
    dec = decide(p[te], *taus); out = {"tau_reject": taus[0], "tau_trust": taus[1]}
    for k in ("trust", "caution", "reject"):
        m = dec == k; out[f"{k}_share"] = m.mean(); out[f"{k}_err"] = 1 - y[te][m].mean() if m.any() else np.nan
    return out
rows = []
for seed in range(10):
    rng = np.random.default_rng(100 + seed)
    # OLD: 50/15/15 of the non-test photos (same proportions as now) -> train / val / cal
    role = np.empty(len(bases), object)                          # per class: 50/80 train, 15/80 val, 15/80 cal
    for c in np.unique(bases.true_label):
        idx = rng.permutation(np.flatnonzero(bases.true_label.to_numpy() == c)); q = (np.arange(len(idx)) + 0.5) / len(idx)
        role[idx] = np.where(q < 50 / 80, "train", np.where(q < 65 / 80, "val", "cal"))
    r = d.base_image_id.map(dict(zip(bases.base_image_id, role))).to_numpy()
    tr, va, ca = (r == "train"), (r == "val"), (r == "cal")
    m = fit_xgb(X[tr], y[tr], clean_weights(clean[tr])); sc = p_correct_score(m, X)
    iso = fit_isotonic(sc[va], y[va], clean_weights(clean[va])); p = apply_isotonic(iso, sc)
    rows.append({"scheme": "old (71 cal photos)", **summarize(p, (tau_for(p[ca], y[ca], .05), tau_for(p[ca], y[ca], .01)))})
    # NEW: 5-fold cross-fit over all 380 non-test photos; final model on all; isotonic + taus from out-of-fold scores
    fold = d.base_image_id.map(dict(zip(bases.base_image_id, strat_folds(bases.true_label.to_numpy(), 5, rng)))).to_numpy()
    oof = np.full(len(d), np.nan)
    for k in range(5):
        trk, hk = (~te) & (fold != k), (~te) & (fold == k)
        oof[hk] = p_correct_score(fit_xgb(X[trk], y[trk], clean_weights(clean[trk])), X[hk])
    nt = ~te
    iso = fit_isotonic(oof[nt], y[nt], clean_weights(clean[nt])); p_oof = apply_isotonic(iso, oof[nt])
    taus = (tau_for(p_oof, y[nt], .05), tau_for(p_oof, y[nt], .01))
    final = fit_xgb(X[nt], y[nt], clean_weights(clean[nt])); p = apply_isotonic(iso, p_correct_score(final, X))
    rows.append({"scheme": "new (5-fold cross-fit, 380 photos)", **summarize(p, taus)})
r = pd.DataFrame(rows); pd.set_option("display.width", 200)
print(r.groupby("scheme").agg(["mean", "std"]).round(3).T.to_string())
