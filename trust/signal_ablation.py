"""Owner: B. Which signals matter, and does trust_score help? Evidence for the pitch; does not change the trust layer.
Replicates C's XGBoost setup (same XGB_PARAMS, clean weights, fold masks) on different feature sets and scores each
held-out family's test rows with C's own aurc / auroc. "all 8" reproduces evaluation.json's trust_layer numbers exactly.
Needs the full-run intermediates in data/ (manifest.csv, prediction_runs.parquet, features.csv). About 20 s.

Run: python -m trust.signal_ablation   -> prints tables, writes data/signal_ablation.json
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from trust.evaluate import aurc, auroc, paired_bootstrap
from trust.run_trust import FAMILIES, fold_masks
from trust.train_failure_model import FEATURES, GROUPS, XGB_PARAMS, clean_weights

man = pd.read_csv("data/manifest.csv")
pred = pd.read_parquet("data/prediction_runs.parquet", columns=["sample_id", "correct"])
feat = pd.read_csv("data/features.csv")
df = man.merge(pred, on="sample_id").merge(feat.drop(columns="failure"), on="sample_id")
assert len(df) == len(man) == 412000
y = df.correct.to_numpy()
is_clean = ((df.family == "clean") & (df.dataset != "cifar10_1")).to_numpy()
base = df.base_image_id.to_numpy()

SETS = {
    "all 8 (committed model)": FEATURES,
    "all 8 + trust_score": FEATURES + ["trust_score"],
    "confidence only": GROUPS["confidence"],
    "confidence + stability": GROUPS["confidence"] + GROUPS["stability"],
    "confidence + familiarity": GROUPS["confidence"] + GROUPS["familiarity"],
    "drop stability": GROUPS["confidence"] + GROUPS["familiarity"],
    "drop familiarity": GROUPS["confidence"] + GROUPS["stability"],
    "drop confidence": GROUPS["stability"] + GROUPS["familiarity"],
}
SETS = {k: v for k, v in SETS.items() if k not in ("drop stability", "drop familiarity")}  # same sets as the "+" rows

rows, boot = [], {}
t0 = time.time()
for F in FAMILIES:
    m = fold_masks(df, F)
    tr, te = m["train"], m["heldout_test"]
    w = clean_weights(is_clean[tr])
    scores = {}
    for name, cols in SETS.items():
        model = xgb.XGBClassifier(**XGB_PARAMS).fit(df.loc[tr, cols], 1 - y[tr], sample_weight=w)
        s = model.predict_proba(df.loc[te, cols])[:, 0]
        scores[name] = s
        rows.append(dict(fold=F, features=name, auroc=auroc(s, y[te]), aurc=aurc(s, y[te])))
    for f in FEATURES + ["trust_score"]:  # single signals, oriented so higher = more likely correct
        s = df.loc[te, f].to_numpy()
        s = -s if f in ("entropy", "tta_std", "knn_dist", "maha_pred") else s
        rows.append(dict(fold=F, features=f"single: {f}", auroc=auroc(s, y[te]), aurc=aurc(s, y[te])))
    boot[F] = paired_bootstrap(scores["all 8 + trust_score"], scores["all 8 (committed model)"], y[te], base[te], 1000)
    print(f"fold {F} done ({time.time() - t0:.0f}s)", file=sys.stderr)

r = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(r.pivot_table(index="features", columns="fold", values="auroc", sort=False)[FAMILIES].round(4).to_string())
print()
print(r.pivot_table(index="features", columns="fold", values="aurc", sort=False)[FAMILIES].round(4).to_string())
print("\ntrust_score added vs committed model, paired bootstrap over base images (1000):")
print(json.dumps(boot, indent=1, default=float))

out = {"note": "held-out family test rows; auroc/aurc from trust.evaluate; score = P(correct), higher = trust more",
       "auroc": r.pivot_table(index="features", columns="fold", values="auroc", sort=False)[FAMILIES].round(4).to_dict("index"),
       "aurc": r.pivot_table(index="features", columns="fold", values="aurc", sort=False)[FAMILIES].round(4).to_dict("index"),
       "trust_score_minus_committed_bootstrap": boot}
with open("data/signal_ablation.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print("wrote data/signal_ablation.json")
