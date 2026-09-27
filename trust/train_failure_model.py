"""Owner: C. Failure models (label: failure = 1 if the ResNet is wrong): logistic regression baselines and XGBoost,
with clean rows carrying 25% of the training weight. Also grouped-SHAP reasons (stability / familiarity / confidence).
"""
import json

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["raw_confidence", "entropy", "margin", "tta_agree", "tta_pconf", "tta_std", "knn_dist", "maha_pred"]
SOFTMAX = ["raw_confidence", "entropy", "margin"]
# SHAP splits credit arbitrarily between near-duplicate features, so reasons are given per group (group sums are exact)
GROUPS = {"stability": ["tta_agree", "tta_pconf", "tta_std"], "familiarity": ["knn_dist", "maha_pred"], "confidence": SOFTMAX}
XGB_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=5, tree_method="hist", random_state=0)
GRID = [dict(XGB_PARAMS, max_depth=d, n_estimators=n) for d in (3, 4, 6) for n in (200, 400)]   # python -m trust.run_trust --select
CLEAN_SHARE = 0.25   # clean rows are ~3% of rows; without this the layer comes out underconfident on clean images
MIN_SHAP = 0.1       # a reason group must add at least this much failure log-odds to be shown
# Reason wording: the image pipelines keep IMAGE_WORDS (the demos' text); trust.fit_any, for any model, uses GENERIC_WORDS
IMAGE_WORDS = {"views": "small flips/shifts", "reference": "clean training images"}
GENERIC_WORDS = {"views": "small input changes", "reference": "the reference examples"}


def clean_weights(is_clean, share=CLEAN_SHARE):
    """Row weights that give clean rows `share` of the total weight; all ones if either part is missing."""
    is_clean = np.asarray(is_clean, bool)
    n_clean, n_other = is_clean.sum(), (~is_clean).sum()
    if n_clean == 0 or n_other == 0:
        return np.ones(len(is_clean))
    return np.where(is_clean, share / (1 - share) * n_other / n_clean, 1.0)


def fit_lr(X, correct, w):
    """Logistic-regression baseline predicting P(correct) from the columns of X."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, correct, logisticregression__sample_weight=w)


def fit_xgb(X, correct, w, params=XGB_PARAMS, features=FEATURES):
    """XGBoost failure model (label: failure = 1 - correct) on the columns `features`."""
    return xgb.XGBClassifier(**params).fit(X[features], 1 - np.asarray(correct), sample_weight=w)


def p_correct_score(model, X, features=FEATURES):
    """Uncalibrated P(correct) from the failure model; isotonic calibration turns it into p_correct."""
    return model.predict_proba(X[features])[:, 0]


def reasons(model, X, decision, reference, features=FEATURES, n_views=3, words=IMAGE_WORDS):
    """JSON list of {signal, text, shap} per row: groups whose SHAP sum pushes failure risk up by >= MIN_SHAP, largest first.
    TRUST rows get []. `reference` = feature rows of correct training predictions, used to phrase values as percentiles.
    A group counts only if all its features are in `features`; `n_views` (TTA views) and `words` phrase the text."""
    groups = {g: fs for g, fs in GROUPS.items() if set(fs) <= set(features)}
    phi = model.get_booster().predict(xgb.DMatrix(X[features]), pred_contribs=True)
    group_shap = {g: phi[:, [features.index(f) for f in fs]].sum(1) for g, fs in groups.items()}
    x = {c: X[c].to_numpy() for c in features}
    ref = {c: np.sort(reference[c].to_numpy()) for c in ("raw_confidence", "tta_pconf", "knn_dist", "maha_pred") if c in x}
    share_above = lambda c: 1 - np.searchsorted(ref[c], x[c], "right") / len(ref[c])   # of correct predictions
    share_below = lambda c: np.searchsorted(ref[c], x[c], "left") / len(ref[c])
    lower_conf = share_above("raw_confidence")
    if "stability" in groups:
        lower_tta, flips = share_above("tta_pconf"), np.rint(n_views * (1 - x["tta_agree"])).astype(int)
    if "familiarity" in groups:
        farther = np.maximum(share_below("knn_dist"), share_below("maha_pred"))

    def text(g, i):
        if g == "stability":
            if flips[i]:
                return f"Prediction changed under {flips[i]} of {n_views} {words['views']}"
            return (f"Class probability under {words['views']} is {x['tta_pconf'][i]:.0%}, "
                    f"lower than {lower_tta[i]:.0%} of correct predictions")
        if g == "familiarity":
            return f"Embedding is farther from {words['reference']} than {farther[i]:.0%} of correct predictions"
        return f"Raw confidence {x['raw_confidence'][i]:.0%} is lower than {lower_conf[i]:.0%} of correct predictions"

    out = []
    for i, d in enumerate(decision):
        items = [] if d == "trust" else sorted(((float(group_shap[g][i]), g) for g in groups if group_shap[g][i] >= MIN_SHAP), reverse=True)
        out.append(json.dumps([dict(signal=g, text=text(g, i), shap=round(v, 3)) for v, g in items]))
    return out
