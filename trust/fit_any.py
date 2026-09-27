"""Fit the trust layer to any classifier's outputs, then score new predictions with it. The classifier is never retrained.

Fit:    python -m trust.fit_any --data <input folder> --out <layer folder>
Score:  python -m trust.fit_any --score <layer folder> --data <input folder> --out <scores.csv>
Python: layer = load_layer(folder); score(layer, logits, tta_logits, embeddings) -> DataFrame

Input folder (step-by-step guide with export examples: docs/USE_WITH_YOUR_MODEL.md):
  predictions.csv  one row per prediction: sample_id (unique), group_id (the original item: all its versions land in one
                   split), label (0..K-1), logit_0 .. logit_{K-1} (raw scores before softmax, any K >= 2). Optional:
                   split (train / val / cal / test; default: by group_id, 50/15/15/20, stratified by label, seed 0) and
                   condition ("clean" or the shift's name; clean rows get 25% of the training weight, and the report
                   breaks results down by condition). Scoring needs only sample_id and the logits.
  classes.txt      optional: the K class names, one per line, in class order.
  tta_logits.npy   optional, N x V x K: logits on V small changes that shouldn't change the answer -> stability signals.
  embeddings.npy   optional, N x D second-to-last-layer vectors, plus reference_embeddings.npy (M x D) and
                   reference_labels.npy (M) from clean training data -> familiarity signals.
Rows of every .npy follow predictions.csv.

The same steps as trust/run_trust.py, on whichever signals the folder provides: XGBoost failure model on train,
isotonic calibration on val, REJECT / TRUST cutoffs on cal (at most 5% / 1% wrong), a report on test.
Writes to the layer folder: xgb.json, isotonic.json, trust_config.json, reference.npz (what scoring needs besides
those: the reference embeddings and the correct train rows the reasons compare against) and report.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from benchmark.make_splits import SPLITS, assign_splits
from trust import signals
from trust.calibrate import apply_isotonic, fit_isotonic
from trust.evaluate import auroc, ece
from trust.run_trust import TARGET_REJECT, TARGET_TRUST, _write_json
from trust.thresholds import decide, tau_for
from trust.train_failure_model import GENERIC_WORDS, SOFTMAX, clean_weights, fit_xgb, p_correct_score, reasons

SEED = 0
MIN_CAL_ROWS, MIN_TRAIN_ERRORS = 1000, 100   # below these, warn: noisy cutoffs, too few mistakes to learn from
DECISIONS = ("trust", "caution", "reject")


class InputError(ValueError):
    """The input breaks the format; the message says what to fix."""


def need(ok, message):
    if not ok:
        raise InputError(message)


def _npy(folder, name):
    return np.load(folder / name, mmap_mode="r") if (folder / name).is_file() else None


def _shape(a):
    return " x ".join(map(str, a.shape))


def load(folder, labeled=True):
    """The validated input folder as a dict: table, logits (N x K), names, tta, emb, bank, bank_labels (None if absent).
    labeled=False (scoring) needs only sample_id and the logits, and ignores the reference files."""
    folder = Path(folder)
    need((folder / "predictions.csv").is_file(), f"no predictions.csv in {folder}")
    df = pd.read_csv(folder / "predictions.csv")
    found = [c for c in df.columns if c.startswith("logit_")]
    cols = [f"logit_{k}" for k in range(len(found))]
    K, n = len(cols), len(df)
    need(K >= 2, "predictions.csv needs one column per class, logit_0 .. logit_{K-1}, with at least 2 classes")
    need(set(found) == set(cols), f"logit columns must be numbered logit_0 .. logit_{K - 1} with no gaps; found {', '.join(found)}")
    missing = [c for c in (["sample_id", "group_id", "label"] if labeled else ["sample_id"]) if c not in df.columns]
    need(not missing, f"predictions.csv is missing the column(s) {', '.join(missing)}")

    def rows_ok(bad, message):   # message may use {line} and {value}: the first bad row's line (1 = header) and value
        bad = np.asarray(bad, bool)
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            raise InputError(message.format(line=i + 2, value=df.iloc[i].to_dict(), count=int(bad.sum())))

    Z = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    rows_ok(~np.isfinite(Z).all(1), "{count} row(s) have a logit that is empty, not a number or infinite (first on line {line})")
    rows_ok(df.sample_id.isna(), "sample_id is empty on line {line}")
    rows_ok(df.sample_id.duplicated(), "sample_id must be unique; '{value[sample_id]}' appears again on line {line}")
    if "condition" in df:
        rows_ok(df.condition.isna(), 'condition is empty on line {line}; use "clean" or the name of the shift')
    if labeled:
        label = pd.to_numeric(df.label, errors="coerce")
        rows_ok(~label.between(0, K - 1) | (label % 1 != 0), f"label must be a class index from 0 to {K - 1}; "
                "line {line} has '{value[label]}'")
        df["label"] = label.astype(int)
        rows_ok(df.group_id.isna(), "group_id is empty on line {line}")
        if "split" in df:
            rows_ok(~df.split.isin(list(SPLITS)), f"split must be one of {', '.join(SPLITS)}; line {{line}} has '{{value[split]}}'")
            rows_ok(df.groupby("group_id").split.transform("nunique") > 1, "group_id '{value[group_id]}' (line {line}) has "
                    "rows in more than one split; every version of an item must be in the same split, or testing leaks into training")

    names = None
    if (folder / "classes.txt").is_file():
        names = [s.strip() for s in (folder / "classes.txt").read_text().splitlines() if s.strip()]
        need(len(names) == K, f"classes.txt has {len(names)} names but predictions.csv has {K} logit columns; write one name per line")
    tta = _npy(folder, "tta_logits.npy")
    if tta is not None:
        need(tta.ndim == 3 and tta.shape[0] == n and tta.shape[2] == K and tta.shape[1] > 0,
             f"tta_logits.npy must be N x V x K = {n} x V x {K} (rows in predictions.csv order); it is {_shape(tta)}")
        need(np.isfinite(tta).all(), "tta_logits.npy has NaN or infinite values")
    emb = _npy(folder, "embeddings.npy")
    if emb is not None:
        need(emb.ndim == 2 and len(emb) == n, f"embeddings.npy must be N x D with N = {n} (rows in predictions.csv order); it is {_shape(emb)}")
        need(np.isfinite(emb).all(), "embeddings.npy has NaN or infinite values")
    bank = bank_labels = None
    if labeled and emb is not None:
        bank, bank_labels = _npy(folder, "reference_embeddings.npy"), _npy(folder, "reference_labels.npy")
        need(bank is not None and bank_labels is not None, "embeddings.npy needs reference_embeddings.npy and reference_labels.npy "
             "next to it: embeddings and labels of clean training data")
        need(bank.ndim == 2 and bank.shape[1] == emb.shape[1],
             f"reference_embeddings.npy must be M x {emb.shape[1]} (the same D as embeddings.npy); it is {_shape(bank)}")
        need(len(bank) >= signals.KNN_K, f"reference_embeddings.npy needs at least {signals.KNN_K} rows; it has {len(bank)}")
        need(np.isfinite(bank).all(), "reference_embeddings.npy has NaN or infinite values")
        need(bank_labels.shape == (len(bank),), f"reference_labels.npy must hold one label per reference row ({len(bank)}); it is {_shape(bank_labels)}")
        classes = np.unique(bank_labels)
        need(np.isin(classes, np.arange(K)).all(), f"reference_labels.npy must be class indexes 0..{K - 1}")
        absent = sorted(set(range(K)) - set(classes.astype(int).tolist()))
        need(not absent, f"reference_labels.npy must include every class; missing {', '.join(map(str, absent))}")
        bank_labels = np.asarray(bank_labels, np.int64)
    elif labeled:
        need(not (folder / "reference_embeddings.npy").exists(), "reference_embeddings.npy is here but embeddings.npy is not; "
             "add embeddings.npy for the familiarity signals, or remove the reference files")
    return dict(table=df, logits=Z, names=names, tta=tta, emb=emb, bank=bank, bank_labels=bank_labels)


def signal_table(Z, tta=None, emb=None, bank=None, bank_labels=None, maha=None):
    """(features, predicted class). Confidence always; stability with TTA logits; familiarity with embeddings plus the
    reference bank and its Mahalanobis fit. Columns come out in trust.train_failure_model.FEATURES order."""
    sm = signals.softmax_stats(Z)
    X = pd.DataFrame({k: sm[k] for k in SOFTMAX})
    if tta is not None:
        X = X.assign(**signals.tta_stats(Z, tta))
    if emb is not None:
        X["knn_dist"] = signals.bank_signals(emb, sm["pred"], bank, bank_labels, n_classes=Z.shape[1])["knn_dist"]
        X["maha_pred"] = signals.maha_pred(emb, sm["pred"], *maha)
    return X, sm["pred"]


def split_by_group(df, seed=SEED):
    """train / val / cal / test per row: whole groups, 50/15/15/20, stratified by label (each group's first row),
    with the benchmark's own splitter."""
    groups = df.groupby("group_id").label.first()
    split_of = dict(zip(groups.index, assign_splits(groups.to_numpy(), np.random.default_rng(seed))))
    return df.group_id.map(split_of).to_numpy()


def fit(data, out, seed=SEED):
    """Fit the trust layer on the input folder `data`, save it to `out`, print the test report and return it."""
    inp = load(data)
    df, Z, tta, emb = inp["table"], inp["logits"], inp["tta"], inp["emb"]
    K = Z.shape[1]
    used = ["confidence"]
    if tta is not None:
        used.append(f"stability ({tta.shape[1]} views)")
    if emb is not None:
        used.append(f"familiarity ({len(inp['bank']):,} reference rows)")
    print(f"{len(df):,} predictions, {K} classes; signals: {', '.join(used)}", flush=True)
    maha = signals.fit_mahalanobis(inp["bank"], inp["bank_labels"], n_classes=K) if emb is not None else None
    X, pred = signal_table(Z, tta, emb, inp["bank"], inp["bank_labels"], maha)
    features = list(X.columns)
    correct = (pred == df.label.to_numpy()).astype(int)
    split = df.split.to_numpy() if "split" in df else split_by_group(df, seed)
    is_clean = (df.condition == "clean").to_numpy() if "condition" in df else np.zeros(len(df), bool)
    tr, va, ca, te = masks = [split == s for s in SPLITS]
    for s, m in zip(SPLITS, masks):
        need(m.any(), f"the {s} split is empty; put some groups in it (or drop the split column to split automatically)")
    errors = int((1 - correct[tr]).sum())
    need(0 < errors < tr.sum(), f"the train split needs both right and wrong predictions (it has {errors:,} wrong of "
         f"{tr.sum():,}): the failure model learns from mistakes")

    model = fit_xgb(X[tr], correct[tr], clean_weights(is_clean[tr]), features=features)
    s = p_correct_score(model, X, features=features)
    iso = fit_isotonic(s[va], correct[va], clean_weights(is_clean[va]))
    p = apply_isotonic(iso, s)
    tau_reject, tau_trust = tau_for(p[ca], correct[ca], TARGET_REJECT), tau_for(p[ca], correct[ca], TARGET_TRUST)

    warns = []
    if ca.sum() < MIN_CAL_ROWS:
        warns.append(f"only {ca.sum():,} rows in cal (about {MIN_CAL_ROWS:,} or more recommended): the cutoffs will be noisy")
    if errors < MIN_TRAIN_ERRORS:
        warns.append(f"only {errors} wrong predictions in train (about {MIN_TRAIN_ERRORS} or more recommended): the failure "
                     "model learns from mistakes, so add more data or harder conditions")
    if np.isinf(tau_trust):
        warns.append(f"no cutoff kept TRUST at or under {TARGET_TRUST:.0%} wrong on cal, so nothing will be TRUSTed; add cal data")
    if np.isinf(tau_reject):
        warns.append(f"no cutoff kept accepted answers at or under {TARGET_REJECT:.0%} wrong on cal, so everything will be REJECTed")

    # Report on test. Raw confidence gets the same decision rule, with its own cutoffs from the same cal rows.
    raw = X.raw_confidence.to_numpy()
    dec = decide(p, tau_reject, tau_trust)
    raw_dec = decide(raw, tau_for(raw[ca], correct[ca], TARGET_REJECT), tau_for(raw[ca], correct[ca], TARGET_TRUST))

    def summary(m):
        c = correct[m]
        tiers = lambda d: {t: {"share": float((d[m] == t).mean()),
                               "error": float(1 - c[d[m] == t].mean()) if (d[m] == t).any() else None} for t in DECISIONS}
        return {"n": int(m.sum()), "accuracy": float(c.mean()),
                "auroc": {"raw_confidence": auroc(raw[m], c), "trust_layer": auroc(s[m], c)},
                "ece": {"raw_confidence": ece(raw[m], c), "trust_layer": ece(p[m], c)},
                "decisions": {"raw_confidence": tiers(raw_dec), "trust_layer": tiers(dec)}}

    report = {"note": "test split only. The trust layer's AUROC ranks by its score before calibration (same order as "
                      "p_correct, without its ties); ECE uses p_correct. Raw confidence's decisions use cutoffs set on the "
                      "same cal rows for the same targets.",
              "rows": {k: int(m.sum()) for k, m in zip(SPLITS, masks)}, "warnings": warns, "test": summary(te)}
    if "condition" in df:
        cond = df.condition.astype(str).to_numpy()
        report["test_by_condition"] = {c: summary(te & (cond == c)) for c in pd.unique(cond[te])}

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(out / "xgb.json")
    (out / "isotonic.json").write_text(json.dumps(iso))
    _write_json(out / "trust_config.json", {
        "features": features, "n_classes": K, "class_names": inp["names"], "n_views": None if tta is None else tta.shape[1],
        "tau_reject": tau_reject, "tau_trust": tau_trust, "targets": {"reject": TARGET_REJECT, "trust": TARGET_TRUST},
        "split_seed": None if "split" in df else seed})
    np.savez(out / "reference.npz", reference=X[tr & (correct == 1)].to_numpy(),
             **({} if emb is None else {"bank": np.asarray(inp["bank"]), "bank_labels": inp["bank_labels"]}))
    _write_json(out / "report.json", report)
    for w in warns:
        print(f"warning: {w}")
    print("cutoffs from cal: " + ", ".join(f"{name} {where} {v:.3f}" if np.isfinite(v) else f"no {name} cutoff"
                                           for name, where, v in (("REJECT", "below", tau_reject), ("TRUST", "at or above", tau_trust))))
    print_report(report)
    print(f"\nsaved the trust layer to {out}/ (report.json has these numbers)")
    return report


def print_report(rep):
    f3 = lambda v: "n/a" if v is None else f"{v:.3f}"
    pc = lambda v: "n/a" if v is None else f"{v:.1%}"
    t, row = rep["test"], "{:52}{:>16}{:>14}".format
    print(f"\nTest split: {t['n']:,} predictions, {pc(t['accuracy'])} of them right.")
    print(row("", "raw confidence", "trust layer"))
    print(row("Tells right from wrong (AUROC, higher = better)", f3(t["auroc"]["raw_confidence"]), f3(t["auroc"]["trust_layer"])))
    print(row("Stated confidence vs reality (ECE, lower = better)", f3(t["ece"]["raw_confidence"]), f3(t["ece"]["trust_layer"])))
    for d in DECISIONS:
        a, b = (t["decisions"][m][d] for m in ("raw_confidence", "trust_layer"))
        print(row(f"{d.upper()}: share of test, share of those wrong", f"{pc(a['share'])}, {pc(a['error'])}", f"{pc(b['share'])}, {pc(b['error'])}"))
    print(f"(both use cutoffs set on cal: TRUST at most {TARGET_TRUST:.0%} wrong, TRUST + CAUTION at most {TARGET_REJECT:.0%} wrong)")
    if "test_by_condition" in rep:
        print(f"\n{'condition':16}{'rows':>9}{'right':>8}{'AUROC raw/trust':>18}{'ECE raw/trust':>16}{'TRUST (wrong)':>17}{'REJECT':>9}")
        for c, r in rep["test_by_condition"].items():
            tl = r["decisions"]["trust_layer"]
            print(f"{c[:15]:16}{r['n']:>9,}{pc(r['accuracy']):>8}"
                  f"{f3(r['auroc']['raw_confidence']) + ' / ' + f3(r['auroc']['trust_layer']):>18}"
                  f"{f3(r['ece']['raw_confidence']) + ' / ' + f3(r['ece']['trust_layer']):>16}"
                  f"{pc(tl['trust']['share']) + ' (' + pc(tl['trust']['error']) + ')':>17}{pc(tl['reject']['share']):>9}")


def load_layer(folder):
    """A layer saved by fit(), ready for score(). Load it once and score many batches."""
    folder = Path(folder)
    need((folder / "trust_config.json").is_file(), f"no trust_config.json in {folder}; point at a folder written by fit")
    cfg = json.loads((folder / "trust_config.json").read_text())
    model = xgb.XGBClassifier()
    model.load_model(folder / "xgb.json")
    ref = np.load(folder / "reference.npz")
    layer = {"config": cfg, "model": model, "iso": json.loads((folder / "isotonic.json").read_text()),
             "reference": pd.DataFrame(ref["reference"], columns=cfg["features"])}
    if "knn_dist" in cfg["features"]:
        layer["bank"], layer["bank_labels"] = ref["bank"], ref["bank_labels"]
        layer["maha"] = signals.fit_mahalanobis(ref["bank"], ref["bank_labels"], n_classes=cfg["n_classes"])
    return layer


def score(layer, logits, tta_logits=None, embeddings=None):
    """p_correct, TRUST / CAUTION / REJECT and reasons for new predictions: logits (N x K), plus TTA logits (N x V x K)
    and embeddings (N x D) if the layer was fit with them (extra inputs are ignored). One row per prediction:
    pred, pred_class (if the layer has class names), raw_confidence, p_correct, decision, reasons (JSON)."""
    cfg = layer["config"]
    K, feats = cfg["n_classes"], cfg["features"]
    Z = np.asarray(logits, np.float64)
    need(Z.ndim == 2 and Z.shape[1] == K, f"logits must be N x {K} (the layer's classes); got {_shape(Z)}")
    need(np.isfinite(Z).all(), "logits have NaN or infinite values")
    tta = emb = None
    if "tta_agree" in feats:
        need(tta_logits is not None, f"this layer uses stability signals: pass tta_logits, N x {cfg['n_views']} x {K}")
        tta = np.asarray(tta_logits)
        need(tta.shape == (len(Z), cfg["n_views"], K) and np.isfinite(tta).all(),
             f"tta_logits must be {len(Z)} x {cfg['n_views']} x {K} (the views the layer was fit on), all finite; got {_shape(tta)}")
    if "knn_dist" in feats:
        need(embeddings is not None, "this layer uses familiarity signals: pass embeddings, N x D")
        emb, D = np.asarray(embeddings), layer["bank"].shape[1]
        need(emb.shape == (len(Z), D) and np.isfinite(emb).all(), f"embeddings must be {len(Z)} x {D}, all finite; got {_shape(emb)}")
    X, pred = signal_table(Z, tta, emb, layer.get("bank"), layer.get("bank_labels"), layer.get("maha"))
    p = apply_isotonic(layer["iso"], p_correct_score(layer["model"], X, features=feats))
    tau = lambda v: np.inf if v is None else v   # null in the config: no cutoff met the target on cal
    d = decide(p, tau(cfg["tau_reject"]), tau(cfg["tau_trust"]))
    res = pd.DataFrame({"pred": pred, "raw_confidence": X.raw_confidence, "p_correct": p, "decision": d,
                        "reasons": reasons(layer["model"], X, d, layer["reference"], feats, cfg["n_views"], GENERIC_WORDS)})
    if cfg["class_names"]:
        res.insert(1, "pred_class", np.asarray(cfg["class_names"])[pred])
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="input folder: predictions.csv and the optional files")
    ap.add_argument("--out", type=Path, required=True, help="fit: the folder to save the layer in; --score: the CSV to write")
    ap.add_argument("--score", type=Path, metavar="LAYER", help="score the predictions in --data with this saved layer instead of fitting")
    args = ap.parse_args()
    try:
        if not args.score:
            fit(args.data, args.out)
            return
        inp = load(args.data, labeled=False)
        res = score(load_layer(args.score), inp["logits"], inp["tta"], inp["emb"])
        res.insert(0, "sample_id", inp["table"].sample_id.to_numpy())
        res.to_csv(args.out, index=False)
        share = res.decision.value_counts(normalize=True)
        print(f"scored {len(res):,} predictions: " + ", ".join(f"{d.upper()} {share.get(d, 0):.1%}" for d in DECISIONS)
              + f"; wrote {args.out}")
    except InputError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
