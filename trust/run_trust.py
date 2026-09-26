"""Owner: C. Entry point: python -m trust.run_trust [data_dir] [--select]
Reads data/manifest.csv, data/prediction_runs.parquet, data/features.csv; runs leave-one-family-out (headline: blur) + CIFAR-10.1.
Writes data/thresholds.json, data/evaluation.json, data/scores.parquet, data/models/.
--select ranks the XGBoost GRID by val AURC (seen families only) and writes nothing.
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from trust.calibrate import apply_isotonic, fit_isotonic
from trust.evaluate import aurc, paired_bootstrap, rc_curve, reliability, summarize
from trust.temperature_scaling import fit_temperature, scaled_confidence
from trust.thresholds import decide, realized, tau_for
from trust.train_failure_model import (CLEAN_SHARE, FEATURES, GRID, SOFTMAX, XGB_PARAMS, clean_weights, fit_lr, fit_xgb,
                                       p_correct_score, reasons)

FAMILIES = ["noise", "blur", "weather", "digital"]
HEADLINE = "blur"
TARGET_REJECT, TARGET_TRUST = 0.05, 0.01
BOOTSTRAP = 1000
LOGITS = [f"logit_{k}" for k in range(10)]
MANIFEST_COLS = ["sample_id", "base_image_id", "dataset", "true_label", "true_class", "corruption", "family", "severity", "split"]
SCORE_COLS = ["sample_id", "base_image_id", "dataset", "corruption", "family", "severity", "true_class", "pred_class",
              "correct", "fold", "raw_confidence", "entropy", "margin", "temp_clean", "temp_corrupted", "p_correct",
              "decision", "reasons"]


def load(data):
    """Join the three input files on sample_id and fail loudly on any contract violation."""
    data = Path(data)
    man = pd.read_csv(data / "manifest.csv")
    pred = pd.read_parquet(data / "prediction_runs.parquet")
    feat = pd.read_csv(data / "features.csv")
    for name, d, cols in [("manifest.csv", man, MANIFEST_COLS),
                          ("prediction_runs.parquet", pred, ["sample_id", "pred_label", "correct"] + LOGITS),
                          ("features.csv", feat, ["sample_id", "failure"] + FEATURES)]:
        missing = sorted(set(cols) - set(d.columns))
        assert not missing, f"{name} is missing columns {missing}"
        assert d.sample_id.is_unique, f"{name} has duplicate sample_ids"
    df = (man[MANIFEST_COLS].merge(pred[["sample_id", "pred_label", "correct"] + LOGITS], on="sample_id")
          .merge(feat[["sample_id", "failure"] + FEATURES], on="sample_id"))
    assert len(df) == len(man) == len(pred) == len(feat), (
        f"sample_ids differ: manifest {len(man)}, predictions {len(pred)}, features {len(feat)}, in all three {len(df)}")
    assert set(df.split) <= {"train", "val", "cal", "test"}, f"unknown split values {set(df.split)}"
    assert set(df.dataset) <= {"cifar10_test", "cifar10c", "cifar10_1"}, f"unknown dataset values {set(df.dataset)}"
    fams = set(df.family[df.dataset != "cifar10_1"]) - {"clean", *FAMILIES}
    assert not fams, f"unknown family values {fams}"
    assert df.groupby("base_image_id").split.nunique().max() == 1, "a base_image_id appears in more than one split"
    assert (df.correct == (df.pred_label == df.true_label)).all(), \
        "correct != (pred_label == true_label): stale or misaligned prediction_runs.parquet"
    assert (df.failure == 1 - df.correct).all(), "features.csv failure != 1 - correct: stale or misaligned features.csv"
    assert not df[FEATURES + LOGITS].isna().any().any(), "NaN in features or logits"
    assert np.allclose(scaled_confidence(df[LOGITS], 1.0), df.raw_confidence, atol=1e-3), \
        "raw_confidence != max softmax(logits): features.csv and prediction_runs.parquet are misaligned"
    return df


def fold_masks(df, heldout):
    """Row masks for one leave-one-family-out fold. CIFAR-10.1 never enters train/val/cal."""
    pool = (df.dataset != "cifar10_1").to_numpy()
    fam, split = df.family.to_numpy(), df.split.to_numpy()
    seen = pool & np.isin(fam, ["clean"] + [f for f in FAMILIES if f != heldout])
    return dict(train=seen & (split == "train"), val=seen & (split == "val"), cal=seen & (split == "cal"),
                heldout_test=pool & (split == "test") & (fam == heldout),
                clean_test=pool & (split == "test") & (fam == "clean"),
                seen_test=seen & (split == "test") & (fam != "clean"))


def run(data="data"):
    t0, data = time.time(), Path(data)
    df = load(data)
    folds = [f for f in FAMILIES if (df.family == f).any()]
    assert HEADLINE in folds, f"no rows for the headline family {HEADLINE}"
    Z, y, X = df[LOGITS].to_numpy(np.float64), df.correct.to_numpy(), df[FEATURES]
    true_label, base, split = df.true_label.to_numpy(), df.base_image_id.to_numpy(), df.split.to_numpy()
    c101 = (df.dataset == "cifar10_1").to_numpy()
    is_clean = (df.family == "clean").to_numpy() & ~c101
    groups = (df.corruption.astype(str) + "_s" + df.severity.astype(str)).to_numpy()
    label_names = dict(zip(df.true_label, df.true_class))
    raw = scaled_confidence(Z, 1.0)
    val_clean = is_clean & (split == "val")
    T_clean = fit_temperature(Z[val_clean], true_label[val_clean])
    temp_clean = scaled_confidence(Z, T_clean)
    (data / "models").mkdir(exist_ok=True)
    ev, scores, thresholds = {"headline_fold": HEADLINE, "T_clean": T_clean, "folds": {}}, [], None

    for F in folds:
        m = fold_masks(df, F)
        tr, va, ca, te = m["train"], m["val"], m["cal"], m["heldout_test"]
        T_corr = fit_temperature(Z[va], true_label[va])
        w_tr = clean_weights(is_clean[tr])
        lr_soft, lr_all = fit_lr(X.loc[tr, SOFTMAX], y[tr], w_tr), fit_lr(X.loc[tr, FEATURES], y[tr], w_tr)
        model = fit_xgb(X[tr], y[tr], w_tr)
        score = p_correct_score(model, X)
        iso = fit_isotonic(score[va], y[va], clean_weights(is_clean[va]))
        p = apply_isotonic(iso, score)
        temp_corr, tta = scaled_confidence(Z, T_corr), df.tta_pconf.to_numpy()
        ls, la = lr_soft.predict_proba(X[SOFTMAX])[:, 1], lr_all.predict_proba(X[FEATURES])[:, 1]
        methods = {  # name: (ranking score, probability)
            "raw_confidence": (raw, raw), "temp_clean": (temp_clean, temp_clean), "temp_corrupted": (temp_corr, temp_corr),
            "tta_only": (tta, tta), "lr_softmax_only": (ls, ls), "lr_all": (la, la),
            "trust_layer": (score, p),   # ranks on the pre-isotonic score: same order as p_correct, without isotonic's ties
        }

        def report(mask):
            if not mask.any():
                return None
            return dict(n=int(mask.sum()), accuracy=float(y[mask].mean()),
                        methods={k: summarize(r[mask], pr[mask], y[mask], groups[mask]) for k, (r, pr) in methods.items()})

        def bootstrap(mask):
            return {f"trust_minus_{k}": paired_bootstrap(score[mask], methods[k][0][mask], y[mask], base[mask], BOOTSTRAP)
                    for k in ("raw_confidence", "temp_corrupted", "tta_only")}

        tau_reject, tau_trust = tau_for(p[ca], y[ca], TARGET_REJECT), tau_for(p[ca], y[ca], TARGET_TRUST)
        cal_clean = ca & is_clean
        fold = dict(
            T_corrupted=T_corr, n_rows=dict(train=int(tr.sum()), val=int(va.sum()), cal=int(ca.sum())),
            thresholds=dict(tau_reject=tau_reject, tau_trust=tau_trust),
            heldout_test=report(te), clean_test=report(m["clean_test"]), seen_test=report(m["seen_test"]),
            broken_promise_heldout={
                "raw_conf_rule_set_on_clean_cal": realized(raw[te], y[te], tau_for(raw[cal_clean], y[cal_clean], TARGET_REJECT), TARGET_REJECT),
                "raw_conf_rule_set_on_seen_cal": realized(raw[te], y[te], tau_for(raw[ca], y[ca], TARGET_REJECT), TARGET_REJECT),
                "trust_layer_reject_rule": realized(p[te], y[te], tau_reject, TARGET_REJECT),
                "trust_layer_trust_rule": realized(p[te], y[te], tau_trust, TARGET_TRUST),
            },
            bootstrap_heldout=bootstrap(te))
        rows = te.copy()   # scores.parquet: each corrupted test row comes from the fold that never saw its family
        if F == HEADLINE:
            sev = df.severity.to_numpy()
            chart = [(0, m["clean_test"])] + [(s, te & (sev == s)) for s in range(1, 6)]
            fold["headline_chart"] = [dict(severity=s, n=int(mm.sum()), accuracy=float(y[mm].mean()),
                                           **{k: float(methods[k][1][mm].mean()) for k in ("raw_confidence", "temp_clean", "temp_corrupted")},
                                           p_correct=float(p[mm].mean())) for s, mm in chart if mm.any()]
            fold["reliability_heldout"] = {k: reliability(methods[k][1][te], y[te])
                                           for k in ("raw_confidence", "temp_clean", "temp_corrupted", "trust_layer")}
            fold["risk_coverage_heldout"] = {k: rc_curve(methods[k][0][te], y[te])
                                             for k in ("raw_confidence", "temp_corrupted", "tta_only", "trust_layer")}
            ev["cifar10_1"] = dict(report(c101), bootstrap=bootstrap(c101)) if c101.any() else None
            thresholds = dict(fold=F, tau_reject=tau_reject, tau_trust=tau_trust,
                              target_error=dict(reject=TARGET_REJECT, trust=TARGET_TRUST), T_clean=T_clean, T_corrupted=T_corr)
            rows |= m["clean_test"] | c101
        ev["folds"][F] = fold
        model.save_model(data / "models" / f"xgb_{F}.json")
        (data / "models" / f"isotonic_{F}.json").write_text(json.dumps(iso))

        decision = decide(p[rows], tau_reject, tau_trust)
        sc = df.loc[rows, ["sample_id", "base_image_id", "dataset", "corruption", "family", "severity", "true_class",
                           "correct", "entropy", "margin"]].reset_index(drop=True)
        sc["pred_class"] = df.pred_label[rows].map(label_names).to_numpy()
        sc["fold"] = F
        for k, v in (("raw_confidence", raw), ("temp_clean", temp_clean), ("temp_corrupted", temp_corr), ("p_correct", p)):
            sc[k] = v[rows]
        sc["decision"] = decision
        sc["reasons"] = reasons(model, X[rows], decision, X[tr & (y == 1)])
        scores.append(sc[SCORE_COLS])
        print(f"fold {F} done ({time.time() - t0:.0f}s)")

    pool = ~c101
    meta = dict(created=datetime.now(timezone.utc).isoformat(timespec="seconds"), n_rows=len(df),
                base_images={s: int(df.base_image_id[pool & (split == s)].nunique()) for s in ("train", "val", "cal", "test")},
                cifar10_1_images=int(c101.sum()), features=FEATURES, xgb_params=XGB_PARAMS, clean_weight_share=CLEAN_SHARE,
                targets=dict(reject=TARGET_REJECT, trust=TARGET_TRUST), bootstrap_resamples=BOOTSTRAP,
                note="trust_layer ranking metrics (aurc, auroc, err_at_*) use the pre-isotonic score; ece and mean_conf use p_correct")
    ev = {"meta": meta, **ev}
    _write_json(data / "evaluation.json", ev)
    _write_json(data / "thresholds.json", thresholds)
    pd.concat(scores, ignore_index=True).to_parquet(data / "scores.parquet", index=False)
    _print_summary(ev, time.time() - t0)
    return ev


def select(data="data"):
    """Rank the XGBoost GRID by mean val AURC over folds (seen families only; test rows untouched). Writes nothing."""
    df = load(data)
    y, X = df.correct.to_numpy(), df[FEATURES]
    is_clean = ((df.family == "clean") & (df.dataset != "cifar10_1")).to_numpy()
    folds = [f for f in FAMILIES if (df.family == f).any()]
    results = []
    for params in GRID:
        vals = []
        for F in folds:
            m = fold_masks(df, F)
            model = fit_xgb(X[m["train"]], y[m["train"]], clean_weights(is_clean[m["train"]]), params)
            vals.append(aurc(p_correct_score(model, X[m["val"]]), y[m["val"]]))
        results.append((float(np.mean(vals)), params))
        print(f"max_depth={params['max_depth']} n_estimators={params['n_estimators']}: mean val AURC {results[-1][0]:.5f}")
    best = min(results, key=lambda r: r[0])[1]
    print(f"best: max_depth={best['max_depth']} n_estimators={best['n_estimators']} (set XGB_PARAMS in trust/train_failure_model.py)")


def _finite(x):
    """inf/nan -> None, so the JSON stays valid for JavaScript (an unreachable threshold is written as null)."""
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _write_json(path, obj):
    path.write_text(json.dumps(_finite(obj), indent=1, allow_nan=False))


def _print_summary(ev, seconds):
    f3 = lambda v: "n/a" if v is None else f"{v:.3f}"
    ci = lambda c: "n/a" if c is None else f"[{c[0]:+.3f}, {c[1]:+.3f}]"
    print(f"\n{ev['meta']['n_rows']:,} rows in {seconds:.0f}s | T_clean {ev['T_clean']:.3f}")
    for F, fo in ev["folds"].items():
        h, t = fo["heldout_test"], fo["thresholds"]
        mt, b = h["methods"], fo["bootstrap_heldout"]["trust_minus_raw_confidence"]
        bp = fo["broken_promise_heldout"]["raw_conf_rule_set_on_clean_cal"]
        print(f"[{F}{'*' if F == ev['headline_fold'] else ''}] held-out acc {h['accuracy']:.3f} | "
              f"AUROC raw {f3(mt['raw_confidence']['auroc'])} tta {f3(mt['tta_only']['auroc'])} trust {f3(mt['trust_layer']['auroc'])} "
              f"(trust-raw {ci(b['auroc_diff_ci'])}) | ECE raw {f3(mt['raw_confidence']['ece'])} "
              f"T_corr {f3(mt['temp_corrupted']['ece'])} trust {f3(mt['trust_layer']['ece'])} | "
              f"tau {f3(t['tau_reject'])}/{f3(t['tau_trust'])} | raw 5% rule from clean cal -> {f3(bp['error'])} error")
    if ev.get("cifar10_1"):
        mt = ev["cifar10_1"]["methods"]
        print(f"[cifar10_1] acc {ev['cifar10_1']['accuracy']:.3f} | AUROC raw {f3(mt['raw_confidence']['auroc'])} "
              f"trust {f3(mt['trust_layer']['auroc'])} | ECE raw {f3(mt['raw_confidence']['ece'])} "
              f"T_clean {f3(mt['temp_clean']['ece'])} trust {f3(mt['trust_layer']['ece'])}")


if __name__ == "__main__":
    args = sys.argv[1:]
    data_dir = next((a for a in args if not a.startswith("--")), "data")
    if "--select" in args:
        select(data_dir)
    else:
        run(data_dir)
