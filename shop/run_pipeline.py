"""Snap-to-Shop trust layer, end to end: photos -> corrupted versions -> ViT-B/16 (subset softmax, 3 TTA views,
embeddings) -> B's signals -> C's temperature scaling, XGBoost + isotonic, cross-fitted thresholds, SHAP reasons.

Scope cut for the hackathon: test = 20% of photos (split by photo, stratified by class), the rest cross-fitted
(5 folds); severities 1/3/5; no leave-one-family-out, no bootstrap, one reliability check. Reuses core functions; changes no core files or results.

Run: python -m shop.run_pipeline [--reuse]   (runs the model stage, shop.pipeline_model, in its own torch-only process
first; --reuse skips it and reuses the saved arrays; needs python -m shop.download_data first;
~20 min on an M3 MacBook, mostly the ViT; --reuse takes seconds)
Writes data/shop/: manifest.csv, features.csv, scores.parquet, embeddings/logits .npy, thresholds.json,
evaluation.json, models/xgb.json, models/isotonic.json.
"""
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from shop.benchmark_data import ARRAY_NAMES as NAMES, N, VERSIONS, build_manifest, list_photos  # noqa: F401
from shop.config import PRODUCT_CLASSES, ROOT, SEED, SPLITS, TARGET_REJECT, TARGET_TRUST, shop_path
from trust import signals
from trust.calibrate import apply_isotonic, fit_isotonic
from trust.evaluate import aurc, auroc, ece
from trust.temperature_scaling import fit_temperature, scaled_confidence, softmax
from trust.thresholds import decide, tau_for
from trust.train_failure_model import FEATURES, clean_weights, fit_xgb, p_correct_score, reasons



K_FOLDS = 5


def photo_folds(manifest, mask, k, seed=SEED):
    """Fold index per row: whole photos go to one fold, stratified by class; -1 outside `mask`."""
    bases = manifest[mask].drop_duplicates("base_image_id")[["base_image_id", "true_label"]]
    rng, fold_of = np.random.default_rng(seed), {}
    for _, g in bases.groupby("true_label"):
        for j, b in enumerate(rng.permutation(g.base_image_id.to_numpy())):
            fold_of[b] = j % k
    return manifest.base_image_id.map(fold_of).fillna(-1).astype(int).to_numpy()


def export_live_reference(bank_emb, bank_labels, reference):
    """What live upload scoring needs besides the committed models: the reference bank and the feature rows of
    correct training predictions (for the reasons' percentiles). Small, committed: data/shop/live_reference.npz."""
    np.savez_compressed(shop_path("live_reference.npz"), bank_emb=bank_emb.astype(np.float32), bank_labels=bank_labels,
                        reference=reference[FEATURES].to_numpy(np.float32), features=np.array(FEATURES))


def main():
    photos = list_photos()
    bank = photos[photos.role == "bank"].reset_index(drop=True)
    manifest = build_manifest(photos)
    print(f"{manifest.base_image_id.nunique()} benchmark photos x {len(VERSIONS)} versions = {len(manifest):,} rows; "
          f"bank {len(bank)} photos; classes {N}")
    manifest.drop(columns="path").to_csv(shop_path("manifest.csv"), index=False)

    if "--reuse" not in sys.argv:  # model stage in its own torch-only process (no XGBoost there)
        subprocess.run([sys.executable, "-W", "ignore", "-m", "shop.pipeline_model"], cwd=ROOT, check=True)
    Z, tta_z, emb, bank_emb = (np.load(shop_path(f"{n}.npy")) for n in NAMES)
    assert len(Z) == len(manifest) and len(bank_emb) == len(bank), "saved arrays don't match; rerun without --reuse"

    # Signals (B's code, 30 classes)
    y_true, bank_y = manifest.true_label.to_numpy(), bank.label.to_numpy()
    sm = signals.softmax_stats(Z)
    pred = sm["pred"]
    correct = (pred == y_true).astype(int)
    means, prec = signals.fit_mahalanobis(bank_emb, bank_y, n_classes=N)
    X = pd.DataFrame({**{k: sm[k] for k in ("raw_confidence", "entropy", "margin")}, **signals.tta_stats(Z, tta_z),
                      **signals.bank_signals(emb, pred, bank_emb, bank_y, n_classes=N),
                      "maha_pred": signals.maha_pred(emb, pred, means, prec)})
    pd.concat([manifest[["sample_id"]], X, pd.Series(1 - correct, name="failure")], axis=1).to_csv(
        shop_path("features.csv"), index=False, float_format="%.10g")

    # Trust layer (C's code), cross-fitted. Only 380 non-test photos exist, so a fixed train/val/cal split leaves 71
    # photos to set each threshold and they swing between splits. Instead: K folds over all non-test photos (by photo,
    # stratified by class); isotonic calibration and both thresholds come from out-of-fold scores; the final model
    # is fit on all non-test photos. The 95 test photos are never used for fitting anything.
    split = manifest.split.to_numpy()
    is_clean = (manifest.family == "clean").to_numpy()
    te, fit = split == "test", split != "test"
    fold = photo_folds(manifest, fit, K_FOLDS)
    T_clean = fit_temperature(Z[fit & is_clean], y_true[fit & is_clean])
    T_corr = fit_temperature(Z[fit], y_true[fit])
    oof = np.full(len(X), np.nan)
    for k in range(K_FOLDS):
        trk, hold = fit & (fold != k), fit & (fold == k)
        oof[hold] = p_correct_score(fit_xgb(X[trk], correct[trk], clean_weights(is_clean[trk])), X[hold])
    iso = fit_isotonic(oof[fit], correct[fit], clean_weights(is_clean[fit]))
    p_oof = apply_isotonic(iso, oof[fit])
    tau_reject, tau_trust = tau_for(p_oof, correct[fit], TARGET_REJECT), tau_for(p_oof, correct[fit], TARGET_TRUST)
    model = fit_xgb(X[fit], correct[fit], clean_weights(is_clean[fit]))
    score = p_correct_score(model, X)
    p = apply_isotonic(iso, score)
    decision = decide(p, tau_reject, tau_trust)
    tr = fit  # reference rows for the reasons' percentiles and the live reference
    why = reasons(model, X, decision, X[tr & (correct == 1)])

    probs = softmax(Z.astype(np.float64))
    top = np.argsort(-probs, axis=1)[:, :3]
    cands = [json.dumps([{"class": PRODUCT_CLASSES[k], "prob": round(float(probs[i, k]), 4)}
                         for j, k in enumerate(top[i]) if j == 0 or probs[i, k] >= 0.05]) for i in range(len(Z))]
    scores = manifest.drop(columns="path").assign(
        pred_label=pred, pred_class=[PRODUCT_CLASSES[k] for k in pred], correct=correct,
        raw_confidence=sm["raw_confidence"], entropy=sm["entropy"], margin=sm["margin"],
        temp_clean=scaled_confidence(Z, T_clean), temp_corrupted=scaled_confidence(Z, T_corr), p_correct=p,
        decision=decision, reasons=why, candidates=cands)
    scores.to_parquet(shop_path("scores.parquet"), index=False)

    # One reliability check on the test split
    raw = scores.raw_confidence.to_numpy()
    by_dec = {d: dict(share=float((decision[te] == d).mean()), error=float(1 - correct[te][decision[te] == d].mean())
                      if (decision[te] == d).any() else None) for d in ("trust", "caution", "reject")}
    by_sev = [dict(severity=int(s), n=int(m.sum()), accuracy=float(correct[m].mean()), raw_confidence=float(raw[m].mean()),
                   p_correct=float(p[m].mean())) for s in sorted(set(manifest.severity)) for m in [te & (manifest.severity.to_numpy() == s)]]
    ev = {"note": "Snap-to-Shop, test split only; demo-scale (ImageNetV2 photos), not a benchmark claim",
          "calibration": f"{K_FOLDS}-fold cross-fit over the {int(manifest[fit].base_image_id.nunique())} non-test photos: "
                         "isotonic and thresholds from out-of-fold scores; final model on all non-test photos",
          "classes": N, "photos": {s: int((manifest.drop_duplicates("base_image_id").split == s).sum()) for s in SPLITS},
          "bank_photos": len(bank), "rows": len(manifest), "test_rows": int(te.sum()),
          "test_accuracy": float(correct[te].mean()), "T_clean": T_clean, "T_corrupted": T_corr,
          "auroc": {"raw_confidence": auroc(raw[te], correct[te]), "trust_layer": auroc(score[te], correct[te])},
          "aurc": {"raw_confidence": aurc(raw[te], correct[te]), "trust_layer": aurc(score[te], correct[te])},
          "ece": {"raw_confidence": ece(raw[te], correct[te]), "temp_clean": ece(scores.temp_clean.to_numpy()[te], correct[te]),
                  "temp_corrupted": ece(scores.temp_corrupted.to_numpy()[te], correct[te]), "p_correct": ece(p[te], correct[te])},
          "decisions_test": by_dec, "by_severity_test": by_sev}
    thresholds = {"tau_reject": tau_reject, "tau_trust": tau_trust,
                  "target_error": {"reject": TARGET_REJECT, "trust": TARGET_TRUST}, "T_clean": T_clean, "T_corrupted": T_corr}
    shop_path("models").mkdir(parents=True, exist_ok=True)
    model.save_model(shop_path("models", "xgb.json"))
    shop_path("models", "isotonic.json").write_text(json.dumps(iso))
    shop_path("thresholds.json").write_text(json.dumps(thresholds, indent=1))
    shop_path("evaluation.json").write_text(json.dumps(ev, indent=1, default=float))
    export_live_reference(bank_emb, bank_y, X[tr & (correct == 1)])
    print(json.dumps({k: ev[k] for k in ("test_accuracy", "auroc", "ece", "decisions_test")}, indent=1, default=float))
    print("thresholds:", thresholds)


if __name__ == "__main__":
    main()
