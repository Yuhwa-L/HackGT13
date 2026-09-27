"""Snap-to-Shop trust layer, end to end: photos -> corrupted versions -> ResNet-18 (subset softmax, 3 TTA views,
embeddings) -> B's signals -> C's temperature scaling, XGBoost + isotonic, cal-split thresholds, SHAP reasons.

Scope cut for the hackathon: one split by base photo (50/15/15/20, stratified by class), severities 1/3/5,
no leave-one-family-out, no bootstrap, one reliability check. Reuses core functions; changes no core files or results.

Run: python -m shop.run_pipeline [--reuse]   (runs the model stage, shop.pipeline_model, in its own torch-only process
first; --reuse skips it and reuses the saved arrays; needs python -m shop.download_data first; ~5-10 min on an M-series GPU)
Writes data/shop/: manifest.csv, features.csv, scores.parquet, embeddings/logits .npy, thresholds.json,
evaluation.json, models/xgb.json, models/isotonic.json.
"""
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from shop.benchmark_data import ARRAY_NAMES as NAMES, N, VERSIONS, build_manifest, list_photos  # noqa: F401
from shop.config import PRODUCT_CLASSES, ROOT, SPLITS, TARGET_REJECT, TARGET_TRUST, shop_path
from trust import signals
from trust.calibrate import apply_isotonic, fit_isotonic
from trust.evaluate import aurc, auroc, ece
from trust.temperature_scaling import fit_temperature, scaled_confidence, softmax
from trust.thresholds import decide, tau_for
from trust.train_failure_model import FEATURES, clean_weights, fit_xgb, p_correct_score, reasons



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

    # Trust layer (C's code): one split, no held-out family
    split = manifest.split.to_numpy()
    is_clean = (manifest.family == "clean").to_numpy()
    tr, va, ca, te = (split == s for s in ("train", "val", "cal", "test"))
    T_clean = fit_temperature(Z[va & is_clean], y_true[va & is_clean])
    T_corr = fit_temperature(Z[va], y_true[va])
    model = fit_xgb(X[tr], correct[tr], clean_weights(is_clean[tr]))
    score = p_correct_score(model, X)
    iso = fit_isotonic(score[va], correct[va], clean_weights(is_clean[va]))
    p = apply_isotonic(iso, score)
    tau_reject, tau_trust = tau_for(p[ca], correct[ca], TARGET_REJECT), tau_for(p[ca], correct[ca], TARGET_TRUST)
    decision = decide(p, tau_reject, tau_trust)
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
    ev = {"note": "Snap-to-Shop, one split by photo, test split only; demo-scale (ImageNetV2 photos), not a benchmark claim",
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
