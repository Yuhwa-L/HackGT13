"""Snap-to-Shop trust layer, end to end: photos -> corrupted versions -> ResNet-18 (subset softmax, 3 TTA views,
embeddings) -> B's signals -> C's temperature scaling, XGBoost + isotonic, cal-split thresholds, SHAP reasons.

Scope cut for the hackathon: one split by base photo (50/15/15/20, stratified by class), severities 1/3/5,
no leave-one-family-out, no bootstrap, one reliability check. Reuses core functions; changes no core files or results.

Run: python -m shop.run_pipeline [--reuse]   (--reuse skips the model and reuses the saved arrays; needs python -m shop.download_data first; ~5-10 min on an M-series GPU)
Writes data/shop/: manifest.csv, features.csv, scores.parquet, embeddings/logits .npy, thresholds.json,
evaluation.json, models/xgb.json, models/isotonic.json.
"""
import xgboost  # noqa: F401  must load before torch: with torch's OpenMP loaded first, XGBoost segfaults on macOS

import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from shop.classes import product_indices
from shop.config import (BANK_VARIANT, BENCHMARK_VARIANTS, CORRUPTIONS, PRODUCT_CLASSES, RAW, SEED, SEVERITIES,
                         SPLITS, TARGET_REJECT, TARGET_TRUST, shop_path)
from shop.corrupt import corrupt
from shop.model import device, forward, load_model, load_photo
from trust import signals
from trust.calibrate import apply_isotonic, fit_isotonic
from trust.evaluate import aurc, auroc, ece
from trust.temperature_scaling import fit_temperature, scaled_confidence, softmax
from trust.thresholds import decide, tau_for
from trust.train_failure_model import FEATURES, clean_weights, fit_xgb, p_correct_score, reasons

N = len(PRODUCT_CLASSES)
VERSIONS = [("clean", 0)] + [(c, s) for c in CORRUPTIONS for s in SEVERITIES]
BATCH = 50
BANK_MIN = 3  # bank photos per class, at least


def list_photos():
    """Benchmark photos and the disjoint reference bank, deduplicated by content (ImageNetV2 variants overlap)."""
    cls_of = {str(i): k for k, i in enumerate(product_indices())}
    rows, seen = [], set()
    for variant in [*BENCHMARK_VARIANTS, BANK_VARIANT]:  # benchmark first: a shared photo stays a benchmark photo
        for p in sorted((RAW / variant).glob("*/*")):
            h = hashlib.md5(p.read_bytes()).hexdigest()
            if h not in seen:
                seen.add(h)
                rows.append(dict(path=str(p), md5=h, label=cls_of[p.parent.name],
                                 role="bank" if variant == BANK_VARIANT else "benchmark"))
    df = pd.DataFrame(rows)
    # Every class needs bank photos for kNN / Mahalanobis. Top up thin classes from the benchmark (disjoint by construction).
    for c in range(N):
        short = BANK_MIN - (df[(df.role == "bank") & (df.label == c)]).shape[0]
        if short > 0:
            df.loc[df[(df.role == "benchmark") & (df.label == c)].sort_values("md5").index[:short], "role"] = "bank"
    return df


def assign_splits(labels, rng):
    """Same scheme as benchmark/make_splits.py: evenly spaced random keys per class, cut at the split fractions."""
    key = np.empty(len(labels))
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        key[idx] = (np.arange(len(idx)) + rng.random()) / len(idx)
    order = np.argsort(key, kind="stable")
    cuts = np.round(np.cumsum(list(SPLITS.values())) * len(labels)).astype(int)
    split = np.empty(len(labels), dtype=object)
    for name, lo, hi in zip(SPLITS, np.r_[0, cuts[:-1]], cuts):
        split[order[lo:hi]] = name
    return split


def build_manifest(photos):
    bench = photos[photos.role == "benchmark"].reset_index(drop=True)
    split = assign_splits(bench.label.to_numpy(), np.random.default_rng(SEED))
    rows = []
    for (_, p), sp in zip(bench.iterrows(), split):
        base = f"prod_{p.md5[:10]}"
        for corr, sev in VERSIONS:
            rows.append(dict(sample_id=f"{base}_{corr}_s{sev}", base_image_id=base, path=p.path,
                             dataset="imagenet_products" if corr == "clean" else "imagenet_products_c",
                             true_label=p.label, true_class=PRODUCT_CLASSES[p.label], corruption=corr,
                             family=CORRUPTIONS.get(corr, "clean"), severity=sev, split=sp))
    m = pd.DataFrame(rows)
    assert m.sample_id.is_unique and (m.groupby("base_image_id").split.nunique() == 1).all()
    return m


def _versions(args):
    path, key = args
    img = load_photo(path)
    return np.stack([corrupt(img, c, s, key) for c, s in VERSIONS])


def run_model(manifest, bank):
    dev, t0 = device(), time.time()
    model = load_model(dev)
    bases = manifest.drop_duplicates("base_image_id")[["path", "base_image_id"]].to_numpy()
    n = len(manifest)
    logits, tta, emb = np.empty((n, N), np.float32), np.empty((n, 3, N), np.float32), np.empty((n, 512), np.float32)
    pos = {sid: i for i, sid in enumerate(manifest.sample_id)}
    with ProcessPoolExecutor() as pool:  # corruptions on CPU workers while the GPU runs the model
        for k, ((_, base), imgs) in enumerate(zip(bases, pool.map(_versions, [tuple(b) for b in bases], chunksize=4))):
            rows = [pos[f"{base}_{c}_s{s}"] for c, s in VERSIONS]
            logits[rows], tta[rows], emb[rows] = forward(model, imgs, dev)
            if k % 100 == 0:
                print(f"  {k}/{len(bases)} photos ({time.time() - t0:.0f}s)")
    bank_imgs = np.stack([load_photo(p) for p in bank.path])
    bank_emb = np.concatenate([forward(model, bank_imgs[i:i + BATCH], dev)[2] for i in range(0, len(bank_imgs), BATCH)])
    print(f"model: {n:,} images + {len(bank):,} bank photos in {time.time() - t0:.0f}s on {dev}")
    return logits, tta, emb, bank_emb


def main():
    photos = list_photos()
    bank = photos[photos.role == "bank"].reset_index(drop=True)
    manifest = build_manifest(photos)
    print(f"{manifest.base_image_id.nunique()} benchmark photos x {len(VERSIONS)} versions = {len(manifest):,} rows; "
          f"bank {len(bank)} photos; classes {N}")
    manifest.drop(columns="path").to_csv(shop_path("manifest.csv"), index=False)

    names = ["logits", "tta_logits", "embeddings", "bank_embeddings"]
    if "--reuse" in sys.argv:  # skip the model: reuse the arrays saved by the last full run
        Z, tta_z, emb, bank_emb = (np.load(shop_path(f"{n}.npy")) for n in names)
        assert len(Z) == len(manifest) and len(bank_emb) == len(bank), "saved arrays don't match; rerun without --reuse"
    else:
        Z, tta_z, emb, bank_emb = run_model(manifest, bank)
        for name, arr in zip(names, (Z, tta_z, emb, bank_emb)):
            np.save(shop_path(f"{name}.npy"), arr)

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
    print(json.dumps({k: ev[k] for k in ("test_accuracy", "auroc", "ece", "decisions_test")}, indent=1, default=float))
    print("thresholds:", thresholds)


if __name__ == "__main__":
    main()
