# Person C: Trust Layer, Calibration, Thresholds, Metrics

Scope: C's work only. A, B and D appear here only as the files C reads from them or hands to them.

## 1. Verdict: the data pipeline works (checked on real data, Sat 26 Sep)

I ran the whole pipeline end to end on real data. The test used 2,000 CIFAR-10 test base images × (clean + 8 CIFAR-10-C types × 5 severities), plus all 2,000 CIFAR-10.1 images, for 84,000 rows. It covered: manifest, then ResNet-18 with 3 TTA views, then signals, then the C stage across all 4 held-out-family folds.

| Check | Result |
|---|---|
| Doc's checkpoint snippet (timm + 3×3 stem + no maxpool), strict load | OK: 122 keys, all matched |
| Clean CIFAR-10 test accuracy | **94.98%**, exactly as the doc states |
| `forward_head(pre_logits=True)` → `fc` equals `model(x)` | OK |
| CIFAR-10-C layout: row i = test image `i % 10000`, severity `i // 10000 + 1` | OK: `labels.npy == tile(test_labels, 5)`, and every sampled corrupted row matches its own clean image |
| CIFAR-10.1 v6 | OK: 2,000 images, 200 per class, ResNet accuracy 88.75% |
| Inference speed on this M1 Pro (MPS) | 84k images × 4 views took 131 s, so the full 410k benchmark is about 11 min. The 50k train bank took 20 s |
| C stage: 4 folds, bootstrap, all exports | **19 s** at 84k rows (about 2 min estimated at 410k) |
| XGBoost on this Mac | **Fails to import** until you run `brew install libomp` |
| Doc's `torch.hub.load_state_dict_from_url` on this Mac's python.org Python 3.13 | **Fails** with `CERTIFICATE_VERIFY_FAILED`. Workaround: `curl -L -o resnet18_cifar10.pth <url>`, then `torch.load` |

**Verdict:** the design holds and there is no leakage in the split logic. You need to fix the 5 contract gaps in §3 and follow the 7 implementation rules in §7. Otherwise some reported numbers will be quietly wrong.

## 2. C's part of the pipeline

```
manifest.csv (B) ──────────┐
prediction_runs.parquet (A)┼── join on sample_id (must be 1:1) + contract asserts
features.csv (B) ──────────┘
        │
        ├── for each held-out family F in {noise, blur, weather, digital}   (seen = clean + the other 3)
        │     train(seen) → LR (softmax-only), LR (all), XGBoost (all); clean rows = 25% of total weight
        │     val(seen)   → T_clean (val clean rows, fit once), T_corrupted, isotonic → p_correct
        │     cal(seen)   → τ_reject (5% error), τ_trust (1% error), raw-confidence rules
        │     test(F)     → metrics + bootstrap + per-row scores     ← headline = blur
        │     test(clean), test(seen families) → secondary tables
        └── CIFAR-10.1 (all 2,000 rows, never trained on) → scored by the headline-fold model
        ↓
thresholds.json   evaluation.json   scores.parquet   models/xgb_<F>.json   models/isotonic_<F>.json
```

C needs no torch and no GPU. Dependencies: `numpy pandas pyarrow scikit-learn xgboost scipy`. You don't need the `shap` package, because XGBoost's `pred_contribs=True` already computes TreeSHAP.

## 3. Contract gaps to settle with A and B now

1. **Logits:** `prediction_runs.parquet` must include flat `logit_0 … logit_9` columns (float32 is fine). Temperature scaling needs the raw logits, and `features.csv` only has scalar features.
2. **CIFAR-10.1 rows:** set `dataset=cifar10_1`, `family=natural`, `split=test`. If B stratifies CIFAR-10.1 into train/val/cal like other base images, or tags it `family=clean`, the `split=="train" & family in seen` filter will quietly train on the natural-shift test set. C also filters out `dataset=="cifar10_1"` itself as a second guard.
3. **Train-bank embeddings have no owner.** kNN and Mahalanobis need embeddings for the 50k CIFAR-10 train images, but the manifest doesn't list those images, so they're not in A's run list. A should also export `train_embeddings.npy` and `train_labels.npy`. It takes about 20 s.
4. **Pin down the TTA feature definitions.** The doc doesn't say which axis the "±2 px shift" uses. The prototype used horizontal shifts with reflect padding. Also fix `tta_pconf` as the mean over the 3 TTA views, and `tta_std` as the standard deviation over the original plus the 3 TTA views.
5. **Float32 softmax causes ties.** Clean images have only 5,532 unique float32 `raw_confidence` values out of 10,000, and ties distort AURC. C recomputes baseline confidences from the logits in float64.

`features.csv` columns that C expects: `sample_id, raw_confidence, entropy, margin, tta_agree, tta_pconf, tta_std, knn_dist, maha_pred, failure`.

## 4. File structure (files C owns)

```
ml-reliability-lab/
├── data/
│   ├── manifest.csv              ← B  (read)
│   ├── prediction_runs.parquet   ← A  (read; needs logit_0..logit_9)
│   ├── features.csv              ← B  (read)
│   ├── thresholds.json           → C  (headline fold τs + temperatures)
│   ├── evaluation.json           → C  (everything D charts)
│   ├── scores.parquet            → C  (per-row results; D builds demo_cache.json from it)
│   └── models/                   → C
│       ├── xgb_<fold>.json       # XGBoost native JSON
│       └── isotonic_<fold>.json  # {"x": [...], "y": [...]}; p_correct = np.interp(score, x, y), verified identical to sklearn
└── trust/                        # features.py and signals.py belong to B
    ├── temperature_scaling.py    # C2  fit_T(logits, labels) → T; conf_T(logits, T)
    ├── train_failure_model.py    # C3, C7  FEATURES, clean weights, LR/XGB fit, grouped-SHAP reasons()
    ├── calibrate.py              # C3  isotonic fit → {x, y} JSON; apply = np.interp
    ├── thresholds.py             # C4  tau_for(conf, correct, target), decide(), broken-promise rows
    ├── evaluate.py               # C5  aurc, auroc (+ within-group), ece, reliability, err_at, rc_curve, boot_diff; __main__ self-check
    └── run_trust.py              # C1, C6  load + asserts, fold loop, writes all C outputs
```

Run everything with `python -m trust.run_trust`. There is no config system; constants go at the top of `run_trust.py`. There is no live-model code either: the demo and the backend read `scores.parquet`. Add a `predict.py` only if live `/predict` actually happens.

## 5. C's tasks, split up so each can be built and checked separately

| # | Task | File | Done when |
|---|---|---|---|
| **C1** | Load and check the contract: join the 3 files on `sample_id` (1:1), then assert: no `base_image_id` in two splits; CIFAR-10.1 only in `test`; `failure == 1 - correct`; no NaN in features; per fold, no train base image in val/cal/test | `run_trust.py` | Asserts pass on the mock files, then on the first real files |
| **C2** | Temperature baselines. `T_clean` is fit on val clean rows, once. `T_corrupted` is fit per fold on val rows from clean + seen families. Minimize NLL over log T with `scipy.optimize.minimize_scalar` | `temperature_scaling.py` | Both T > 1 (smoke: 1.57 and 2.03–2.46). On held-out blur, ECE ranks raw > T_clean > T_corrupted |
| **C3** | Failure model. Label = `failure`. Fit LR (softmax-only), LR (all features) and XGBoost (all features; fixed `max_depth=4, n_estimators=300, lr=0.05, subsample=0.8, colsample=0.8`). Clean rows get `w = n_corrupt / (3 * n_clean)`, which gives them 25% of total weight. Fit isotonic on val with the same weights and export it as `{x, y}` | `train_failure_model.py`, `calibrate.py` | Sanity check: LR softmax-only ≈ raw confidence on AUROC. XGBoost beats raw on held-out AUROC |
| **C4** | Decisions. On **cal only**, set τ_reject as the lowest threshold (so the most predictions accepted) whose error among accepted rows is ≤ 5%. Set τ_trust the same way for ≤ 1%. Cut only at tie boundaries; if no threshold meets the target, return `inf`. Broken-promise table: for the raw-confidence rule set on **clean** cal, the raw-confidence rule set on **seen** cal, and both trust-layer rules, report actual coverage and error on the held-out test set | `thresholds.py` | TRUST, CAUTION and REJECT all occur on held-out test, and the actual errors are logged |
| **C5** | Metrics. AURC (tie-aware); AUROC pooled and within each corruption × severity group (skip single-class groups); ECE with 15 bins plus reliability bins; error at 20% and 50% coverage; risk–coverage curve with 50 points; paired bootstrap over **base images** (1,000 resamples) for XGB−raw, XGB−T_corrupted and XGB−TTA-only | `evaluate.py` | `python trust/evaluate.py` self-check passes: all-correct → AURC 0; conf=[.9,.8,.7,.1] with correct=[1,1,1,0] → AURC 0.0625, and the reverse order → 0.521; all tied → overall error; ECE is 0 when conf equals correctness; unreachable target → `inf` |
| **C6** | Fold runner and exports. Leave-one-family-out ×4 (headline = blur), plus CIFAR-10.1, clean test and seen-family test. Write thresholds.json, evaluation.json, scores.parquet and models/ (schemas in §6) | `run_trust.py` | All outputs load. scores.parquet has one row per test sample. D confirms the schema |
| **C7** | SHAP reasons. See rule 3 in §7 | `train_failure_model.py` | The 3 demo-story rows (clean TRUST, moderate CAUTION, heavy REJECT) produce sensible text |
| C8 (stretch) | Split-conformal sets for CAUTION on cal; the Geifman–El-Yaniv bound on τ; a ConfidNet head | — | Only once C1–C7 are frozen |

MUST (from the doc's MVP list): C1–C4, C5 without bootstrap, and C6 for the headline fold. NICE: the other 3 folds, CIFAR-10.1, bootstrap, C7.

**Model selection (7–9 PM, val only):** try a small grid, `max_depth ∈ {3,4,6}` × `n_estimators ∈ {200,400}`. Choose by val AURC on seen families only and use one config for every fold. Skip the image-quality stats: the doc's proxy run showed they hurt on unseen families.

## 6. Output schemas (share with D)

**thresholds.json** (headline fold):
```json
{"fold": "blur", "tau_reject": 0.7512, "tau_trust": 0.9539,
 "target_error": {"reject": 0.05, "trust": 0.01}, "T_clean": 1.567, "T_corrupted": 2.377}
```

**evaluation.json**:
```
headline_fold, T_clean
folds.<family>:
  T_corrupted
  heldout_test | clean_test | seen_test: {n, accuracy, <method>: {aurc, auroc, auroc_within, ece, mean_conf, err_at_20, err_at_50}}
  thresholds: {tau_reject, tau_trust}
  broken_promise_heldout.<rule>: {tau, coverage, error}
  bootstrap: {xgb_minus_raw | xgb_minus_temp_corrupted | xgb_minus_tta: {aurc_diff_ci, auroc_diff_ci}}
  headline fold only:
    headline_chart: [{severity 0-5, n, accuracy, raw_confidence, temp_clean, temp_corrupted, p_correct}]   # severity 0 = clean test
    reliability_heldout.<method>: [{lo, hi, n, conf, acc}]
    risk_coverage_heldout.<method>: [[coverage, risk], ...]
cifar10_1: same shape as heldout_test
methods: raw_confidence, temp_clean, temp_corrupted, tta_pconf_only, lr_softmax_only, lr_all, xgb_score, p_correct
```
Size in the smoke run: 54 KB.

**scores.parquet** has one row per test sample. Each corrupted row is scored by **the fold model that never saw its family**. Clean and CIFAR-10.1 rows are scored by the headline model.
`sample_id, base_image_id, dataset, corruption, family, severity, true_class, pred_class, correct, fold, raw_confidence, entropy, margin, temp_clean, temp_corrupted, p_correct, decision, reasons` (reasons is a JSON string of `[{signal, text, shap}]`).
Each row maps one-to-one onto the §11 `/predict` response, so demo_cache.json is just a filter of this file on the ~30 demo base images. Those images must come from the test split.

## 7. Implementation rules from the smoke run

1. **Compute ranking metrics on the pre-isotonic XGB score, not on p_correct.** Isotonic calibration collapsed 4,150 test rows into only 141 distinct p_correct values, and the ties hurt AURC. With noise held out, AURC was 0.2756 for the XGB score, 0.2807 for raw confidence and 0.2838 for p_correct: the tied version lost to the baseline it actually beats. The ordering is the same because isotonic is monotone. Use p_correct only for ECE, thresholds and display.
2. **Every table needs the TTA-only baseline.** It ranks failures as well as XGBoost. On held-out blur, AUROC was 0.887 for TTA-only and 0.886 for XGBoost, and the paired-bootstrap difference CI was [−0.008, +0.008]. What XGBoost adds is calibration (ECE 0.085 → 0.020) and one number that means something. It does not rank better. Pitch it that way.
3. **Per-feature SHAP reasons are misleading.** `raw_confidence`, `entropy` and `margin` are nearly functions of one another, so SHAP splits credit between them with arbitrary signs. In the smoke run, a p=0.99 TRUST image was given "Top two classes are close (margin 1.00)". Fix: sum SHAP within 3 groups (SHAP is additive, so group sums are exact): **stability** = tta_*, **familiarity** = knn_dist + maha_pred, **confidence** = raw_confidence + entropy + margin. Show a group only if its sum is ≥ 0.1, and only for CAUTION or REJECT. Put the actual value and its percentile among correct train predictions in the text, for example "class prob under flips/shifts 94% (lower than 82% of correct predictions)".
4. **The broken promise is only broken for the rule set on clean data.** On held-out blur, the raw-confidence 5% rule set on clean cal had **17.2%** actual error. The same rule set on seen-corrupted cal had 4.2% (kept). The trust layer had 4.3% at slightly higher coverage (66.0% vs 63.8%). Report all three.
5. **Noise is the weak fold.** Accuracy is 49%. XGBoost vs raw AUROC is +0.003 with a CI that crosses 0, and p_correct is still overconfident (mean 0.67 vs accuracy 0.49, ECE 0.18). Keep blur as the headline and show noise in the fold table.
6. **Clean weighting helps a little, so keep it.** Mean clean p_correct went from 0.919 to 0.927 (accuracy 0.952), and held-out ECE went from 0.026 to 0.020. The layer is still conservative on clean data: it REJECTs 11.8% of clean images while only 4.8% are wrong.
7. **TRUST's 1% target is slightly missed under unseen shift** (1.6% on blur, 7.2% on noise). Report the actual numbers and don't claim "≤1%".

## 8. Smoke-run numbers (proxy only; the pitch uses the final full run)

Held-out **blur** test (4,150 rows, ResNet accuracy 81.0%):

| Method | AURC ↓ | AUROC | Within-group AUROC | ECE | Error @20% coverage |
|---|---|---|---|---|---|
| Raw confidence | 0.0495 | 0.869 | 0.853 | 0.126 | 0.8% |
| Temp-scaled (clean val) | 0.0496 | 0.870 | 0.852 | 0.077 | 0.8% |
| Temp-scaled (corrupted val) | 0.0501 | 0.869 | 0.850 | 0.039 | 1.0% |
| TTA-only (tta_pconf) | 0.0440 | 0.887 | 0.878 | 0.085 | 0.4% |
| LR, softmax-only | 0.0499 | 0.866 | 0.849 | 0.061 | 0.8% |
| LR, all features | 0.0457 | 0.880 | 0.866 | 0.033 | 0.1% |
| **XGBoost + isotonic** | **0.0437** | **0.886** | **0.873** | **0.020** | **0.0%** |

XGBoost − raw, paired bootstrap over base images: AURC [−0.011, −0.002], AUROC [+0.008, +0.027] (both significant). By fold, the AUROC gain over raw is: noise +0.003 (not significant), blur +0.017, weather +0.019, digital +0.017. CIFAR-10.1: AUROC 0.892 vs raw 0.884; ECE: p_correct 0.040, T_clean 0.034, raw 0.069.

Headline chart, blur held out:

| Severity | 0 (clean) | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Accuracy | 95.2 | 93.0 | 89.4 | 83.7 | 78.1 | **60.8** |
| Raw confidence | 97.8 | 97.3 | 96.1 | 93.5 | 92.6 | **88.7** |
| Temp-scaled (clean) | 95.7 | 94.5 | 92.6 | 88.6 | 86.8 | 80.6 |
| Temp-scaled (corrupted) | 88.4 | 86.2 | 83.4 | 78.3 | 75.5 | 67.5 |
| p_correct | 92.7 | 89.6 | 86.2 | 80.6 | 77.0 | **67.2** |

Decisions on held-out blur: TRUST 45.7% (1.6% error), CAUTION 20.3% (10.5% error), REJECT 34.0% (47.3% error). The demo story works: of 172 confidently wrong (raw > 85%) severity-5 blur predictions, 128 were REJECTed. Feature importance (mean |SHAP|): tta_pconf 2.00, maha_pred 0.78, knn_dist 0.59, entropy 0.39, tta_std 0.33, margin 0.25, raw_confidence 0.22, tta_agree 0.01.

## 9. Timeline from now (Sat ~1 PM, behind the doc's 10–1 slot)

| When | C does | Evidence |
|---|---|---|
| 1–2:30 | C1 + C5 (with self-check) against mock files | Asserts and self-check pass |
| 2:30–4 | C2 + C3 + C4 on the first real files, headline fold only | **4 PM:** real p_correct + thresholds.json |
| 4–7 | C6: all 4 folds + CIFAR-10.1 → evaluation.json v1 to D | D draws the headline chart from real JSON |
| 7–9 | Model selection on val only; freeze the config | thresholds.json frozen together with the prediction freeze |
| 9–12 | Bootstrap (3 comparisons), C7 reasons, final scores.parquet | evaluation.json v2; demo rows reviewed |
| 12 AM | **Feature freeze** | |
| 12–2 | Integration: D's demo_cache.json from scores.parquet | The 3 story images render |
| 2–4 | Verify every claim: trace each pitch number back to evaluation.json | Claims checklist |

## 10. Setup on this Mac

```bash
brew install libomp                      # XGBoost 3.4.1 will not import without it
python3 -m venv .venv && .venv/bin/pip install numpy pandas pyarrow scikit-learn xgboost scipy
```
