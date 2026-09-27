# Person D: Demo, Dashboard, Integration

**Status, Sat 26 Sep ~5 PM:** built and tested. `backend/build_demo_cache.py`, `backend/main.py` and `frontend/` (demo + research tabs) exist; `python -m backend.test_backend` runs 3 checks (PNG encoding, cache contents and story logic on the real trust layer, the server's files and `/predict`). The page was driven in a real browser: every control, both tabs, no overflow at 1280 px or 400 px, no console errors. `data/demo_cache.json` now holds the **full run** (see §0), 30 test photos, 3 per class, 5.2 MB.

## 0. Short answers

- **Which data?** The full benchmark, run Sat 26 Sep 17:57–18:16 on the Mac's GPU with the team's own code: B's manifest (all 10,000 CIFAR-10 test photos × 41 versions + 2,000 CIFAR-10.1 = 412,000 rows), A's inference and embeddings (the device line now tries NVIDIA, then Apple GPU, then CPU), B's features, C's trust layer, D's cache. Total 19 min; to redo it, run the six commands in REQUIREMENTS.md order (`benchmark.make_splits --n-base 10000`, `inference.run_inference`, `inference.extract_embeddings`, `trust.features`, `trust.run_trust`, `backend.build_demo_cache`).
- **What runs at the expo?** Only the web page and a tiny local server. It uses precomputed files, no GPU, no internet (no CDNs, no web fonts), so it cannot break during judging.

## 1. The pitch

The README is the checked version of this argument: every number in it was matched against `data/`.

**One-sentence claim:** a classifier's confidence lies when its inputs shift. Our trust layer turns each prediction into one calibrated probability of being right, plus a TRUST / CAUTION / REJECT decision, and it keeps its promises on corruption types it never saw.

**Arguments for the service.** Numbers from the full run, blur held out (the trust layer never trained on any blur).

1. **The problem is real.** At the worst blur level the ResNet is right 60.3% of the time but claims 87.5%. A "5% error" rule tuned on clean images delivers **20.0%** error on unseen blur.
2. **Fixing the average is not enough.** Temperature scaling on corrupted data is almost as well calibrated as the trust layer (ECE 0.030 vs 0.024), but it finds failing predictions no better than raw confidence (AUROC 0.861 vs 0.863); the trust layer reaches 0.884. So with cutoffs tuned on the same corrupted calibration photos, raw confidence accepts 59% of blurred photos at 4.2% error and the trust layer accepts 64% at 4.6%. Say this fair comparison, not only "20% vs 4.6%" (the 20% rule was tuned on clean photos).
3. **Safer automation.** TRUST covers 42.0% of unseen-blur images at 1.1% error. The 20% most-trusted have 0.2% error (raw confidence: 1.0%). REJECT catches 592 of 797 confidently wrong severe-blur predictions (74%).
4. **A built-in drift alarm.** The REJECT rate rises from 14% on clean images to 61% at the worst blur and 79% at the worst noise. A rising REJECT rate tells an operator the inputs changed (dirty lens, fog, compression) before accuracy is measured.
5. **Plug-in and cheap.** The classifier is never retrained. The layer needs only the model's outputs, 3 extra passes and an embedding lookup, and it trains in about 25 s on 412,000 predictions. Any classifier can get one after a quick retrain.
6. **Explainable.** Every CAUTION and REJECT comes with plain-language reasons (unstable under small flips, unfamiliar image, low confidence).
7. **Honestly evaluated.** It is tested only on families it never trained on, with bootstrap intervals over photos, plus a real-world shift set (CIFAR-10.1).

**Service shape:** `POST /predict` returns the prediction, raw confidence, baselines, `p_correct`, the decision and the reasons. Uses: auto-approve TRUST, send CAUTION for review, escalate REJECT to a human or a bigger model, and chart the REJECT rate over time as a drift monitor.

## 2. Demo pipeline

```
data/manifest.csv ─┐
data/prediction_runs.parquet ─┼─► python -m trust.run_trust ─► evaluation.json, thresholds.json, scores.parquet
data/features.csv ─┘                                                    │
data/raw/ (CIFAR pixels) ─────────────────────────────────┐             │
                                                          ▼             ▼
                                    python -m backend.build_demo_cache ─► data/demo_cache.json
                                                                        │
                                    python -m backend.main ─► http://localhost:8000  (demo + research tabs)
```

- `build_demo_cache` picks 30 story images from the **test split** (3 per class; the trust layer never trained on them). For each it stores all 41 versions: 32×32 PNG pixels, the ResNet prediction, the baselines, p_correct, the decision and the reasons. Each corrupted version is scored by the fold that never saw its family. It also stores per-family curves (accuracy, confidence, p_correct and decision shares by severity) for the drift chart.
- `backend.main` is a standard-library server: the page, the three data files, and `POST /predict` answered from the cache in the §11 contract format. It binds to localhost and serves only whitelisted files.

## 3. Files

| File | What |
|---|---|
| `backend/build_demo_cache.py` | scores.parquet + manifest + pixels → data/demo_cache.json |
| `backend/main.py` | local server: page, data files, POST /predict |
| `backend/test_backend.py` | checks for the cache builder and the server |
| `frontend/index.html`, `frontend/app.js` | the demo and research tabs (plain HTML/JS, inline SVG charts, system fonts) |
| `data/demo_cache.json` | generated; committed so the demo runs from a fresh clone |

## 4. The page

**Demo tab (single image):**
- story buttons for the three beats: 1 clean, 2 moderate, 3 heavy
- a gallery of the 30 images
- a corruption picker (8 types, grouped by family) and a severity slider (0 = clean)
- a strip of all 6 severities with their decisions
- the result panel: prediction vs. truth, the confidence ladder (raw → temperature-scaled clean → temperature-scaled corrupted → p_correct), the decision badge and the reasons

**Research tab:**
- headline numbers
- confidence vs. accuracy by severity (headline family)
- the drift chart: all four families, REJECT share and p_correct vs. accuracy by severity
- calibration
- the broken-promise check

## 5. One-minute story (at the table)

The cache builder picks the story automatically: the most dramatic blur photo in the test split. On the full run it is a dog photo (`c10_01977`) with defocus blur. The three buttons at the top of the demo tab play it:

1. **Clean:** ResNet says dog (right), raw confidence >99%, trust layer >99% → TRUST.
2. **Defocus blur, severity 3:** still dog (right) and raw confidence still 99%, but p_correct 89% → CAUTION (the image looks unfamiliar).
3. **Defocus blur, severity 4:** ResNet says **cat** (wrong) with 91% confidence; p_correct 26% → REJECT, because the prediction is unstable under small flips/shifts and the image is far from clean training images.
4. **Switch to Research:** confidence stays high while accuracy falls and p_correct follows it. The "5% promise" breaks for raw confidence (20.0%) and holds for the trust layer (4.6%). The REJECT rate rises with severity, which is the drift alarm.

## 6. Judge Q&A (honest answers)

- *Does it beat simple baselines?* On unseen blur it ranks failures better than raw confidence (AUROC 0.884 vs 0.863, 95% CI of the gain [+0.017, +0.026]) and slightly better than the TTA-only signal ([+0.002, +0.008]); it beats raw confidence on all four held-out families. Calibration: ECE 0.024 vs 0.096 for TTA-only and 0.137 for raw. Against temperature scaling fit on corrupted data it is only a little better on blur (0.024 vs 0.030) and digital (0.028 vs 0.029), clearly better on noise (0.158 vs 0.214) and weather (0.034 vs 0.066).
- *Where does it fail?* Noise is the hardest family: accuracy is 49.7% but mean p_correct is 65.5%, so it is still overconfident there. On CIFAR-10.1 (real-world shift) it gains over raw confidence (AUROC 0.896 vs 0.884) but the interval touches zero, TTA-only ranks slightly better (0.903), and temperature scaling on clean data is better calibrated (ECE 0.035 vs 0.049).
- *Did you try the Trust Score (Jiang et al.)?* Yes, B computes it; adding it changed validation AURC by less than 0.0001, so the model keeps its 8 features.
- *Why not retrain the ResNet?* The classifier is the system under test; keeping it frozen makes the comparison clean, and real users often cannot retrain theirs.
- *Cost?* 4 forward passes per image plus an embedding lookup; XGBoost itself takes microseconds.
- *Clean images?* The layer is a little cautious there: it rejects 14.5% of clean images while 4.7% are wrong.

## 7. Expo checklist and timeline

- Done: demo and research tabs working; full run finished 18:16 and the demo is on it. Commit the code and the regenerated files in `data/` (json, scores.parquet, models/).
- 4–6 AM: record the backup video of the story and the research tab.
- At the table: `python -m backend.main`, laptop offline-safe, video ready.
