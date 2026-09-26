# Person B: Benchmark, Manifest, Splits and Signals

Scope: B's work only. A, C and D appear here only as the files B hands to them or reads from them.

## 1. Status (Sat 26 Sep, ~2 PM)

| Part | Status |
|---|---|
| B1: loaders for CIFAR-10, CIFAR-10.1 and CIFAR-10-C | **Done, tested** |
| B2: `load_images(manifest)` for A | **Done, tested** |
| B3: `manifest.csv` plus the split by base image | **Done, tested.** `data/manifest.csv` is written at 2,000 base images (84,000 rows) |
| B4: mock `prediction_runs.parquet` and `features.csv` for C | Next |
| B5–B7: signals, then `features.csv` and a sanity report | After A's first real prediction file |
| B8 (stretch): `corruptions.py` for live `/predict` | Only if the core is stable |

## 2. Manifest contract (frozen; tell B before changing it)

One row per evaluated image. `base_image_id` is the split unit.

| Column | Clean CIFAR-10 test | CIFAR-10-C | CIFAR-10.1 |
|---|---|---|---|
| sample_id | `c10_clean_00123_s0` | `c10c_gaussian_noise_00123_s3` | `c101_natural_00123_s0` |
| base_image_id | `c10_00123` | `c10_00123` | `c101_00123` |
| dataset | `cifar10_test` | `cifar10c` | `cifar10_1` |
| image_index (row in the source array) | `i` | `(severity-1)*10000 + i` | `j` |
| true_label / true_class | int / name | int / name | int / name |
| corruption | `clean` | type name | `natural` |
| family | `clean` | noise / blur / weather / digital | `natural` |
| severity | 0 | 1–5 | 0 |
| split | from its base image | from its base image | always `test` |

- **Families:**

  | Family | Corruptions |
  |---|---|
  | noise | gaussian_noise, impulse_noise |
  | blur | defocus_blur, motion_blur |
  | weather | fog, brightness |
  | digital | contrast, jpeg_compression |
- **Split:** by `base_image_id`, stratified by class, seed 0: train 50 / val 15 / cal 15 / test 20. Every corrupted version of a base image lands in the same split as its clean image.
- **CIFAR-10.1** is `family=natural, split=test`. It can never match C's `split=="train" & family in seen` filter.
- **Folds:** the manifest has no fold columns. The leave-one-family-out folds come from `family`.
- **CIFAR-10 train set:** never a manifest row. It is only the kNN and Mahalanobis reference bank.
- **Column order:** `sample_id, base_image_id, dataset, image_index, true_label, true_class, corruption, family, severity, split`.

## 3. How to use it

```bash
python -m benchmark.make_splits                  # 2,000 base images → 84,000 rows (default)
python -m benchmark.make_splits --n-base 10000   # full benchmark → 412,000 rows (~1.3 s)
python -m benchmark.make_splits --n-base 40      # 1 PM-style tiny pipeline → 1,650 rows
```

| `--n-base` | CIFAR-10 rows (×41 per base image) | CIFAR-10.1 rows | Total |
|---|---|---|---|
| 40 (tiny, ≤ 50) | 1,640 | 10 (a stratified subset, `n/5`, at least 1 per class) | 1,650 |
| 2,000 | 82,000 | 2,000 | 84,000 |
| 10,000 | 410,000 | 2,000 | 412,000 |

**For A.** This is the one function that turns manifest rows into pixels:

```python
import pandas as pd
from benchmark.load_cifar10c import load_images

m = pd.read_csv("data/manifest.csv")
x = load_images(m)          # (N, 32, 32, 3) uint8, same row order as m; any slice of m works
```

- It needs only the `dataset`, `corruption` and `image_index` columns.
- CIFAR-10-C files are memory-mapped, so the 8 × 150 MB files are not all loaded into RAM.
- Normalization (mean/std) and the TTA views stay in A's code.

**Other loaders:**
- In `benchmark/load_cifar.py`: `load_cifar10_test()`, `load_cifar10_train()` (the reference bank), `load_cifar10_1()` and `class_names()`.
- In `benchmark/load_cifar10c.py`: `load_cifar10c(name)`, `load_cifar10c_labels()`, `c10c_index(i, severity)` and `CORRUPTIONS` (corruption → family).

All loaders are cached and return **read-only** arrays. Call `.copy()` before modifying one in place.

## 4. What was checked

`make_splits.py` asserts all of the following on every run, and fails loudly if one breaks:
- `sample_id` is unique.
- No missing values.
- No base image appears in two splits.
- Each CIFAR-10 base image has exactly 41 rows, all with the same label.
- CIFAR-10.1 rows are only `natural`/`test`, and no `natural` row appears outside test.
- Split shares are exact overall and within 2 images per class.
- `image_index` points at the right label in every source file, including CIFAR-10-C `labels.npy` for all corrupted rows.
- CIFAR-10-C `image_index % 10000` matches the base image, and `// 10000 + 1` matches the severity.

A separate test script also checked:

| Check | Result |
|---|---|
| Pixel layout against a from-scratch decode of `test_batch` (channel order, row-major) | OK |
| Class counts: train 5,000 per class; CIFAR-10.1 200 per class | OK |
| Manifest asserts at `--n-base` 10, 40, 50, 51, 2,000, 10,000 | All pass |
| Same seed gives an identical manifest; seed 1 gives a different one | OK |
| Class × split at 10,000 base images | Exactly 500 / 150 / 150 / 200 for every class |
| Every corrupted row has the same split and label as its clean row | OK (400,000 rows) |
| `load_images` on 3,000 shuffled rows mixing all three datasets | Pixel-identical to the source `.npy` / batch files |
| Mean squared difference from the clean image, severity 1 → 5 (200 base images) | Rises with severity for all 8 types (table below) |

| Corruption | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|
| gaussian_noise | 101 | 223 | 392 | 492 | 602 |
| impulse_noise | 206 | 410 | 602 | 1032 | 1442 |
| defocus_blur | 4 | 26 | 58 | 100 | 200 |
| motion_blur | 138 | 279 | **440** | **429** | 567 |
| fog | 185 | 711 | 1150 | 1660 | 2318 |
| brightness | 108 | 447 | 994 | 1719 | 3562 |
| contrast | 187 | 748 | 1078 | 1467 | 2162 |
| jpeg_compression | 57 | 87 | 98 | 110 | 128 |

## 5. Things the team should know

1. **The 2,000 and 10,000 manifests don't share splits.** The same base image can be train in one and test in the other. Never mix results or trained models across the two sizes.
2. **Motion blur is not strictly monotone:** severity 4 is slightly closer to clean than severity 3. I believe this is how CIFAR-10-C's motion blur is built, not a bug, since the other seven types rise every step. Expect it in C's per-severity charts too.
3. **The 2,000-image manifest has 2,000 CIFAR-10 base images, not 4,000.** The other 2,000 base images are CIFAR-10.1.
4. **`manifest.csv` is gitignored** (`data/*.csv`). Share it through the team drive. Anyone with `data/raw/` can also rebuild the identical file in about a second.

## 6. Handoffs

- **To A:** use the manifest and `load_images` above. Please also export:
  - `train_embeddings.npy` + `train_labels.npy`: the 50k train set, needed for kNN and Mahalanobis.
  - Flat `logit_0 … logit_9` in `prediction_runs.parquet`.
  - `tta_logits.npy` (N × 3 × 10) and `embeddings.npy`, in `prediction_runs.parquet` row order. Views: hflip, +2 px, −2 px horizontal shift with reflect padding.
- **To C:** CIFAR-10.1 rows are `family=natural, corruption=natural, split=test`. Mock files are next (§7).

## 7. Next steps

| # | Task | File | Done when |
|---|---|---|---|
| B4 | Mock `prediction_runs.parquet` + `features.csv` with the exact schemas | `trust/features.py --mock` | C's C1 asserts run on them |
| B5 | Signals, float64 from logits; `k` = original predicted class | `trust/signals.py` | Sanity checks pass |
| B6 | Join on `sample_id`; write `sample_id, raw_confidence, entropy, margin, tta_agree, tta_pconf, tta_std, knn_dist, maha_pred, failure` | `trust/features.py` | No NaN; clean accuracy ≈ 94.98%, CIFAR-10.1 ≈ 88.75% |
| B7 | Per-feature means by family × severity, single-feature AUROC | `trust/features.py` | Shared with C |

The B5 signals:

| Signal | Definition |
|---|---|
| `tta_agree` | Share of the 3 TTA views whose argmax is `k` |
| `tta_pconf` | Mean of p_k over the 3 views |
| `tta_std` | Std of p_k over the original image plus the 3 views |
| `knn_dist` | Cosine distance to the 10th-nearest train embedding |
| `maha_pred` | Mahalanobis distance to the predicted class's mean, shared covariance |

## 8. Setup

The datasets are exactly as in [REQUIREMENTS.md](../REQUIREMENTS.md) §2.3. B1–B3 need only `numpy` and `pandas`. The CIFAR-10-C stream took about 14 minutes here (13:38 → 13:52).
