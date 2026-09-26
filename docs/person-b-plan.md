# Person B: Benchmark, Manifest, Splits and Signals

Scope: B's work only. A, C and D appear here only as the files B hands to them or reads from them.

## 1. Status (Sat 26 Sep, ~2 PM)

| Part | Status |
|---|---|
| B1: loaders for CIFAR-10, CIFAR-10.1 and CIFAR-10-C | **Done, tested** |
| B2: `load_images(manifest)` for A | **Done, tested** |
| B3: `manifest.csv` plus the split by base image | **Done, tested.** `data/manifest.csv` is written at 2,000 base images (84,000 rows) |
| B4: mock A outputs + `features.csv` for C | **Done, tested.** `python -m trust.features --mock` writes everything to `data/mock/` |
| B5: signals (`trust/signals.py`) | **Done, tested** against scipy and sklearn references |
| B6: `features.csv` builder (`trust/features.py`) | **Done, tested on mocks.** Needs one rerun on A's real files |
| B7: sanity report | **Done.** Printed by `python -m trust.features` |
| Trust Score (doc §9.2 / §16 nice-to-have) | **Done, tested.** Extra `trust_score` column in `features.csv`; C decides whether to train on it |
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
- **To C:** CIFAR-10.1 rows are `family=natural, corruption=natural, split=test`. The mock files are ready (§7).

## 7. Signals and features.csv (B4–B7)

### How to run

```bash
python -m trust.features --mock   # fake A outputs → data/mock/ (10,200 rows, ~3 s), then data/mock/features.csv
python -m trust.features          # real run: reads A's files in data/, writes data/features.csv + sanity report
```

`features.csv` columns, in manifest row order: `sample_id, raw_confidence, entropy, margin, tta_agree, tta_pconf, tta_std, knn_dist, maha_pred, trust_score, failure`. `trust_score` is extra: C's `load()` selects columns by name, so it is ignored until C adds it to `FEATURES`. Corruption, family and severity are **not** included; C joins them from the manifest.

### What B reads from A (the contract the mocks follow)

| File | Contents |
|---|---|
| `prediction_runs.parquet` | `sample_id, pred_label, correct, logit_0 … logit_9`. The mock also carries the other §8.1 fields: `base_image_id, split, true_label, raw_confidence, top2_prob, entropy, margin, corruption, severity` |
| `tta_logits.npy` | N × 3 × 10, **in `prediction_runs.parquet` row order**. Views: hflip, +2 px, −2 px (reflect padding) |
| `embeddings.npy` | N × 512, same row order |
| `train_embeddings.npy`, `train_labels.npy` | The 50k CIFAR-10 train set: the kNN and Mahalanobis bank |

Rows are joined on `sample_id`, so A can write them in any order. The run **stops with a clear message** if any of these hold:
- A manifest row is missing, or an unknown `sample_id` appears.
- A `sample_id` is duplicated.
- The `.npy` shapes don't match the parquet.
- `pred_label != argmax(logits)`.
- `correct` disagrees with the manifest's `true_label`.
- Any feature is NaN or inf.

`correct` may be int or bool.

### Signal definitions

Everything is computed in float64 from the logits. `k` = the class predicted on the original image.

| Feature | Definition |
|---|---|
| raw_confidence | top-1 softmax probability |
| entropy | −Σ p log p (nats) |
| margin | top-1 minus top-2 probability |
| tta_agree | share of the 3 TTA views whose argmax is `k` |
| tta_pconf | mean of p_k over the 3 TTA views |
| tta_std | std of p_k over the original plus the 3 views (population std) |
| knn_dist | 1 − cosine similarity to the 10th-nearest train embedding (all 50k train images) |
| maha_pred | Mahalanobis distance to class `k`'s train mean, using a shared covariance (tiny ridge, `pinvh`) |
| trust_score | Trust Score (Jiang et al. 2018): distance to the nearest train embedding of any other class ÷ distance to the nearest one of class `k`. Euclidean on unit-normalized embeddings (like `knn_dist`, so contrast/brightness scaling of activations doesn't matter); no density filtering. Higher = more trustworthy; > 1 means class `k` is the nearest class |
| failure | 1 − correct |

### Tests

All pass.
- **softmax stats:** checked against `scipy.special.softmax` / `scipy.stats.entropy`, including logits of 1000 (no overflow).
- **TTA stats:** checked on a hand-worked example.
- **kNN:** checked against sklearn brute-force cosine `NearestNeighbors`, with a chunk size that doesn't divide N. A bank point's k=1 distance is 0.
- **Mahalanobis:** checked against `scipy.spatial.distance.mahalanobis`. A rank-deficient bank still gives finite distances.
- **Trust Score:** checked against a brute-force per-class nearest-neighbour search, with unsorted bank labels and an uneven chunk size. A query that is itself a bank point gets a huge but finite score. Predicting the nearest class always gives a score ≥ 1.
- **C compatibility:** with the extra column, `python -m trust.run_trust data/mock` and `python -m trust.test_trust` both pass.
- **AUROC** (in the sanity report): checked against `sklearn.roc_auc_score` with ties.
- **End-to-end on shuffled mock files:** 40 random rows were recomputed by hand from their own parquet and `.npy` rows and match.
- **Contract violations:** a dropped row, a short `tta_logits`, a wrong `pred_label`, a wrong `correct` and a duplicate `sample_id` are each caught.

**Runtime at full scale** (412k rows × 50k bank, M-series Mac): about 2¼ min, 1.4 GB peak. Almost all of it is the kNN search; the Trust Score shares that pass and adds under 1%.

### About the mocks

`trust/mock_runs.py` is a small synthetic model, not the ResNet: embeddings are class prototypes plus noise, and logits come from a linear head on them. It is tuned so the headline shape looks real:

| Rows | Mock accuracy | Real accuracy (C's smoke run) |
|---|---|---|
| Clean | 96% | 95% |
| CIFAR-10.1 | 87% | 89% |
| Blur, severity 5 | 57% at 89% raw confidence | 61% at 89% |

Use the mocks for schemas, joins and plumbing only. Their signal quality is **not** realistic: `knn_dist` barely varies, and TTA ranks worse than raw confidence, the opposite of the real smoke run. The mock bank has 10k rows, not 50k. `data/mock/` is gitignored; anyone with `data/raw/` can regenerate it in a few seconds.

## 8. Next steps

1. When A's first real files land in `data/`: run `python -m trust.features`, check the sanity report (clean accuracy ≈ 94.98%, CIFAR-10.1 ≈ 88.75%, `tta_pconf` the strongest single feature on blur), then share `features.csv` with C through the team drive.
2. Rerun it after A's final full-size inference run (10,000 base images).
3. Stretch B8: `benchmark/corruptions.py` for live `/predict`.

## 9. Setup

The datasets are exactly as in [REQUIREMENTS.md](../REQUIREMENTS.md) §2.3. B1–B3 need only `numpy` and `pandas`. B4–B7 also need `pyarrow` and `scipy`, and `scikit-learn` for the tests only. The CIFAR-10-C stream took about 14 minutes here (13:38 → 13:52).
