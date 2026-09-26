# ML Reliability Lab: Predict When Models Fail (HackGT 13)

This project adds a trust layer on top of a frozen CIFAR-10 ResNet-18. For each prediction it outputs one calibrated `p_correct` and a TRUST / CAUTION / REJECT decision. It is evaluated on corruption families it never trained on and on CIFAR-10.1.

- Setup and downloads: [REQUIREMENTS.md](REQUIREMENTS.md)

## Layout

| Path | Owner | What |
|---|---|---|
| `inference/` | A | ResNet-18 loader, inference with 3 TTA views, embeddings |
| `benchmark/` | B | CIFAR loaders, manifest, base-image splits |
| `trust/features.py`, `trust/signals.py` | B | features.csv (TTA, kNN, Mahalanobis) |
| `trust/` (all other files) | C | temperature scaling, XGBoost + isotonic, thresholds, metrics |
| `backend/`, `frontend/` | D | demo UI, research dashboard, optional live `/predict` |
| `data/` | all | shared artifacts (see below) |
| `data/raw/` | all | downloaded datasets and checkpoint (not in git) |

Run modules from the repo root with `python -m <folder>.<file>`, for example `python -m trust.run_trust`. This lets imports like `from trust.evaluate import ...` resolve.

## Data files

| File in `data/` | Made by | Used by | In git? |
|---|---|---|---|
| `manifest.csv` | B | A, C | no, too large (share via team drive) |
| `prediction_runs.parquet` | A | B, C | no |
| `embeddings.npy`, `tta_logits.npy`, `train_embeddings.npy`, `train_labels.npy` | A | B | no |
| `features.csv` | B | C | no |
| `thresholds.json`, `evaluation.json`, `models/` | C | D | yes |
| `scores.parquet` | C | D | yes |
| `demo_cache.json` | D | demo | yes |

Team drive for the files not in git: _add link_
