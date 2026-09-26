# ML Reliability Lab (HackGT 13)

A trust layer on top of a frozen CIFAR-10 ResNet-18. For each prediction we extract signals (softmax
confidence/entropy/margin, test-time-augmentation agreement, embedding kNN and Mahalanobis distance),
train XGBoost + isotonic calibration to output one number, `p_correct`, and map it to
**TRUST / CAUTION / REJECT** with thresholds set for a target error rate. Evaluation uses held-out
corruption families (leave-one-family-out on CIFAR-10-C) and natural shift (CIFAR-10.1), split by base
image. A thin Visa "Snap-to-Shop" wrapper (`shop/`) applies the same trust layer to product photos
and uses the decision to gate what a GenAI shopping assistant may do.

**Status: scaffolding only.** Contracts, config, plumbing, Makefile, backend/frontend shells and spec
tests exist. Every piece of real logic is a stub marked `TODO(owner)` — see [TASKS.md](TASKS.md).

## Quickstart

```bash
make setup            # .venv + pip install -e . (+ shop extras) + npm install
make test             # spec tests; stubs show as XFAIL until implemented
make check-isolation  # shop/ isolation rules
make backend          # FastAPI on :8000
make frontend         # Vite on :5173 (proxies /api, /images to :8000)
```

Once `common/mock.py` is implemented (TODO(D)): `make mock && make validate` gives the backend and
frontend data to run against before any real results exist.

Real data (manual, never automatic): `make download` prints instructions; then `bash benchmark/download.sh`.

## Owners

| | area |
|---|---|
| **A** | inference (`inference/`, `scripts/check_clean_accuracy.py`) + **shop wrapper** (`shop/`, with help from whoever is free) |
| **B** | benchmark + signals (`benchmark/`, `trust/signals.py`, `trust/features.py`, schema validators) |
| **C** | trust ML + eval (`trust/temperature_scaling.py`, `calibrate.py`, `train_failure_model.py`, `thresholds.py`, `metrics.py`, `reasons.py`, `evaluate.py`) |
| **D** | UI + charts (`frontend/`, `backend/main.py`, `common/mock.py`, `trust/build_demo_cache.py`) |

## Artifact contracts

Source of truth: [`common/schemas.py`](common/schemas.py) (mirrored in [`frontend/src/types.ts`](frontend/src/types.ts)).
Change the Python contract first, then the TS types, then tell the team. All writes go through
`common/io.py`, which validates. **Every mock artifact carries `is_mock`**; the UI shows a MOCK DATA banner.

| artifact | produced by | schema | notes |
|---|---|---|---|
| `data/manifest.csv` | `benchmark/make_manifest.py` (B) | `MANIFEST_COLUMNS` | one row per evaluated image; split unit = `base_image_id` |
| `data/reference_bank.npy`, `reference_labels.npy` | `inference/reference_bank.py` (A) | `ARRAY_SPECS` | CIFAR-10 **train** embeddings, kNN/Mahalanobis only |
| `data/prediction_runs.parquet` | `inference/run_inference.py` (A) | `PREDICTION_RUNS_COLUMNS` | one row per `sample_id` |
| `data/embeddings.npy`, `embeddings_ids.npy`, `tta_logits.npy` | `inference/run_inference.py` (A) | `ARRAY_SPECS` | same row order; TTA view order = `config.tta.views` |
| `data/features.csv` | `trust/features.py` (B) | `FEATURES_COLUMNS` | model inputs = `MODEL_FEATURES` only |
| `data/models/<fold>/…` | `trust/train_failure_model.py` (C) | — | LR + XGBoost + isotonic per LOFO fold |
| `data/thresholds.json` | `trust/thresholds.py` (C) | `Thresholds` | fit on cal split only |
| `data/evaluation.json` | `trust/evaluate.py` (C) | `Evaluation` | every method × fold + CIFAR-10.1 |
| `data/demo_cache.json` + `data/demo_images/*.png` | `trust/build_demo_cache.py` (D) | `DemoCache` / `PredictResponse` | expo demo runs from this, offline |
| `data/shop/*` (same as above + `shop_cache.json`, `catalog.json`) | `shop/` (A) | same schemas, `ShopCache`, `Catalog` | `dataset = imagenet_products[_c]` |

## Pipeline

`make pipeline LIMIT=50` runs steps 1–8 in order; each step exits with `TODO(owner)` until implemented.
Every script accepts `--mock` and `--limit N` and reads settings only from `config.yaml`.

1. `benchmark/make_manifest.py` → 2. `inference/reference_bank.py` → 3. `inference/run_inference.py` →
4. `trust/features.py` → 5. `trust/train_failure_model.py` → 6. `trust/thresholds.py` →
7. `trust/evaluate.py` → 8. `trust/build_demo_cache.py`

## API (backend/main.py)

| method | path | status |
|---|---|---|
| GET | `/api/health` | working |
| GET | `/api/evaluation` | working (404 until artifact exists) |
| GET | `/api/demo/samples` | TODO(D) |
| GET | `/api/demo/{base_image_id}?corruption=&severity=` | TODO(D) |
| POST | `/api/predict` | returns 503 (optional bonus, TODO(A)) |
| GET/POST | `/api/shop/{samples,identify,assist,checkout}` | mounted only if `data/shop/` artifacts exist; TODO(A) |

## Isolation rules (shop/)

1. Nothing outside `shop/` imports `shop`, except the guarded router mount in `backend/main.py`.
2. Every write in `shop/` goes through `shop.paths.shop_path()` (asserts target is under `data/shop/`).
3. Deleting `shop/` leaves core tests, `make mock`, the backend and tabs 1–2 working.

## Checkpoints

- **First hour:** confirm checkpoint accuracy (~95%), lock corruption list, headline fold, schemas, who has a GPU; choose the shop image source and fix any unresolved product class names.
- **1 PM:** tiny real pipeline (`make pipeline LIMIT=50`) flows to the UI.
- **4 PM:** full real core path; base-image split, TTA features, single `p_correct`, corrupted-val temperature scaling all in. Shop pipeline starts (owner A).
- **7–9 PM:** freeze core predictions and splits; tuning on val only. Shop tiny pipeline working.
- **12 AM:** feature freeze. **If the shop wrapper isn't working end-to-end, drop it** and hide tab 3.
- **4–6 AM:** backup demo video; Devpost write-up (Oracle track + Visa challenge).
- **8 AM:** hacking ends. Submit to Devpost and expo.hexlabs.org before this.
- **9:00–11:15 AM:** expo judging, running from caches.

## LLM config

Copy `.env.example` → `.env`. `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` (OpenAI-compatible endpoint;
team plans to use Grok). `DEMO_OFFLINE=1` forces the deterministic offline assistant.
