# TASKS

Every stub raises `NotImplementedError("TODO(X): …")` or exits with `TODO(X)`. Find yours with:

```bash
grep -rn "TODO(A" --include=*.py --include=*.ts --include=*.tsx --include=*.sh .
```

`make test` reports unimplemented functions as XFAIL; when you implement one, its spec test starts
running for real. `pytest -rx` lists what's left.

## First hour (whole team)
- [ ] Confirm checkpoint clean accuracy (A: `scripts/check_clean_accuracy.py`)
- [ ] Lock corruption list + `headline_holdout_family` in `config.yaml`
- [ ] Review `common/schemas.py` together; freeze contracts
- [ ] Who has a GPU? Set `benchmark.n_base_images` (2000 on CPU)
- [ ] Shop image source (`shop/download_instructions.md`); fix unresolved product names in `shop/shop_config.yaml`

## A — inference + shop
- [ ] `inference/model.py`: `load_model`, `normalize`, `embed_and_logits`
- [ ] `inference/tta.py`: `hflip`, `shift`, `apply_view` (spec: `tests/test_tta.py`)
- [ ] `inference/reference_bank.py`: step 2
- [ ] `inference/run_inference.py`: step 3 (batched, resumable, tqdm)
- [ ] `scripts/check_clean_accuracy.py`
- [ ] `backend/main.py::predict` (optional bonus)
- [ ] `shop/classes.py`: exact resolve, fail loudly with close matches
- [ ] `shop/corrupt.py` (+ record any `imagecorruptions` install failure in `shop/README.md`)
- [ ] `shop/model.py`: subset-restricted softmax + 512-d embeddings
- [ ] `shop/make_manifest.py`, `shop/run_pipeline.py`, `shop/build_shop_cache.py`
- [ ] `shop/catalog.py`: fictional catalog
- [ ] `shop/gate.py`: `effective_decision`, `allowed_actions`, `can_checkout`, `enforce` (spec: `tests/shop/test_gate.py`)
- [ ] `shop/assistant.py`: prompt, LLM call w/ 5 s timeout, offline templates
- [ ] `shop/router.py`: 4 routes (checkout 403 unless gate allows)
- [ ] `common/mock.py::write_all_shop`

## B — benchmark + signals
- [ ] `benchmark/download.sh`: verify URLs
- [ ] `benchmark/load_cifar.py`, `benchmark/load_cifar10c.py` (spec: `tests/test_cifar10c_index.py`)
- [ ] `benchmark/make_splits.py`: `assign_splits`, `lofo_folds` (spec: `tests/test_splits.py`)
- [ ] `benchmark/make_manifest.py`: step 1
- [ ] `common/schemas.py`: `check_manifest_invariants`, dtype checks in `check_columns`
- [ ] `scripts/validate_artifacts.py`: array shape/dtype checks
- [ ] `trust/signals.py`: `softmax_stats`, `tta_signals`, `knn_distance`, `mahalanobis_pred` (spec: `tests/test_signals.py`)
- [ ] `trust/features.py`: step 4; finalize `MODEL_FEATURES`

## C — trust ML + eval
- [ ] `trust/temperature_scaling.py`, `trust/calibrate.py` (spec: `tests/test_temperature_and_calibration.py`)
- [ ] `trust/metrics.py`: all metrics + `bootstrap_ci` over base images (spec: `tests/test_metrics.py`)
- [ ] `trust/thresholds.py`: `loosest_threshold`, `decide`, `broken_promise`, CLI (spec: `tests/test_thresholds.py`)
- [ ] `trust/train_failure_model.py`: step 5 (LR + XGBoost per fold, clean-row weight, isotonic on val)
- [ ] `trust/reasons.py`: `shap_reasons` + `REASON_TEXT` copy
- [ ] `trust/evaluate.py`: step 7

## D — UI + charts
- [ ] `common/mock.py`: all core generators (unblocks everyone's UI work — do this first)
- [ ] `backend/main.py`: `/api/demo/samples`, `/api/demo/{base_image_id}`
- [ ] `trust/build_demo_cache.py`: step 8 (32 → 256 px nearest-neighbor PNGs)
- [ ] `frontend/src/tabs/TrustDemo.tsx`
- [ ] `frontend/src/tabs/ResearchDashboard.tsx` (Recharts, no hardcoded numbers)
- [ ] `frontend/src/tabs/SnapToShop.tsx` (with A)
- [ ] `frontend/src/api.ts`: offline fallback for demo lookups
