# shop/ — Visa "Snap-to-Shop" wrapper (isolated)

Owner: **A** (with help from whoever is free). Drop rule: if this isn't working end-to-end by
**12 AM**, hide tab 3 and move on.

## Isolation rules (enforced by `scripts/check_isolation.py` and `tests/shop/test_isolation.py`)
1. Nothing outside `shop/` imports `shop` — except the guarded router mount in `backend/main.py`.
2. Every write in `shop/` goes through `shop.paths.shop_path()` (asserts target is under `data/shop/`).
3. Deleting `shop/` must leave core tests, `make mock`, the backend, and tabs 1–2 working.

## Files
| file | what | status |
|---|---|---|
| `shop_config.yaml` | all wrapper settings | done |
| `paths.py` | `shop_path()` write guard | done |
| `classes.py` | product names → ImageNet indices; fail loudly + print close matches | TODO(A) |
| `make_manifest.py` | manifest (same schema, `dataset=imagenet_products[_c]`) | TODO(A) |
| `corrupt.py` | 8 MVP corruptions × 5 severities @ 224 px (`imagecorruptions`) | TODO(A) |
| `model.py` | torchvision ResNet-18, subset-restricted softmax, 512-d embeddings | TODO(A) |
| `run_pipeline.py` | reuse core `inference/` + `trust/` functions with shop paths | TODO(A) |
| `catalog.py` | fictional catalog (`"fictional": true`) | TODO(A) |
| `gate.py` | decision → allowed actions (**source of truth**) | `ALLOWED` defined, functions TODO(A) |
| `assistant.py` | LLM assistant + offline fallback | TODO(A) |
| `router.py` | FastAPI router for `/api/shop/*` | routes stubbed (501) |
| `build_shop_cache.py` | `data/shop/shop_cache.json` + PNGs | TODO(A) |

See `download_instructions.md` for image sourcing (decide in the first hour).

## `imagecorruptions` install notes
Installed via `pip install -e ".[shop]"`. It can conflict with NumPy / scikit-image versions.
If it fails, record the error here and implement the 8 MVP corruptions directly in `corrupt.py` (TODO(A)).

_Install log:_ (fill in after `make setup`)
