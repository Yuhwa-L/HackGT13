# ML Reliability Lab — see README.md and TASKS.md
PY      ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
LIMIT   ?=
MOCK    ?=
PIPE_ARGS = $(if $(LIMIT),--limit $(LIMIT)) $(if $(MOCK),--mock)

.PHONY: help setup download mock test validate check-isolation check-accuracy pipeline shop-pipeline \
        backend frontend frontend-fallback demo

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-18s %s\n",$$1,$$2}'

setup: ## install Python deps (.venv) + frontend deps
	@command -v uv >/dev/null && uv venv .venv || python3 -m venv .venv
	.venv/bin/python -m pip install -U pip
	.venv/bin/python -m pip install -e .
	-.venv/bin/python -m pip install -e ".[shop]" || echo "imagecorruptions failed — record the error in shop/README.md (TODO(A))"
	cd frontend && npm install

download: ## print instructions; run benchmark/download.sh only when invoked explicitly
	@echo "Datasets are NOT downloaded automatically. Review, then run:"
	@echo "  bash benchmark/download.sh"

mock: ## generate all mock artifacts (core + shop)
	$(PY) -m common.mock

test: ## pytest (core + shop); stubs show as xfail
	$(PY) -m pytest -q -rx

validate: ## validate every artifact in data/ against common/schemas.py
	$(PY) -m scripts.validate_artifacts

check-isolation: ## shop/ isolation rules
	$(PY) -m scripts.check_isolation

check-accuracy: ## clean CIFAR-10 accuracy check for the checkpoint
	$(PY) -m scripts.check_clean_accuracy

pipeline: ## core steps 1-8 (LIMIT=N, MOCK=1)
	$(PY) -m benchmark.make_manifest     $(PIPE_ARGS)
	$(PY) -m inference.reference_bank    $(PIPE_ARGS)
	$(PY) -m inference.run_inference     $(PIPE_ARGS)
	$(PY) -m trust.features              $(PIPE_ARGS)
	$(PY) -m trust.train_failure_model   $(PIPE_ARGS)
	$(PY) -m trust.thresholds            $(PIPE_ARGS)
	$(PY) -m trust.evaluate              $(PIPE_ARGS)
	$(PY) -m trust.build_demo_cache      $(PIPE_ARGS)

shop-pipeline: ## shop steps end-to-end (LIMIT=N, MOCK=1)
	$(PY) -m shop.classes
	$(PY) -m shop.make_manifest
	$(PY) -m shop.corrupt
	$(PY) -m shop.run_pipeline $(PIPE_ARGS)
	$(PY) -m shop.build_shop_cache

backend: ## FastAPI on :8000
	$(PY) -m uvicorn backend.main:app --reload --port 8000

frontend: ## Vite dev server on :5173
	cd frontend && npm run dev

frontend-fallback: ## copy JSON artifacts into frontend/public/fallback/
	mkdir -p frontend/public/fallback
	-cp data/evaluation.json data/demo_cache.json data/thresholds.json frontend/public/fallback/ 2>/dev/null
	-cp data/shop/shop_cache.json data/shop/catalog.json frontend/public/fallback/ 2>/dev/null
	@ls frontend/public/fallback

demo: ## build caches, copy fallbacks, start backend + frontend
	$(PY) -m trust.build_demo_cache
	-$(PY) -m shop.build_shop_cache
	$(MAKE) frontend-fallback
	$(MAKE) -j2 backend frontend
