"""FastAPI app (plan.md Section 9). Owner: D.

Run: make backend  (uvicorn backend.main:app --reload)

Loads artifacts on startup if present. Endpoints that depend on unimplemented logic return 501 with
a TODO(owner) so the frontend can be built against the contract today.
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from common import io, schemas

app = FastAPI(title="ML Reliability Lab")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CORE_ARTIFACTS = (io.EVALUATION, io.DEMO_CACHE, io.THRESHOLDS)
STATE: dict = {"artifacts": {}, "mock": {}, "shop_enabled": False}


def _load_core() -> None:
    data = io.data_dir()
    for name in CORE_ARTIFACTS:
        path = data / name
        if path.exists():
            STATE["artifacts"][name] = io.read_json(path)
            STATE["mock"][name] = io.is_mock_artifact(path)
        else:
            STATE["artifacts"][name] = None


def _mount_shop() -> None:
    """The ONE allowed import of shop/ from outside it (isolation rule 1). Guarded so the core app
    runs if shop/ is deleted."""
    try:
        from shop.router import router, shop_artifacts_present  # noqa: WPS433
        from shop.paths import shop_path
    except ImportError:
        return
    if not shop_artifacts_present():
        return
    app.include_router(router)
    STATE["shop_enabled"] = True
    for name in ("shop_cache.json", "catalog.json"):
        STATE["mock"][f"shop/{name}"] = io.is_mock_artifact(shop_path(name))
    shop_images = shop_path("demo_images")
    if shop_images.is_dir():
        app.mount("/shop-images", StaticFiles(directory=shop_images), name="shop-images")


_load_core()
_mount_shop()

_demo_images = io.data_dir() / "demo_images"
_demo_images.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=_demo_images), name="images")


def _require(name: str):
    art = STATE["artifacts"].get(name)
    if art is None:
        raise HTTPException(status_code=404, detail=f"{name} not found — run `make mock` or the pipeline")
    return art


@app.get("/api/health", response_model=schemas.HealthResponse)
def health():
    loaded = {name: STATE["artifacts"].get(name) is not None for name in CORE_ARTIFACTS}
    return schemas.HealthResponse(
        status="ok",
        artifacts=loaded,
        any_mock=any(STATE["mock"].values()),
        shop_enabled=STATE["shop_enabled"],
    )


@app.get("/api/evaluation", response_model=schemas.Evaluation)
def evaluation():
    return _require(io.EVALUATION)


@app.get("/api/demo/samples", response_model=list[schemas.DemoSampleSummary])
def demo_samples():
    """Group demo_cache samples by base_image_id -> available corruptions / severities. TODO(D)."""
    _require(io.DEMO_CACHE)
    raise HTTPException(status_code=501, detail="TODO(D): /api/demo/samples")


@app.get("/api/demo/{base_image_id}", response_model=schemas.PredictResponse)
def demo_sample(base_image_id: str, corruption: str = "clean", severity: int = 0):
    """Lookup by (base_image_id, corruption, severity) in demo_cache. TODO(D)."""
    _require(io.DEMO_CACHE)
    raise HTTPException(status_code=501, detail="TODO(D): /api/demo/{base_image_id}")


@app.post("/api/predict", response_model=schemas.PredictResponse)
def predict(image: Optional[dict] = None):
    """Optional bonus: live inference. Must return 503 with a clear message if model/bank not loaded.
    The UI never requires this. TODO(A)."""
    raise HTTPException(status_code=503, detail="Live inference not loaded; UI uses demo_cache.json")
