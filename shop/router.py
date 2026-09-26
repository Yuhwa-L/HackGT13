"""FastAPI router for /api/shop/*. Mounted by backend/main.py only if data/shop/ artifacts exist.
Owner: A (routes) / D (frontend consumer).
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from common import schemas
from shop.paths import shop_path

router = APIRouter(prefix="/api/shop", tags=["shop"])

SHOP_ARTIFACTS = ("shop_cache.json", "catalog.json")


def shop_artifacts_present() -> bool:
    return all(shop_path(name).exists() for name in SHOP_ARTIFACTS)


def _todo(what: str):
    raise HTTPException(status_code=501, detail=f"TODO(A): {what}")


@router.get("/samples")
def samples() -> list[dict]:
    """Curated shop photos (photo_id, true_class, available corruption/severity variants, image_url)."""
    _todo("/api/shop/samples")


@router.post("/identify", response_model=schemas.ShopIdentifyResponse)
def identify(body: dict):
    """{photo_id} -> ShopIdentifyResponse from shop_cache.json."""
    _todo("/api/shop/identify")


@router.post("/assist", response_model=schemas.AssistResponse)
def assist(body: schemas.AssistRequest):
    """See shop/assistant.py + shop/gate.py. Gate is enforced server-side."""
    _todo("/api/shop/assist")


@router.post("/checkout", response_model=schemas.CheckoutResponse)
def checkout(body: schemas.CheckoutRequest):
    """403 unless gate.can_checkout(...). Mock order summary only; never collects payment data."""
    _todo("/api/shop/checkout")
