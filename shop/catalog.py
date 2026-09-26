"""Fictional product catalog -> data/shop/catalog.json (schema: common.schemas.Catalog). Owner: A.

3-5 products per class, invented brand names/prices/descriptions, deterministic from config seed,
"fictional": true. No real brands.
"""
from __future__ import annotations


def generate_catalog(product_classes: list[str], seed: int) -> dict:
    """TODO(A)."""
    raise NotImplementedError("TODO(A): generate_catalog")


def products_for_class(catalog: dict, class_name: str) -> list[dict]:
    """TODO(A)."""
    raise NotImplementedError("TODO(A): products_for_class")
