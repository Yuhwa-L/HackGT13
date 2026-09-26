"""Loader for shop/shop_config.yaml."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

SHOP_CONFIG_PATH = Path(__file__).resolve().parent / "shop_config.yaml"


class ShopConfig(BaseModel):
    paths: dict[str, str]
    model: dict[str, str]
    product_classes: list[str]
    images_per_class: int
    corruptions: str | list[str]
    severities: list[int]
    image_size: int
    n_curated_photos: int


@lru_cache(maxsize=None)
def load_shop_config(path: str | Path = SHOP_CONFIG_PATH) -> ShopConfig:
    with open(path) as f:
        return ShopConfig.model_validate(yaml.safe_load(f))
