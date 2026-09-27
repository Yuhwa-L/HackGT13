"""Visa wrapper settings (replaces plan.md's shop_config.yaml: a Python module, no YAML dependency).

Isolation (plan.md §11): nothing outside shop/ imports shop/ except the guarded mount in backend/main.py, and every file
shop/ writes goes through shop_path(), which asserts the path is under data/shop/.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "shop"
RAW = DATA / "raw"
SEED = 0

# Resolved against torchvision's ImageNet category names by shop/classes.py (exact match or a loud failure).
PRODUCT_CLASSES = [
    "backpack", "sunglasses", "running shoe", "sandal", "Loafer", "sweatshirt", "jean", "cardigan", "sock", "wallet",
    "purse", "digital watch", "coffee mug", "water bottle", "teapot", "frying pan", "toaster", "espresso maker",
    "laptop", "computer keyboard", "mouse", "cellular telephone", "iPod", "binoculars", "acoustic guitar",
    "electric guitar", "umbrella", "sleeping bag", "lipstick", "perfume",
]

# ImageNetV2 (Recht et al. 2019): public, ImageNet-validation-style, 10 images per class per variant.
# Two variants are benchmark photos; the third is the disjoint clean reference bank for kNN / Mahalanobis.
IMAGENETV2_URL = "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-{}.tar.gz"
BENCHMARK_VARIANTS = ["matched-frequency", "threshold0.7"]
BANK_VARIANT = "top-images"

CORRUPTIONS = {"gaussian_noise": "noise", "impulse_noise": "noise", "defocus_blur": "blur", "motion_blur": "blur",
               "fog": "weather", "brightness": "weather", "contrast": "digital", "jpeg_compression": "digital"}
SEVERITIES = [1, 3, 5]
SPLITS = {"train": 0.50, "val": 0.15, "cal": 0.15, "test": 0.20}
TARGET_REJECT, TARGET_TRUST = 0.05, 0.01
IMAGE_SIZE = 224
TTA_SHIFT = 14  # px at 224; the same 6% of the width as the core's 2 px at 32

# Checkout policy amounts are DEMO SETTINGS, not recommendations.
CHECKOUT_POLICY = {"trust_one_tap_cap": 150.0, "confirmed_cap": 75.0}


def shop_path(*parts):
    """A path under data/shop/. Every write in shop/ goes through this."""
    p = DATA.joinpath(*parts).resolve()
    assert p == DATA.resolve() or DATA.resolve() in p.parents, f"shop/ may only write under data/shop/: {p}"
    return p
