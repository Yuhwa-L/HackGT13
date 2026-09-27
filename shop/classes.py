"""Resolve PRODUCT_CLASSES to ImageNet indices. Fails loudly with close matches; never guesses an index.
The result is cached in data/shop/product_indices.json so torch-free processes (the trust stage, the server) never
need torchvision."""
import difflib
import json
from functools import lru_cache

from shop.config import PRODUCT_CLASSES, shop_path


@lru_cache(maxsize=None)
def imagenet_categories():
    from torchvision.models import ResNet18_Weights
    return tuple(ResNet18_Weights.IMAGENET1K_V1.meta["categories"])


@lru_cache(maxsize=None)
def product_indices():
    """ImageNet class index for each product class, in PRODUCT_CLASSES order."""
    cache = shop_path("product_indices.json")
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved["classes"] == PRODUCT_CLASSES:
            return tuple(saved["indices"])
    idx = _resolve()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"classes": PRODUCT_CLASSES, "indices": list(idx)}, indent=1))
    return idx


def _resolve():
    cats = imagenet_categories()
    out, bad = [], []
    for name in PRODUCT_CLASSES:
        hits = [i for i, c in enumerate(cats) if c == name]
        if len(hits) == 1:
            out.append(hits[0])
        else:
            bad.append(f"{name!r}: {'ambiguous' if hits else 'no exact match'}; close: "
                       f"{difflib.get_close_matches(name, cats, 5, 0.5)}")
    if bad:
        raise SystemExit("Fix PRODUCT_CLASSES in shop/config.py:\n  " + "\n  ".join(bad))
    return tuple(out)


if __name__ == "__main__":
    for name, i in zip(PRODUCT_CLASSES, product_indices()):
        print(f"{i:4d}  {name}")
