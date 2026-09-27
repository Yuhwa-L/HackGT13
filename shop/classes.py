"""Resolve PRODUCT_CLASSES to ImageNet indices. Fails loudly with close matches; never guesses an index."""
import difflib
from functools import lru_cache

from shop.config import PRODUCT_CLASSES


@lru_cache(maxsize=None)
def imagenet_categories():
    from torchvision.models import ResNet18_Weights
    return tuple(ResNet18_Weights.IMAGENET1K_V1.meta["categories"])


@lru_cache(maxsize=None)
def product_indices():
    """ImageNet class index for each product class, in PRODUCT_CLASSES order."""
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
