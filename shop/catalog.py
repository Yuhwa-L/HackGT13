"""Catalog of real products (shop/catalog_data.py, approximate prices) and fictional shopper profiles.

Products are real, widely sold models listed from general knowledge: no web lookups or API calls. Prices are
approximate typical list prices and may be out of date; the UI says so. Not affiliated with any brand; nothing is sold.
Shoppers are invented. Both are deterministic.

Run: python -m shop.catalog   -> data/shop/catalog.json, data/shop/profiles.json
"""
import json
from urllib.parse import quote_plus

import numpy as np

from shop.catalog_data import REAL_PRODUCTS
from shop.config import PRODUCT_CLASSES, SEED, shop_path

STYLES = ["outdoor", "minimal", "classic", "sporty", "tech", "cozy", "luxe", "budget"]
PRICE_NOTE = "Approximate list price; may be out of date."


def make_catalog():
    missing = [c for c in PRODUCT_CLASSES if not REAL_PRODUCTS.get(c)]
    assert not missing, f"no products listed for {missing}"
    items = []
    for c, cls in enumerate(PRODUCT_CLASSES):
        for k, (brand, model, price, tags, spec) in enumerate(REAL_PRODUCTS[cls]):
            assert set(tags) <= set(STYLES), (brand, model, tags)
            name = f"{brand} {model}"
            items.append({"product_id": f"p{c:02d}{k}", "class": cls, "name": name, "brand": brand,
                          "price": float(price), "style_tags": list(tags), "description": spec,
                          "search_url": "https://www.google.com/search?q=" + quote_plus(name),
                          "real": True, "approx_price": True})
    return items


def make_profiles(catalog, seed=SEED):
    rng = np.random.default_rng(seed + 1)
    people = [("maya", "Maya", 120.0, ["outdoor", "sporty"]), ("jordan", "Jordan", 60.0, ["budget", "minimal"]),
              ("priya", "Priya", 400.0, ["luxe", "tech"])]
    out = []
    for pid, name, budget, styles in people:
        liked = [p for p in catalog if set(p["style_tags"]) & set(styles) and p["price"] <= budget * 1.2]
        pick = rng.choice(len(liked), min(6, len(liked)), replace=False)
        history = [{"product_id": liked[i]["product_id"], "date": f"2026-0{1 + j % 8}-{10 + j:02d}"}
                   for j, i in enumerate(sorted(pick))]
        out.append({"profile_id": pid, "name": name, "budget_per_item": budget, "style_tags": styles,
                    "purchase_history": history, "fictional": True})
    return out


def main():
    catalog = make_catalog()
    profiles = make_profiles(catalog)
    shop_path().mkdir(parents=True, exist_ok=True)
    shop_path("catalog.json").write_text(json.dumps({"real_products": True, "price_note": PRICE_NOTE,
                                                     "products": catalog}, indent=1))
    shop_path("profiles.json").write_text(json.dumps({"fictional": True, "profiles": profiles}, indent=1))
    print(f"wrote {len(catalog)} real products (approximate prices) and {len(profiles)} fictional profiles")


if __name__ == "__main__":
    main()
