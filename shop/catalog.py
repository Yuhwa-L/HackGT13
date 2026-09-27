"""Fictional catalog and shopper profiles, deterministic from the seed. Every brand, product and shopper is invented.

Run: python -m shop.catalog   -> data/shop/catalog.json, data/shop/profiles.json
"""
import json

import numpy as np

from shop.config import PRODUCT_CLASSES, SEED, shop_path

BRANDS = ["Northloop", "Marlowe & Pine", "Quillfield", "Tessary", "Ardent Co.", "Brisk Harbor", "Lumen Row", "Oakvale"]
STYLES = ["outdoor", "minimal", "classic", "sporty", "tech", "cozy", "luxe", "budget"]
PRICE = {  # (low, high) demo price range per class, USD
    "backpack": (35, 160), "sunglasses": (20, 180), "running shoe": (60, 170), "sandal": (20, 90), "Loafer": (50, 190),
    "sweatshirt": (25, 90), "jean": (30, 120), "cardigan": (30, 130), "sock": (6, 25), "wallet": (15, 110),
    "purse": (30, 240), "digital watch": (25, 220), "coffee mug": (8, 35), "water bottle": (10, 45),
    "teapot": (15, 80), "frying pan": (20, 140), "toaster": (25, 120), "espresso maker": (60, 450),
    "laptop": (380, 1800), "computer keyboard": (20, 180), "mouse": (10, 90), "cellular telephone": (150, 1100),
    "iPod": (60, 250), "binoculars": (40, 300), "acoustic guitar": (120, 900), "electric guitar": (180, 1200),
    "umbrella": (12, 60), "sleeping bag": (40, 220), "lipstick": (8, 40), "perfume": (25, 160),
}
ADJ = {"outdoor": "Trail", "minimal": "Essential", "classic": "Heritage", "sporty": "Tempo", "tech": "Smart",
       "cozy": "Hearth", "luxe": "Signature", "budget": "Everyday"}


def _pretty(cls):
    return {"Loafer": "loafer", "jean": "jeans", "cellular telephone": "phone", "iPod": "music player"}.get(cls, cls)


def make_catalog(seed=SEED, per_class=4):
    rng = np.random.default_rng(seed)
    items = []
    for c, cls in enumerate(PRODUCT_CLASSES):
        lo, hi = PRICE[cls]
        prices = np.sort(np.round(rng.uniform(lo, hi, per_class) / 5) * 5 - 0.01)
        for k, price in enumerate(prices):
            tags = [str(t) for t in rng.choice([s for s in STYLES if s != "budget"], 2, replace=False)]
            if price <= lo + 0.3 * (hi - lo):  # the cheapest items in a class are the budget picks, never "luxe"
                tags = [tags[0] if tags[0] != "luxe" else "classic", "budget"]
            brand = str(rng.choice(BRANDS))
            name = f"{brand} {ADJ[tags[0]]} {_pretty(cls).title()}"
            items.append({"product_id": f"p{c:02d}{k}", "class": cls, "name": name, "brand": brand,
                          "price": float(max(price, 4.99)), "style_tags": tags,
                          "description": f"A {tags[0]}, {tags[1]} {_pretty(cls)} from {brand}.", "fictional": True})
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
    shop_path("catalog.json").write_text(json.dumps({"fictional": True, "products": catalog}, indent=1))
    shop_path("profiles.json").write_text(json.dumps({"fictional": True, "profiles": profiles}, indent=1))
    print(f"wrote {len(catalog)} fictional products and {len(profiles)} fictional profiles")


if __name__ == "__main__":
    main()
