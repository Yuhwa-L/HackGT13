"""Shopper profiles. The default is the shopper's own profile, sent by the browser with each request and validated
here; a few fictional example profiles (data/shop/profiles.json) remain as a proof of concept."""
from shop.catalog import STYLES
from shop.config import RISK_LEVELS

MAX_OWNED = 12


def normalize_profile(raw, catalog_ids):
    """A user-defined profile from the browser -> a validated profile dict. Raises ValueError on bad input."""
    if not isinstance(raw, dict):
        raise ValueError("profile must be an object")
    name = " ".join(str(raw.get("name") or "").split())[:40] or "Shopper"
    try:
        budget = float(raw.get("budget_per_item"))
    except (TypeError, ValueError):
        raise ValueError("budget_per_item must be a number")
    if not 1 <= budget <= 10000:
        raise ValueError("budget_per_item must be between 1 and 10000")
    styles = [s for s in dict.fromkeys(raw.get("style_tags") or []) if s in STYLES][:4]
    risk = raw.get("risk") or "normal"
    if risk not in RISK_LEVELS:
        raise ValueError(f"risk must be one of {sorted(RISK_LEVELS)}")
    owned = [str(p) for p in dict.fromkeys(raw.get("owned") or []) if str(p) in catalog_ids][:MAX_OWNED]
    return {"profile_id": "custom", "name": name, "budget_per_item": round(budget, 2), "style_tags": styles,
            "risk": risk, "purchase_history": [{"product_id": p, "date": ""} for p in owned], "custom": True}
