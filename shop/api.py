"""Snap-to-Shop HTTP routes, mounted into backend/main.py's standard-library server behind a guarded import.

GET  /api/shop/samples    curated photos (with images) + the demo story
GET  /api/shop/profiles   fictional shoppers
POST /api/shop/identify   {photo_id} -> cached prediction, trust decision, candidates, reasons
POST /api/shop/assist     {photo_id, profile_id, user_message?, confirmed_class?} -> assistant response
POST /api/shop/quote      {photo_id, confirmed_class?, cart} -> checkout tier for this cart; places nothing
POST /api/shop/checkout   {photo_id, profile_id, confirmed_class?, cart: [{product_id, qty}], confirmations}
GET  /shop.js             the tab's script

The server never trusts client-sent decisions or totals: both are recomputed from the cache and the catalog.
Mock checkout only: no payment data is ever collected.
"""
import hashlib
import json
from pathlib import Path

from shop.assistant import assist
from shop.checkout_policy import checkout_requirements
from shop.config import CHECKOUT_POLICY, DATA
from shop.gate import effective_decision, is_allowed, shoppable_classes

STATIC = Path(__file__).resolve().parent / "static"
MAX_QTY = 20


class ShopAPI:
    def __init__(self, data=DATA):
        cache = json.loads((data / "shop_cache.json").read_text())
        self.samples = {"meta": cache["meta"], "story": cache["story"], "photos": cache["photos"]}
        self.items = {v["photo_id"]: v for p in cache["photos"] for v in p["versions"]}
        self.catalog = json.loads((data / "catalog.json").read_text())["products"]
        self.by_id = {p["product_id"]: p for p in self.catalog}
        self.profiles = {p["profile_id"]: p for p in json.loads((data / "profiles.json").read_text())["profiles"]}

    @classmethod
    def load(cls):
        """None (and a note) when the shop artifacts aren't built; the core demo keeps working."""
        try:
            return cls()
        except (OSError, KeyError, ValueError) as e:
            print(f"Snap-to-Shop tab disabled: {type(e).__name__}: {e}")
            return None

    # ---- routing: returns (status, content_type, bytes) or None when the path isn't a shop route ----
    def handle(self, method, path, body=b""):
        path = path.split("?")[0]
        if method == "GET" and path == "/shop.js":
            return 200, "text/javascript; charset=utf-8", (STATIC / "shop.js").read_bytes()
        if not path.startswith("/api/shop/"):
            return None
        route = (method, path[len("/api/shop/"):])
        try:
            req = json.loads(body or b"{}") if method == "POST" else {}
            if not isinstance(req, dict):
                raise ValueError("body must be a JSON object")
            if route == ("GET", "samples"):
                return self._ok(self.samples)
            if route == ("GET", "profiles"):
                return self._ok({"profiles": list(self.profiles.values())})
            if route == ("POST", "identify"):
                return self._ok(self._public(self._item(req)))
            if route == ("POST", "assist"):
                return self._ok(assist(self._item(req), self._profile(req), self.catalog,
                                       str(req.get("user_message") or "")[:500], self._confirmed(req)))
            if route == ("POST", "quote"):
                eff, lines, total, policy = self._price(req)
                return self._ok({**policy, "total": total, "effective_decision": eff})
            if route == ("POST", "checkout"):
                return self._checkout(req)
        except PermissionError as e:
            pol = e.args[0]
            if route == ("POST", "quote"):
                return self._ok({**pol, "total": 0.0})
            return self._json(403, {k: pol[k] for k in ("tier", "explanation", "confirmations_required")})
        except LookupError as e:
            return self._json(404, {"error": str(e)})
        except (ValueError, TypeError) as e:
            return self._json(400, {"error": str(e)})
        return self._json(404, {"error": f"unknown shop endpoint {method} {path}"})

    # ---- helpers ----
    def _item(self, req):
        item = self.items.get(str(req.get("photo_id")))
        if item is None:
            raise LookupError("unknown photo_id")
        return item

    def _profile(self, req):
        prof = self.profiles.get(str(req.get("profile_id")))
        if prof is None:
            raise LookupError("unknown profile_id")
        return prof

    @staticmethod
    def _confirmed(req):
        c = req.get("confirmed_class")
        return str(c) if c else None

    @staticmethod
    def _public(item):
        return {k: v for k, v in item.items() if k != "image"}

    def _price(self, req):
        """Recompute the effective decision, cart lines and total server-side; never trust the client's."""
        item = self._item(req)
        eff = effective_decision(item["decision"], self._confirmed(req), item["candidates"])
        if not is_allowed(eff, "checkout"):  # gate first: a locked decision is a 403, whatever the cart holds
            raise PermissionError(checkout_requirements(eff, item["p_correct"], 0.0, CHECKOUT_POLICY))
        classes = set(shoppable_classes(eff, item["pred_class"], item["candidates"], self._confirmed(req)))
        lines = []
        for line in req.get("cart") or []:
            p = self.by_id.get(str(line.get("product_id")))
            qty = int(line.get("qty", 1))
            if p is None or p["class"] not in classes or not 1 <= qty <= MAX_QTY:
                raise ValueError("cart has an unknown, out-of-class or invalid item")
            lines.append({"product_id": p["product_id"], "name": p["name"], "price": p["price"], "qty": qty})
        total = round(sum(line["price"] * line["qty"] for line in lines), 2)
        return eff, lines, total, checkout_requirements(eff, item["p_correct"], total, CHECKOUT_POLICY)

    def _checkout(self, req):
        profile = self._profile(req)
        eff, lines, total, policy = self._price(req)
        if not lines:
            return self._json(400, {"error": "cart is empty"})
        confirmations = int(req.get("confirmations") or 0)
        if not is_allowed(eff, "checkout") or not policy["allowed"] or confirmations < policy["confirmations_required"]:
            return self._json(403, {"tier": policy["tier"], "explanation": policy["explanation"],
                                    "confirmations_required": policy["confirmations_required"], "total": total})
        order_id = "MOCK-" + hashlib.sha256(json.dumps([req.get("photo_id"), lines]).encode()).hexdigest()[:8].upper()
        return self._ok({"order_id": order_id, "items": lines, "total": total, "tier": policy["tier"],
                         "confirmations_used": confirmations, "shopper": profile["name"], "explanation": policy["explanation"],
                         "note": "Mock order for the demo: nothing is charged and no payment data is collected."})

    @staticmethod
    def _json(code, obj):
        return code, "application/json", json.dumps(obj).encode()

    def _ok(self, obj):
        return self._json(200, obj)
