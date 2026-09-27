"""Shopping-assistant HTTP routes, mounted into backend/main.py's standard-library server behind a guarded import.

GET  /api/shop/samples    curated photos (with images) + the demo story
GET  /api/shop/profiles   example (fictional) shoppers, style tags, risk levels, catalog names for "things I own"
POST /api/shop/identify   {photo_id} -> cached prediction, trust decision, candidates, reasons
POST /api/shop/upload     {image: "data:image/jpeg;base64,..."} -> live prediction + trust decision (a new photo_id)
GET  /api/shop/status     {"upload": "loading" | "ready" | "error", "upload_error"?}
GET  /api/shop/evaluation {test: data/shop/evaluation.json, real_photos: data/shop/real_photo_eval.json or null}
POST /api/shop/assist     {photo_id, profile | profile_id, user_message?, confirmed_class?} -> assistant response
                          (profile = the shopper's own {name, budget_per_item, style_tags, risk, owned}; the default)
POST /api/shop/quote      {photo_id, confirmed_class?, cart} -> checkout tier for this cart; places nothing
POST /api/shop/checkout   {photo_id, profile | profile_id, confirmed_class?, cart: [{product_id, qty}], confirmations}
                          -> a trust-trail receipt (evidence + HMAC fingerprint + mock one-time token)
POST /api/shop/verify     {receipt} -> {"valid": bool}: was this evidence signed by this server, unchanged?
GET  /shop.js             the tab's script

The server never trusts client-sent decisions or totals: both are recomputed from the cache and the catalog.
Mock checkout only: no payment data is ever collected.
"""
import json
from collections import OrderedDict
from pathlib import Path

from shop.assistant import assist
from shop.checkout_policy import checkout_requirements
from shop.catalog import STYLES
from shop.config import DATA, RISK_LEVELS
from shop.gate import effective_decision, is_allowed, shoppable_classes
from shop.live import LiveScorer
from shop.receipt import ReceiptSigner
from shop.profiles import normalize_profile

STATIC = Path(__file__).resolve().parent / "static"
MAX_QTY = 20
MAX_UPLOAD = 4_000_000   # bytes of JSON; the browser downsizes photos to ~800 px first
MAX_UPLOADS_KEPT = 100


class ShopAPI:
    MAX_UPLOAD = MAX_UPLOAD

    def __init__(self, data=DATA):
        cache = json.loads((data / "shop_cache.json").read_text())
        self.samples = {"meta": cache["meta"], "story": cache["story"], "photos": cache["photos"]}
        self.items = {v["photo_id"]: v for p in cache["photos"] for v in p["versions"]}
        self.catalog = json.loads((data / "catalog.json").read_text())["products"]
        self.by_id = {p["product_id"]: p for p in self.catalog}
        self.profiles = {p["profile_id"]: p for p in json.loads((data / "profiles.json").read_text())["profiles"]}
        self.uploads = OrderedDict()   # photo_id -> live-scored item, most recent last
        self.live = LiveScorer()
        self.thresholds = json.loads((data / "thresholds.json").read_text())
        ev, real = data / "evaluation.json", data / "real_photo_eval.json"   # results shown on the tab's evaluation card
        self.evaluation = {"test": json.loads(ev.read_text()) if ev.exists() else None,
                           "real_photos": json.loads(real.read_text()) if real.exists() else None,
                           "thresholds": self.thresholds}
        self.signer = ReceiptSigner()

    @classmethod
    def load(cls):
        """None (and a note) when the shop artifacts aren't built; the core demo keeps working."""
        try:
            api = cls()
            api.live.warm_up()   # loads torch + the model in the background; uploads say "loading" until ready
            return api
        except (OSError, KeyError, ValueError) as e:
            print(f"Shopping assistant tab disabled: {type(e).__name__}: {e}")
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
            if route == ("GET", "evaluation"):
                return self._ok(self.evaluation)
            if route == ("GET", "status"):
                st = self.live.status()
                return self._ok({"upload": st, **({"upload_error": self.live.error} if st == "error" else {})})
            if route == ("POST", "upload"):
                try:
                    item = self.live.score(req.get("image"))
                except RuntimeError as e:
                    return self._json(503, {"error": f"Live upload is unavailable: {e}"})
                self.uploads[item["photo_id"]] = item
                self.uploads.move_to_end(item["photo_id"])
                while len(self.uploads) > MAX_UPLOADS_KEPT:
                    self.uploads.popitem(last=False)
                return self._ok(item)
            if route == ("GET", "samples"):
                return self._ok(self.samples)
            if route == ("GET", "profiles"):
                return self._ok({"examples": list(self.profiles.values()), "style_tags": STYLES,
                                 "risk_levels": RISK_LEVELS,
                                 "catalog": [{k: p[k] for k in ("product_id", "name", "class")} for p in self.catalog]})
            if route == ("POST", "identify"):
                return self._ok(self._public(self._item(req)))
            if route == ("POST", "assist"):
                prof = self._profile(req)
                return self._ok(assist(self._item(req), prof, self.catalog, str(req.get("user_message") or "")[:500],
                                       self._confirmed(req), policy_cfg=RISK_LEVELS[prof.get("risk", "normal")]))
            if route == ("POST", "quote"):
                eff, lines, total, policy = self._price(req)
                return self._ok({**policy, "total": total, "effective_decision": eff})
            if route == ("POST", "checkout"):
                return self._checkout(req)
            if route == ("POST", "verify"):
                return self._ok({"valid": self.signer.verify(req.get("receipt"))})
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
        pid = str(req.get("photo_id"))
        item = self.items.get(pid) or self.uploads.get(pid)
        if item is None:
            raise LookupError("unknown photo_id")
        return item

    def _profile(self, req):
        """The shopper's own profile (default) or one of the fictional examples."""
        if req.get("profile") is not None:
            return normalize_profile(req["profile"], self.by_id)
        prof = self.profiles.get(str(req.get("profile_id")))
        if prof is None:
            raise LookupError("send a profile object or a known example profile_id")
        return {**prof, "risk": prof.get("risk", "normal")}

    @staticmethod
    def _confirmed(req):
        c = req.get("confirmed_class")
        return str(c) if c else None

    @staticmethod
    def _public(item):
        return {k: v for k, v in item.items() if k != "image"}

    def _price(self, req, profile=None):
        """Recompute the effective decision, cart lines and total server-side; never trust the client's."""
        item = self._item(req)
        cfg = RISK_LEVELS[(profile or self._profile(req)).get("risk", "normal")]
        eff = effective_decision(item["decision"], self._confirmed(req), item["candidates"])
        if not is_allowed(eff, "checkout"):  # gate first: a locked decision is a 403, whatever the cart holds
            raise PermissionError(checkout_requirements(eff, item["p_correct"], 0.0, cfg))
        classes = set(shoppable_classes(eff, item["pred_class"], item["candidates"], self._confirmed(req)))
        lines = []
        for line in req.get("cart") or []:
            p = self.by_id.get(str(line.get("product_id")))
            qty = int(line.get("qty", 1))
            if p is None or p["class"] not in classes or not 1 <= qty <= MAX_QTY:
                raise ValueError("cart has an unknown, out-of-class or invalid item")
            lines.append({"product_id": p["product_id"], "name": p["name"], "price": p["price"], "qty": qty})
        total = round(sum(line["price"] * line["qty"] for line in lines), 2)
        return eff, lines, total, checkout_requirements(eff, item["p_correct"], total, cfg)

    def _checkout(self, req):
        profile = self._profile(req)
        eff, lines, total, policy = self._price(req, profile)
        if not lines:
            return self._json(400, {"error": "cart is empty"})
        confirmations = int(req.get("confirmations") or 0)
        if not is_allowed(eff, "checkout") or not policy["allowed"] or confirmations < policy["confirmations_required"]:
            return self._json(403, {"tier": policy["tier"], "explanation": policy["explanation"],
                                    "confirmations_required": policy["confirmations_required"], "total": total})
        receipt = self.signer.issue(item=self._item(req), confirmed_class=self._confirmed(req), eff=eff, policy=policy,
                                    confirmations=confirmations, risk=profile.get("risk", "normal"), lines=lines,
                                    total=total, thresholds=self.thresholds)
        return self._ok({**receipt, "items": lines, "total": total, "tier": policy["tier"],
                         "confirmations_used": confirmations, "shopper": profile["name"], "explanation": policy["explanation"],
                         "note": "Mock order for the demo: nothing is charged and no payment data is collected."})

    @staticmethod
    def _json(code, obj):
        return code, "application/json", json.dumps(obj).encode()

    def _ok(self, obj):
        return self._json(200, obj)
