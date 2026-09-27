"""Trust-trail receipts: every agent purchase carries the evidence it was made on, and can be verified later.

The evidence records what the model saw and said, what the shopper confirmed, the trust layer's p_correct against the
thresholds in force, the checkout tier, and the confirmations given. It's fingerprinted with an HMAC-SHA256 under a
per-server secret: changing any field (say, bumping p_correct, or claiming TRUST) makes verification fail.

The payment is a MOCK one-time token bound to this order's fingerprint. No card number, account or payment data
exists anywhere in this system; the token only shows where a real network token would sit.
"""
import hashlib
import hmac
import json
import secrets
from datetime import datetime, timezone

EVIDENCE_FIELDS = ("photo_id", "model_said", "raw_confidence", "p_correct", "thresholds", "decision",
                   "shopper_confirmed", "effective_decision", "checkout_tier", "confirmations_required",
                   "confirmations_given", "careful_checkout", "items", "total", "time_utc")


def _norm(x):
    """Numbers as floats rounded to 6 places, so a browser round trip (which turns 75.0 into 75) can't change the hash."""
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if isinstance(x, (int, float)):
        return round(float(x), 6)
    if isinstance(x, dict):
        return {k: _norm(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    return x


def _canonical(evidence):
    return json.dumps(_norm({k: evidence[k] for k in EVIDENCE_FIELDS}), sort_keys=True, separators=(",", ":")).encode()


class ReceiptSigner:
    def __init__(self, secret=None):
        self._secret = secret or secrets.token_bytes(32)   # per server run: receipts verify on the server that issued them

    def fingerprint(self, evidence):
        return hmac.new(self._secret, _canonical(evidence), hashlib.sha256).hexdigest()

    def issue(self, *, item, confirmed_class, eff, policy, confirmations, risk, lines, total, thresholds, now=None):
        evidence = {
            "photo_id": item["photo_id"], "model_said": item["pred_class"],
            "raw_confidence": round(float(item["raw_confidence"]), 4), "p_correct": round(float(item["p_correct"]), 4),
            "thresholds": {"reject_below": round(float(thresholds["tau_reject"]), 4),
                           "trust_at_or_above": round(float(thresholds["tau_trust"]), 4)},
            "decision": item["decision"], "shopper_confirmed": confirmed_class if eff == "trust_confirmed" else None,
            "effective_decision": eff, "checkout_tier": policy["tier"],
            "confirmations_required": policy["confirmations_required"], "confirmations_given": confirmations,
            "careful_checkout": risk, "items": [{k: line[k] for k in ("product_id", "name", "price", "qty")} for line in lines],
            "total": total, "time_utc": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        }
        fp = self.fingerprint(evidence)
        return {
            "order_id": "MOCK-" + fp[:8].upper(),
            "evidence": evidence,
            "fingerprint": fp,
            "payment": {"method": "mock one-time token", "token": "mocktok_" + secrets.token_hex(8),
                        "bound_to": fp[:16], "single_use": True,
                        "note": "Demo only: no card or account data exists in this system, and nothing is charged."},
        }

    def verify(self, receipt):
        """True if the receipt's evidence is exactly what this server signed."""
        try:
            ev, fp = receipt["evidence"], str(receipt["fingerprint"])
            return hmac.compare_digest(self.fingerprint(ev), fp) and receipt.get("order_id") == "MOCK-" + fp[:8].upper()
        except (KeyError, TypeError, AttributeError):
            return False
