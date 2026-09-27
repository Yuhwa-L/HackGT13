"""What the shopping assistant may do for a trust decision. The source of truth: pure functions, no LLM involved.

CAUTION upgrades to "trust_confirmed" only when the shopper confirms a class the model actually proposed.
"trust_confirmed" allows the cart like "trust" does, but the checkout policy treats it more strictly.
"""
ALLOWED = {
    "trust": {"recommend", "add_to_cart", "checkout"},
    "trust_confirmed": {"recommend", "add_to_cart", "checkout"},
    "caution": {"compare", "ask_disambiguation", "recommend"},
    "reject": {"ask_retake"},
}
DEFAULT_ACTION = {"trust": "recommend", "trust_confirmed": "recommend", "caution": "ask_disambiguation",
                  "reject": "ask_retake"}


def effective_decision(decision, confirmed_class=None, candidates=()):
    """decision in {trust, caution, reject}; candidates = [{"class": ..., "prob": ...}, ...]."""
    if decision not in ("trust", "caution", "reject"):
        raise ValueError(f"unknown decision {decision!r}")
    if decision == "caution" and confirmed_class is not None and confirmed_class in {c["class"] for c in candidates}:
        return "trust_confirmed"
    return decision


def allowed_actions(effective):
    return sorted(ALLOWED[effective])


def is_allowed(effective, action):
    return action in ALLOWED[effective]


def cart_allowed(effective):
    return is_allowed(effective, "add_to_cart")


def shoppable_classes(effective, predicted_class, candidates=(), confirmed_class=None):
    """Classes whose products the assistant may show: the prediction when trusted, the confirmed class once
    confirmed, the candidate classes while comparing under CAUTION, nothing under REJECT."""
    if effective == "trust":
        return [predicted_class]
    if effective == "trust_confirmed":
        return [confirmed_class]
    if effective == "caution":
        return [c["class"] for c in candidates]
    return []
