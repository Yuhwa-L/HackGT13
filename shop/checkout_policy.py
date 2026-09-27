"""Risk-adjusted checkout: how much friction a purchase gets, from the effective trust decision. Pure and tested.
The cap amounts come from shop.config.CHECKOUT_POLICY and are demo settings.
"""


def checkout_requirements(effective_decision, p_correct, order_total, cfg):
    """-> {"allowed", "tier", "max_one_tap_total", "confirmations_required", "explanation"}"""
    one_tap_cap, confirmed_cap = float(cfg["trust_one_tap_cap"]), float(cfg["confirmed_cap"])
    pct = f"p_correct {p_correct:.2f}"
    if effective_decision == "trust":
        if one_tap_cap > 0 and order_total <= one_tap_cap:  # a zero cap turns one-tap off
            return _req(True, "one_tap", one_tap_cap, 0,
                        f"One-tap checkout: the trust layer verified this identification ({pct}).")
        if one_tap_cap <= 0:
            return _req(True, "confirm", 0.0, 1,
                        f"Verified identification ({pct}); your careful-checkout setting confirms every purchase.")
        return _req(True, "confirm", one_tap_cap, 1,
                    f"Verified identification ({pct}), but the order is over the ${one_tap_cap:.0f} one-tap cap: "
                    "confirm once.")
    if effective_decision == "trust_confirmed":
        if confirmed_cap > 0 and order_total <= confirmed_cap:
            return _req(True, "confirm", 0.0, 1,
                        f"You picked the item yourself after the model was unsure ({pct}): confirm once.")
        over = (f"and the order is over ${confirmed_cap:.0f}" if confirmed_cap > 0
                else "and your careful-checkout setting asks for two confirmations")
        return _req(True, "confirm_twice", 0.0, 2,
                    f"You picked the item yourself after the model was unsure ({pct}), {over}: confirm twice.")
    if effective_decision == "caution":
        return _req(False, "blocked_until_confirmed", 0.0, None,
                    f"The model isn't sure what this is ({pct}). Pick the right item to unlock checkout.")
    if effective_decision == "reject":
        return _req(False, "blocked", 0.0, None,
                    f"The photo can't be identified reliably ({pct}). Checkout is locked; try a clearer photo.")
    raise ValueError(f"unknown effective decision {effective_decision!r}")


def _req(allowed, tier, cap, confirmations, explanation):
    return {"allowed": allowed, "tier": tier, "max_one_tap_total": cap, "confirmations_required": confirmations,
            "explanation": explanation}
