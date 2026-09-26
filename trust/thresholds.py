"""Owner: C. TRUST / CAUTION / REJECT thresholds chosen on the cal split only (5% error -> tau_reject, 1% -> tau_trust),
plus the broken-promise check (what a rule set on one split actually delivers on another).
"""
import numpy as np


def tau_for(conf, correct, target):
    """Loosest threshold (largest coverage) whose accepted set {conf >= tau} has error <= target.
    Cuts only between distinct conf values; returns inf (accept nothing) if no cut meets the target."""
    conf = np.asarray(conf)
    order = np.argsort(-conf, kind="stable")
    c, err = conf[order], 1 - np.asarray(correct, float)[order]
    ends = np.r_[np.nonzero(c[1:] != c[:-1])[0], len(c) - 1]           # last index of each tie group
    ok = np.nonzero(np.cumsum(err)[ends] / (ends + 1) <= target)[0]
    return float(c[ends[ok[-1]]]) if len(ok) else float("inf")


def decide(p_correct, tau_reject, tau_trust):
    """REJECT below tau_reject, TRUST at or above tau_trust, CAUTION in between."""
    p = np.asarray(p_correct)
    return np.where(p < tau_reject, "reject", np.where(p >= tau_trust, "trust", "caution"))


def realized(conf, correct, tau, target):
    """Coverage and error that the rule 'accept if conf >= tau' actually delivers on another set."""
    accepted = np.asarray(conf) >= tau
    return dict(tau=tau, target_error=target, coverage=float(accepted.mean()),
                error=float(1 - np.asarray(correct, float)[accepted].mean()) if accepted.any() else None)
