"""Owner: C. Metrics: AURC (primary), AUROC pooled and within corruption x severity, ECE + reliability bins,
error at fixed coverage, risk-coverage curves, bootstrap CIs over base images.

Conventions: `conf` ranks predictions (higher = more trusted), `correct` is 0/1, `w` are optional row weights
(bootstrap resample counts). Ties in `conf` are never broken arbitrarily: AURC takes the risk at the end of each
tie group and AUROC counts ties as 1/2.
"""
import numpy as np


def _rank_prep(conf):
    """Descending sort order and the tie-group id of each sorted position; reused across bootstrap resamples."""
    conf = np.asarray(conf)
    order = np.argsort(-conf, kind="stable")
    c = conf[order]
    return order, np.cumsum(np.r_[True, c[1:] != c[:-1]]) - 1


def _aurc_auroc(prep, correct, w):
    order, group = prep
    n_groups = group[-1] + 1
    total = np.bincount(group, w[order], n_groups)                     # weight per tie group
    bad = np.bincount(group, (w * (1 - correct))[order], n_groups)     # weighted errors per tie group
    good = total - bad
    cum_w, cum_bad = np.cumsum(total), np.cumsum(bad)
    risk = np.divide(cum_bad, cum_w, out=np.zeros(n_groups), where=cum_w > 0)
    aurc = float(np.sum(total / cum_w[-1] * risk))
    n_good, n_bad = good.sum(), bad.sum()
    if n_good == 0 or n_bad == 0:
        return aurc, None
    bad_below = n_bad - cum_bad                                        # wrong predictions with strictly lower conf
    return aurc, float(np.sum(good * (bad_below + 0.5 * bad)) / (n_good * n_bad))


def _both(conf, correct, w=None):
    correct = np.asarray(correct, float)
    w = np.ones_like(correct) if w is None else np.asarray(w, float)
    return _aurc_auroc(_rank_prep(conf), correct, w)


def aurc(conf, correct, w=None):
    """Area under the risk-coverage curve (lower is better)."""
    return _both(conf, correct, w)[0]


def auroc(conf, correct, w=None):
    """P(conf of a correct prediction > conf of a wrong one), ties count 1/2. None if only one class is present."""
    return _both(conf, correct, w)[1]


def auroc_within(conf, correct, groups):
    """Mean AUROC inside each group (e.g. corruption x severity); single-class groups are skipped."""
    conf, correct, groups = np.asarray(conf), np.asarray(correct), np.asarray(groups)
    vals = [auroc(conf[groups == g], correct[groups == g]) for g in np.unique(groups)]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _bins(prob, correct, bins):
    prob = np.asarray(prob, float)
    b = np.minimum((prob * bins).astype(int), bins - 1)
    return np.bincount(b, minlength=bins), np.bincount(b, prob, bins), np.bincount(b, np.asarray(correct, float), bins)


def ece(prob, correct, bins=15):
    """Expected calibration error with equal-width bins."""
    n, sum_prob, sum_correct = _bins(prob, correct, bins)
    return float(np.abs(sum_correct - sum_prob).sum() / n.sum())


def reliability(prob, correct, bins=15):
    """Non-empty bins as [{lo, hi, n, conf, acc}] for the reliability diagram."""
    n, sum_prob, sum_correct = _bins(prob, correct, bins)
    return [dict(lo=i / bins, hi=(i + 1) / bins, n=int(n[i]), conf=float(sum_prob[i] / n[i]), acc=float(sum_correct[i] / n[i]))
            for i in range(bins) if n[i]]


def err_at_coverage(conf, correct, coverage):
    """Error rate among the `coverage` fraction of most-trusted predictions."""
    correct = np.asarray(correct, float)
    k = max(1, int(round(coverage * len(correct))))
    return float(1 - correct[np.argsort(-np.asarray(conf), kind="stable")[:k]].mean())


def rc_curve(conf, correct, points=50):
    """[[coverage, selective risk], ...] at `points` evenly spaced coverages, for the risk-coverage chart."""
    err = 1 - np.asarray(correct, float)[np.argsort(-np.asarray(conf), kind="stable")]
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    ks = np.unique(np.linspace(1, len(err), points).astype(int))
    return [[float(k / len(err)), float(risk[k - 1])] for k in ks]


def summarize(rank, prob, correct, groups):
    """All per-method metrics. Ranking metrics use `rank`, calibration metrics use `prob` (the same array for baselines)."""
    a, r = _both(rank, correct)
    return dict(aurc=a, auroc=r, auroc_within=auroc_within(rank, correct, groups), ece=ece(prob, correct),
                mean_conf=float(np.mean(prob)), err_at_20=err_at_coverage(rank, correct, .2),
                err_at_50=err_at_coverage(rank, correct, .5))


def paired_bootstrap(conf_a, conf_b, correct, base_ids, n=1000, seed=0):
    """metric(a) - metric(b) for AURC and AUROC with 95% CIs, resampling base images (not rows) with replacement."""
    correct = np.asarray(correct, float)
    prep_a, prep_b = _rank_prep(conf_a), _rank_prep(conf_b)
    _, inv = np.unique(base_ids, return_inverse=True)
    n_base = inv.max() + 1
    rng = np.random.default_rng(seed)
    d_aurc, d_auroc = [], []
    for _ in range(n):
        w = np.bincount(rng.integers(0, n_base, n_base), minlength=n_base)[inv].astype(float)
        (a1, r1), (a2, r2) = _aurc_auroc(prep_a, correct, w), _aurc_auroc(prep_b, correct, w)
        d_aurc.append(a1 - a2)
        if r1 is not None and r2 is not None:
            d_auroc.append(r1 - r2)
    ci = lambda d: [float(x) for x in np.percentile(d, [2.5, 97.5])] if d else None
    (a1, r1), (a2, r2) = _both(conf_a, correct), _both(conf_b, correct)
    return dict(aurc_diff=a1 - a2, aurc_diff_ci=ci(d_aurc),
                auroc_diff=None if r1 is None or r2 is None else r1 - r2, auroc_diff_ci=ci(d_auroc))
