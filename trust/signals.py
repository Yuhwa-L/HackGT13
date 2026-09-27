"""Owner: B. Signals beyond the softmax: TTA agreement (tta_agree, tta_pconf, tta_std), kNN distance
(cosine distance to the 10th-nearest clean train embedding), Mahalanobis distance to the predicted-class mean (shared covariance)
and the optional Trust Score (Jiang et al. 2018).

All functions take plain arrays and compute in float64, so float32 inputs from A don't create ties (see person-c-plan §3.5).
"""
import numpy as np
from scipy.linalg import pinvh

N_CLASSES = 10
KNN_K = 10
CHUNK = 2048  # rows per block for the kNN / Mahalanobis loops; a 2048 x 50000 float32 similarity block is ~400 MB


def softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def softmax_stats(logits):
    """pred (argmax), raw_confidence (top-1 prob), entropy (nats), margin (top-1 minus top-2 prob)."""
    p = softmax(logits)
    top2 = np.sort(p, axis=1)[:, -2:]
    entropy = -(p * np.log(np.clip(p, 1e-300, None))).sum(axis=1)
    return {"pred": p.argmax(axis=1), "raw_confidence": top2[:, 1], "entropy": entropy,
            "margin": top2[:, 1] - top2[:, 0]}


def tta_stats(logits, tta_logits):
    """Stability of the original prediction k under the 3 TTA views (N x 3 x 10).

    tta_agree = share of views whose argmax is k; tta_pconf = mean p_k over the views;
    tta_std = std of p_k over the original plus the views.
    """
    p = softmax(logits)
    pv = softmax(tta_logits)
    k = p.argmax(axis=1)
    pk_views = np.take_along_axis(pv, k[:, None, None], axis=2)[:, :, 0]  # N x 3
    pk_all = np.column_stack([p[np.arange(len(k)), k], pk_views])        # N x 4
    return {"tta_agree": (pv.argmax(axis=2) == k[:, None]).mean(axis=1),
            "tta_pconf": pk_views.mean(axis=1),
            "tta_std": pk_all.std(axis=1)}


def _unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def knn_dist(emb, bank, k=KNN_K, chunk=CHUNK):
    """Cosine distance from each row of emb to its k-th nearest row of bank (Sun et al. 2022).

    The matmul runs in float32 (it dominates the cost); the distance is returned as float64.
    """
    q, b = _unit(emb), _unit(bank)
    out = np.empty(len(q))
    for s in range(0, len(q), chunk):
        sim = q[s:s + chunk] @ b.T
        out[s:s + chunk] = 1.0 - np.partition(sim, -k, axis=1)[:, -k]
    return out


def bank_signals(emb, pred, bank, bank_labels, k=KNN_K, chunk=CHUNK, n_classes=N_CLASSES):
    """knn_dist plus trust_score from one pass over the train bank (the similarity matmul is the expensive part).

    trust_score (Jiang et al. 2018, no density filtering): distance to the nearest bank embedding of any other class
    divided by distance to the nearest one of the predicted class. Euclidean on unit-normalized embeddings, like
    knn_dist, so it ignores activation scale. Higher = more trustworthy; above 1 means the predicted class is nearest.
    """
    q = _unit(emb)
    order = np.argsort(bank_labels, kind="stable")
    b = _unit(bank)[order]
    starts = np.searchsorted(bank_labels[order], np.arange(n_classes))
    knn, trust = np.empty(len(q)), np.empty(len(q))
    for s in range(0, len(q), chunk):
        sim = q[s:s + chunk] @ b.T
        knn[s:s + chunk] = 1.0 - np.partition(sim, -k, axis=1)[:, -k]
        nearest = np.maximum.reduceat(sim, starts, axis=1).astype(np.float64)  # best similarity per class
        d = np.sqrt(np.maximum(2.0 - 2.0 * nearest, 0.0))
        rows, p = np.arange(len(d)), pred[s:s + chunk]
        d_pred = d[rows, p].copy()
        d[rows, p] = np.inf
        trust[s:s + chunk] = d.min(axis=1) / np.maximum(d_pred, 1e-6)
    return {"knn_dist": knn, "trust_score": trust}


def fit_mahalanobis(bank, bank_labels, n_classes=10, ridge=1e-6):
    """Class means and the shared (tied) precision matrix from the train reference bank."""
    x = np.asarray(bank, dtype=np.float64)
    means = np.stack([x[bank_labels == c].mean(axis=0) for c in range(n_classes)])
    centered = x - means[bank_labels]
    cov = centered.T @ centered / len(x)
    cov += ridge * np.trace(cov) / len(cov) * np.eye(len(cov))  # ridge scaled to the feature variance
    return means, pinvh(cov)


def maha_pred(emb, pred, means, precision, chunk=CHUNK):
    """Mahalanobis distance from each embedding to the mean of its predicted class."""
    out = np.empty(len(emb))
    for s in range(0, len(emb), chunk):
        d = np.asarray(emb[s:s + chunk], dtype=np.float64) - means[pred[s:s + chunk]]
        out[s:s + chunk] = np.sqrt(np.maximum(((d @ precision) * d).sum(axis=1), 0.0))
    return out
