"""Per-prediction signals. Owner: B.

Shop reuses these with num_classes=30, so never assume 10 classes.
"""
from __future__ import annotations


def softmax_stats(logits):
    """logits (N,C) -> dict of (N,) arrays: raw_confidence, top2_prob, entropy, margin.

    margin = top1_prob - top2_prob. entropy in nats. TODO(B).
    """
    raise NotImplementedError("TODO(B): softmax_stats")


def tta_signals(orig_logits, tta_logits):
    """orig (N,C), tta (N,V,C) -> dict of (N,) arrays:
      tta_agree  share of TTA views whose argmax == original argmax
      tta_pconf  mean prob of the ORIGINAL predicted class across TTA views
      tta_std    std of that prob across original + TTA views
    TODO(B).
    """
    raise NotImplementedError("TODO(B): tta_signals")


def knn_distance(emb, bank, k: int, chunk_size: int = 4096):
    """Cosine distance to the k-th nearest bank embedding. (N,) float.

    L2-normalize both; chunked matmul in torch (bank is 50k x 512). No sklearn brute force. TODO(B).
    """
    raise NotImplementedError("TODO(B): knn_distance")


def mahalanobis_pred(emb, bank, bank_labels, pred_labels, ridge: float):
    """Class means + shared covariance (+ ridge*I) fit on the bank; distance of each emb to the mean
    of ITS predicted class. (N,) float. TODO(B).
    """
    raise NotImplementedError("TODO(B): mahalanobis_pred")


def trust_score(*args, **kwargs):
    """Optional extra signal. TODO(B): decide whether to implement."""
    raise NotImplementedError("TODO(B): trust_score (optional)")
