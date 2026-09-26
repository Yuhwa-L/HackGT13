"""Owner: C. Temperature scaling baselines: T_clean fit on clean val logits; T_corrupted fit per fold on val logits from
clean + seen families. Scaling changes how sharp the probabilities are, never the predicted class.
"""
import numpy as np
from scipy.optimize import minimize_scalar


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def fit_temperature(logits, labels):
    """Scalar T > 0 minimizing the NLL of softmax(logits / T) against the true labels."""
    z, labels = np.asarray(logits, np.float64), np.asarray(labels)

    def nll(log_t):
        s = z / np.exp(log_t)
        s = s - s.max(1, keepdims=True)
        return (np.log(np.exp(s).sum(1)) - s[np.arange(len(labels)), labels]).mean()

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def scaled_confidence(logits, T):
    """Top-1 probability of softmax(logits / T), computed in float64 (float32 softmax ties many confident rows)."""
    return softmax(np.asarray(logits, np.float64) / T).max(1)
