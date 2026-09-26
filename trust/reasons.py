"""SHAP reasons for a single prediction. Owner: C.

SIGN CONVENTION: the XGBoost model predicts FAILURE, while the UI shows p_correct. A POSITIVE
pred_contribs value pushes TOWARD failure (lower p_correct). Reasons are the top features with the
largest positive contributions.
"""
from __future__ import annotations

# One sentence per MODEL_FEATURES entry, phrased for the shopper/judge. TODO(C/D): finalize copy.
REASON_TEXT: dict[str, str] = {
    "raw_confidence": "TODO: copy for raw_confidence",
    "entropy": "TODO: copy for entropy",
    "margin": "TODO: copy for margin",
    "tta_agree": "TODO: copy for tta_agree",
    "tta_pconf": "TODO: copy for tta_pconf",
    "tta_std": "TODO: copy for tta_std",
    "knn_dist": "TODO: copy for knn_dist",
    "maha_pred": "TODO: copy for maha_pred",
}


def shap_reasons(booster, X_row, feature_names: list[str], top_n: int = 3) -> list[dict]:
    """booster.predict(DMatrix, pred_contribs=True) -> top_n [{signal, text, shap}] pushing toward
    failure (schemas.Reason). Drop the bias column. TODO(C).
    """
    raise NotImplementedError("TODO(C): shap_reasons")
