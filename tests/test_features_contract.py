from common.schemas import ANALYSIS_ONLY_COLUMNS, FEATURES_COLUMNS
from trust.features import MODEL_FEATURES


def test_analysis_only_columns_not_model_features():
    for col in ("corruption", "family", "severity", "split", "base_image_id"):
        assert col not in MODEL_FEATURES
    assert not set(MODEL_FEATURES) & ANALYSIS_ONLY_COLUMNS


def test_model_features_exist_in_features_csv():
    assert set(MODEL_FEATURES) <= set(FEATURES_COLUMNS)
