"""Infrastructure tests — these run for real today."""
from common import io, schemas
from common.config import load_config


def test_config_loads():
    cfg = load_config()
    assert abs(sum(cfg.splits.fractions.values()) - 1.0) < 1e-9
    assert set(cfg.benchmark.families) <= set(schemas.FAMILIES)
    assert cfg.benchmark.headline_holdout_family in cfg.benchmark.families


def test_every_json_artifact_has_is_mock():
    for name, model in io.JSON_MODELS.items():
        assert "is_mock" in model.model_fields, name


def test_every_table_has_is_mock():
    for name, spec in io.TABLE_SPECS.items():
        assert "is_mock" in spec, name


def test_predict_response_roundtrip_uses_class_alias():
    raw = {
        "sample_id": "c10c_motion_blur_00123_s4", "base_image_id": "c10_00123",
        "corruption": "motion_blur", "severity": 4, "image_url": "/images/x.png", "true_class": "dog",
        "target_model": {"prediction": "cat", "raw_confidence": 0.5, "entropy": 0.5, "margin": 0.1,
                         "top_k": [{"class": "cat", "prob": 0.5}]},
        "baselines": {"temp_scaled_clean": 0.5, "temp_scaled_corrupted": 0.5},
        "trust_layer": {"p_correct": 0.5, "decision": "reject",
                        "reasons": [{"signal": "tta_pconf", "text": "t", "shap": 1.0}]},
    }
    m = schemas.PredictResponse.model_validate(raw)
    assert m.model_dump(by_alias=True)["target_model"]["top_k"][0]["class"] == "cat"


def test_json_write_validates(tmp_path):
    import pytest
    with pytest.raises(Exception):
        io.write_json({"is_mock": True}, tmp_path / io.THRESHOLDS)
