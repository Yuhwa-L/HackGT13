"""ALL data contracts (plan.md Section 5) in one place.

- JSON artifacts are pydantic models.
- Tables (CSV / parquet) are column specs: {column: dtype-kind}.
- Arrays are shape specs.

The shop pipeline uses the SAME schemas with dataset in {"imagenet_products", "imagenet_products_c"}.
Change a contract here first, then mirror it in frontend/src/types.ts, then tell the team.
"""
from __future__ import annotations

from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

Dataset = Literal["cifar10_test", "cifar10c", "cifar10_1", "imagenet_products", "imagenet_products_c"]
Family = Literal["clean", "noise", "blur", "weather", "digital"]
Split = Literal["train", "val", "cal", "test"]
Decision = Literal["trust", "caution", "reject"]
Method = Literal[
    "raw_confidence",
    "temp_scaled_clean",
    "temp_scaled_corrupted",
    "tta_only",
    "logreg",
    "trust_layer",
]

DATASETS: tuple[str, ...] = Dataset.__args__  # type: ignore[attr-defined]
FAMILIES: tuple[str, ...] = Family.__args__  # type: ignore[attr-defined]
SPLITS: tuple[str, ...] = Split.__args__  # type: ignore[attr-defined]
DECISIONS: tuple[str, ...] = Decision.__args__  # type: ignore[attr-defined]
METHODS: tuple[str, ...] = Method.__args__  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Table column specs. dtype kinds: "str", "int", "float", "bool", "list[float]", "enum:<Name>"
# ---------------------------------------------------------------------------

TableSpec = dict[str, str]

# 5.1 manifest.csv — one row per evaluated image
MANIFEST_COLUMNS: TableSpec = {
    "sample_id": "str",          # c10c_gaussian_noise_00123_s3 | c10_00123 | c101_00042
    "base_image_id": "str",      # c10_00123 | c101_00042 — THE split unit
    "dataset": "enum:Dataset",
    "image_index": "int",        # row in source array
    "true_label": "int",
    "true_class": "str",
    "corruption": "str",         # "clean" or corruption type name
    "family": "enum:Family",
    "severity": "int",           # 0 clean, 1-5
    "split": "enum:Split",
    "is_mock": "bool",
}

# 5.2 prediction_runs.parquet — one row per sample_id
PREDICTION_RUNS_COLUMNS: TableSpec = {
    "sample_id": "str",
    "base_image_id": "str",
    "split": "enum:Split",
    "true_label": "int",
    "pred_label": "int",
    "correct": "int",            # 0/1
    "raw_confidence": "float",
    "top2_prob": "float",
    "entropy": "float",
    "margin": "float",
    "logits": "list[float]",     # len = num_classes (10 for core)
    "corruption": "str",
    "family": "enum:Family",
    "severity": "int",
    "is_mock": "bool",
}

# 5.4 features.csv
FEATURES_COLUMNS: TableSpec = {
    "sample_id": "str",
    "base_image_id": "str",
    "split": "enum:Split",
    "family": "enum:Family",
    "corruption": "str",
    "severity": "int",
    "raw_confidence": "float",
    "entropy": "float",
    "margin": "float",
    "tta_agree": "float",
    "tta_pconf": "float",
    "tta_std": "float",
    "knn_dist": "float",
    "maha_pred": "float",
    "trust_score": "float?",     # nullable
    "failure": "int",            # 0/1
    "is_mock": "bool",
}

# Columns that must NEVER be model inputs (see trust/features.py::MODEL_FEATURES)
ANALYSIS_ONLY_COLUMNS: frozenset[str] = frozenset(
    {"corruption", "family", "severity", "split", "base_image_id", "sample_id", "is_mock", "failure"}
)

# 5.3 arrays: name -> (dtype, shape with symbolic dims)
ARRAY_SPECS: dict[str, tuple[str, tuple]] = {
    "embeddings.npy": ("float32", ("N", 512)),
    "embeddings_ids.npy": ("str", ("N",)),               # sample_id order
    "tta_logits.npy": ("float32", ("N", 3, "C")),       # view order = config.tta.views
    "reference_bank.npy": ("float32", ("M", 512)),       # CIFAR-10 TRAIN embeddings (M = 50000 for core)
    "reference_labels.npy": ("int64", ("M",)),
}

# ---------------------------------------------------------------------------
# JSON models
# ---------------------------------------------------------------------------


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# 5.5 thresholds.json
class Thresholds(_Model):
    is_mock: bool
    fold: str
    tau_reject: Optional[float]
    tau_trust: Optional[float]
    reject_target_error: float
    trust_target_error: float
    cal_error_at_reject: Optional[float]
    cal_error_at_trust: Optional[float]


# 5.6 evaluation.json
class MethodMetrics(_Model):
    aurc: float
    auroc_pooled: float
    auroc_within_group: float
    ece: float
    brier: float
    nll: float
    error_at_coverage: dict[str, float]          # keys are str(coverage), e.g. "0.5"
    ci95: dict[str, tuple[float, float]]         # metric -> [lo, hi]


class SeverityRow(_Model):
    severity: int
    accuracy: float
    mean_raw_conf: float
    mean_temp_clean: float
    mean_temp_corrupted: float
    mean_p_correct: float


class ReliabilityBin(_Model):
    bin_lo: float
    bin_hi: float
    conf: float
    acc: float
    count: int


class RiskCoveragePoint(_Model):
    coverage: float
    risk: float


class BrokenPromise(_Model):
    target_error: float
    rule_source: str
    actual_error: float


class FoldResult(_Model):
    methods: dict[str, MethodMetrics]            # keys in METHODS
    by_severity: list[SeverityRow]
    reliability: dict[str, list[ReliabilityBin]]
    risk_coverage: dict[str, list[RiskCoveragePoint]]
    broken_promise: BrokenPromise


class NaturalShiftResult(_Model):
    methods: dict[str, MethodMetrics]


class Evaluation(_Model):
    is_mock: bool
    headline_fold: str
    folds: dict[str, FoldResult]                 # keyed by held-out family
    natural_shift: dict[str, NaturalShiftResult] # e.g. {"cifar10_1": ...}


# 5.7 PredictResponse / demo_cache.json
class TopK(_Model):
    class_: str = Field(alias="class")
    prob: float


class TargetModel(_Model):
    prediction: str
    raw_confidence: float
    entropy: float
    margin: float
    top_k: list[TopK]


class Baselines(_Model):
    temp_scaled_clean: float
    temp_scaled_corrupted: float


class Reason(_Model):
    signal: str
    text: str
    shap: float


class TrustLayer(_Model):
    p_correct: float
    decision: Decision
    reasons: list[Reason]


class PredictResponse(_Model):
    sample_id: str
    base_image_id: str
    corruption: str
    severity: int
    image_url: str
    true_class: str
    target_model: TargetModel
    baselines: Baselines
    trust_layer: TrustLayer


class DemoCache(_Model):
    is_mock: bool
    samples: list[PredictResponse]               # keyed by (base_image_id, corruption, severity)


# ---------------------------------------------------------------------------
# Shop (Section 10) — same PredictResponse plus candidates
# ---------------------------------------------------------------------------


class Candidate(_Model):
    class_: str = Field(alias="class")
    prob: float


class ShopIdentifyResponse(PredictResponse):
    photo_id: str
    candidates: list[Candidate]


class ShopCache(_Model):
    is_mock: bool
    samples: list[ShopIdentifyResponse]


class Product(_Model):
    product_id: str
    name: str
    brand: str
    price: float
    description: str
    class_: str = Field(alias="class")


class Catalog(_Model):
    fictional: Literal[True]
    is_mock: bool
    products: list[Product]


class AssistRequest(_Model):
    photo_id: str
    user_message: str
    confirmed_class: Optional[str] = None


class ProductSummary(_Model):
    product_id: str
    name: str
    price: float
    class_: str = Field(alias="class")


class AssistResponse(_Model):
    decision: Decision
    allowed_actions: list[str]
    assistant_message: str
    candidates: list[Candidate]
    products: list[ProductSummary]
    cart_allowed: bool
    llm_used: bool


class CheckoutRequest(_Model):
    photo_id: str
    product_ids: list[str]
    confirmed_class: Optional[str] = None


class CheckoutResponse(_Model):
    order_id: str
    items: list[ProductSummary]
    total: float
    is_mock: Literal[True] = True                # checkout is ALWAYS a mock; no payment data


# ---------------------------------------------------------------------------
# API: /api/health, /api/demo/samples
# ---------------------------------------------------------------------------


class HealthResponse(_Model):
    status: Literal["ok"]
    artifacts: dict[str, bool]                   # artifact name -> loaded?
    any_mock: bool
    shop_enabled: bool


class DemoSampleSummary(_Model):
    base_image_id: str
    true_class: str
    corruptions: list[str]
    severities: list[int]


# ---------------------------------------------------------------------------
# Table validation helpers
# ---------------------------------------------------------------------------


def check_columns(df: pd.DataFrame, spec: TableSpec) -> list[str]:
    """Return a list of human-readable problems (empty = OK). Checks column presence only.

    TODO(B): add dtype-kind checks (int/float/bool/enum membership) per `spec`.
    """
    missing = [c for c in spec if c not in df.columns]
    extra = [c for c in df.columns if c not in spec]
    problems = [f"missing column: {c}" for c in missing]
    problems += [f"unexpected column: {c}" for c in extra]
    return problems


def check_manifest_invariants(df: pd.DataFrame, cifar10c_types: dict[str, list[str]]) -> list[str]:
    """Cross-row invariants for manifest.csv (Section 5.1). Return list of problems (empty = OK).

    Must enforce:
      - sample_id unique
      - every base_image_id maps to exactly one split
      - cifar10_1 rows are always split == "test"
      - severity == 0  iff  corruption == "clean"
      - family consistent with cifar10c_types (clean -> "clean")

    TODO(B): implement.
    """
    raise NotImplementedError("TODO(B): manifest invariants")
