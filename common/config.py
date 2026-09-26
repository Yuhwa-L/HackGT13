"""Typed loader for config.yaml. Every script reads settings from here; no magic constants in scripts."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


class PathsConfig(BaseModel):
    data: Path
    raw: Path
    demo_images: Path


class SplitsConfig(BaseModel):
    unit: str
    fractions: dict[str, float]
    stratify_by: str


class BenchmarkConfig(BaseModel):
    cifar10c_types: dict[str, list[str]]   # family -> corruption types
    severities: list[int]
    n_base_images: int
    headline_holdout_family: str

    @property
    def families(self) -> list[str]:
        return list(self.cifar10c_types)

    @property
    def corruption_to_family(self) -> dict[str, str]:
        return {c: fam for fam, cs in self.cifar10c_types.items() for c in cs}


class ModelConfig(BaseModel):
    checkpoint_url: str
    mean: list[float]
    std: list[float]
    expected_clean_acc_min: float


class TTAConfig(BaseModel):
    views: list[str]


class SignalsConfig(BaseModel):
    knn_k: int
    knn_metric: str
    maha_ridge: float


class ThresholdTargets(BaseModel):
    reject_target_error: float
    trust_target_error: float


class TrustConfig(BaseModel):
    clean_row_weight: float
    thresholds: ThresholdTargets


class EvalConfig(BaseModel):
    ece_bins: int
    bootstrap_resamples: int
    coverage_points: list[float]


class DemoConfig(BaseModel):
    n_curated_base_images: int


class Config(BaseModel):
    seed: int
    paths: PathsConfig
    splits: SplitsConfig
    benchmark: BenchmarkConfig
    model: ModelConfig
    tta: TTAConfig
    signals: SignalsConfig
    trust: TrustConfig
    eval: EvalConfig
    demo: DemoConfig

    def resolve(self, p: Path | str) -> Path:
        """Resolve a config-relative path against the repo root."""
        p = Path(p)
        return p if p.is_absolute() else REPO_ROOT / p


@lru_cache(maxsize=None)
def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    with open(path) as f:
        return Config.model_validate(yaml.safe_load(f))
