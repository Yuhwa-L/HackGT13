"""Read/write helpers for every artifact. All pipeline scripts write through here.

JSON writes validate against the pydantic model in common/schemas.py.
Table writes check columns against the column spec.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd
from pydantic import BaseModel

from common import schemas
from common.config import load_config

M = TypeVar("M", bound=BaseModel)

# Canonical artifact filenames (relative to a data root: data/ for core, data/shop/ for shop)
MANIFEST = "manifest.csv"
PREDICTION_RUNS = "prediction_runs.parquet"
FEATURES = "features.csv"
EMBEDDINGS = "embeddings.npy"
EMBEDDING_IDS = "embeddings_ids.npy"
TTA_LOGITS = "tta_logits.npy"
REFERENCE_BANK = "reference_bank.npy"
REFERENCE_LABELS = "reference_labels.npy"
THRESHOLDS = "thresholds.json"
EVALUATION = "evaluation.json"
DEMO_CACHE = "demo_cache.json"
SHOP_CACHE = "shop_cache.json"
CATALOG = "catalog.json"

JSON_MODELS: dict[str, type[BaseModel]] = {
    THRESHOLDS: schemas.Thresholds,
    EVALUATION: schemas.Evaluation,
    DEMO_CACHE: schemas.DemoCache,
    SHOP_CACHE: schemas.ShopCache,
    CATALOG: schemas.Catalog,
}
TABLE_SPECS: dict[str, schemas.TableSpec] = {
    MANIFEST: schemas.MANIFEST_COLUMNS,
    PREDICTION_RUNS: schemas.PREDICTION_RUNS_COLUMNS,
    FEATURES: schemas.FEATURES_COLUMNS,
}


def data_dir() -> Path:
    return load_config().resolve(load_config().paths.data)


def models_dir() -> Path:
    return data_dir() / "models"


class ArtifactValidationError(ValueError):
    pass


# ---- JSON -------------------------------------------------------------------


def write_json(obj: BaseModel | dict, path: Path, model: type[BaseModel] | None = None) -> Path:
    model = model or JSON_MODELS.get(Path(path).name)
    if model is not None:
        obj = model.model_validate(obj if isinstance(obj, dict) else obj.model_dump(by_alias=True))
    payload = obj.model_dump(by_alias=True, mode="json") if isinstance(obj, BaseModel) else obj
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def read_json(path: Path, model: type[M] | None = None) -> M | dict:
    raw = json.loads(Path(path).read_text())
    model = model or JSON_MODELS.get(Path(path).name)  # type: ignore[assignment]
    return model.model_validate(raw) if model is not None else raw


# ---- Tables -----------------------------------------------------------------


def _check_table(df: pd.DataFrame, path: Path) -> None:
    spec = TABLE_SPECS.get(Path(path).name)
    if spec is None:
        return
    problems = schemas.check_columns(df, spec)
    if problems:
        raise ArtifactValidationError(f"{path}: " + "; ".join(problems))


def write_table(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    _check_table(df, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


# ---- Arrays -----------------------------------------------------------------


def write_array(arr: np.ndarray, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return path


def read_array(path: Path, mmap: bool = False) -> np.ndarray:
    return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)


def is_mock_artifact(path: Path) -> bool:
    """True if a JSON artifact has is_mock=true or a table has any is_mock row."""
    path = Path(path)
    if path.suffix == ".json":
        return bool(json.loads(path.read_text()).get("is_mock", False))
    if path.suffix in (".csv", ".parquet"):
        df = read_table(path)
        return bool("is_mock" in df.columns and df["is_mock"].any())
    return False

