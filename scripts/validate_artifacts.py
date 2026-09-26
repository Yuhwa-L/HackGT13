"""Validate every artifact that exists in data/ and data/shop/ against common/schemas.py.

Exit code 1 on any problem. Missing artifacts are reported but not errors.
"""
from __future__ import annotations

import sys
from pathlib import Path

from common import io, schemas
from common.config import load_config

JSON_ARTIFACTS = [io.THRESHOLDS, io.EVALUATION, io.DEMO_CACHE]
SHOP_JSON_ARTIFACTS = [io.THRESHOLDS, io.EVALUATION, io.SHOP_CACHE, io.CATALOG]
TABLE_ARTIFACTS = [io.MANIFEST, io.PREDICTION_RUNS, io.FEATURES]
ARRAY_ARTIFACTS = [io.EMBEDDINGS, io.EMBEDDING_IDS, io.TTA_LOGITS, io.REFERENCE_BANK, io.REFERENCE_LABELS]


def validate_dir(root: Path, json_names: list[str]) -> list[str]:
    problems: list[str] = []
    cfg = load_config()
    for name in json_names:
        p = root / name
        if not p.exists():
            print(f"  - {p} (missing)")
            continue
        try:
            io.read_json(p)
            print(f"  ok {p}")
        except Exception as e:  # noqa: BLE001
            problems.append(f"{p}: {e}")
    for name in TABLE_ARTIFACTS:
        p = root / name
        if not p.exists():
            print(f"  - {p} (missing)")
            continue
        df = io.read_table(p)
        errs = schemas.check_columns(df, io.TABLE_SPECS[name])
        if name == io.MANIFEST:
            try:
                errs += schemas.check_manifest_invariants(df, cfg.benchmark.cifar10c_types)
            except NotImplementedError:
                print(f"  ! {p}: manifest invariants not implemented yet (TODO(B))")
        problems += [f"{p}: {e}" for e in errs]
        if not errs:
            print(f"  ok {p}")
    for name in ARRAY_ARTIFACTS:
        p = root / name
        if p.exists():
            # TODO(B): check dtype/shape against schemas.ARRAY_SPECS and row alignment with embeddings_ids.
            print(f"  ok {p} (shape {io.read_array(p, mmap=True).shape}; spec check TODO)")
    return problems


def main() -> None:
    data = io.data_dir()
    print(f"[validate] {data}")
    problems = validate_dir(data, JSON_ARTIFACTS)
    if (data / "shop").is_dir():
        print(f"[validate] {data / 'shop'}")
        problems += validate_dir(data / "shop", SHOP_JSON_ARTIFACTS)
    if problems:
        print("\nFAILED:\n" + "\n".join(f"  {p}" for p in problems))
        sys.exit(1)
    print("\n[validate] all present artifacts OK")


if __name__ == "__main__":
    main()
