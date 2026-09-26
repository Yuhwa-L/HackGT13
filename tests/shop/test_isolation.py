"""Isolation rule 2: every write in shop/ goes through shop_path()."""
import re
from pathlib import Path

import pytest

from shop.paths import shop_path

SHOP = Path(__file__).resolve().parents[2] / "shop"
# Raw write calls that must not appear in shop/ (use shop_path(...) and common.io helpers instead).
# TODO(A): tighten if you add new write patterns.
FORBIDDEN = re.compile(r"open\([^)]*['\"]w|\.to_csv\(|\.to_parquet\(|np\.save\(|\.save\(['\"]")


def test_shop_path_rejects_escape():
    assert shop_path("x.json").name == "x.json"
    with pytest.raises(ValueError):
        shop_path("..", "evil.json")


def test_no_raw_writes_in_shop():
    offenders = []
    for p in SHOP.rglob("*.py"):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if FORBIDDEN.search(line) and "shop_path" not in line:
                offenders.append(f"{p.name}:{i}: {line.strip()}")
    assert offenders == []
