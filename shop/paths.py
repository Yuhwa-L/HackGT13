"""Every shop write goes through shop_path(), which asserts the target resolves under data/shop/.

tests/shop/test_isolation.py scans shop/ for writes that bypass this helper.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SHOP_DATA = (REPO_ROOT / "data" / "shop").resolve()


def shop_path(*parts: str | Path) -> Path:
    p = (SHOP_DATA.joinpath(*parts)).resolve()
    if p != SHOP_DATA and SHOP_DATA not in p.parents:
        raise ValueError(f"shop write outside data/shop/: {p}")
    return p
