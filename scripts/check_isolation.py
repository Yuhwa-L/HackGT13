"""Isolation rule 1 (plan.md Section 11): nothing outside shop/ imports `shop`, except the guarded
router mount in backend/main.py. Exit 1 on violation.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWED = {REPO_ROOT / "backend" / "main.py"}
SKIP_DIRS = {"shop", "frontend", ".venv", "node_modules", ".git", "data", ".claude"}


def imports_shop(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(), filename=str(path))
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name == "shop" or a.name.startswith("shop.") for a in node.names):
            lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module and (node.module == "shop" or node.module.startswith("shop.")):
            lines.append(node.lineno)
    return lines


def violations(root: Path = REPO_ROOT) -> list[str]:
    out = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root)
        if rel.parts[0] in SKIP_DIRS or path in ALLOWED:
            continue
        # tests/shop/ is part of the shop test suite and may import shop.
        if rel.parts[:2] == ("tests", "shop"):
            continue
        out += [f"{rel}:{ln} imports shop" for ln in imports_shop(path)]
    return out


def main() -> None:
    v = violations()
    if v:
        print("Isolation violations:\n" + "\n".join(f"  {x}" for x in v))
        sys.exit(1)
    print("[check-isolation] OK")


if __name__ == "__main__":
    main()
