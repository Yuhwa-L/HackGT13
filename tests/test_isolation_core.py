"""Isolation rule 1 + rule 3 (core must work without shop/)."""
import subprocess
import sys
from pathlib import Path

from scripts.check_isolation import violations

ROOT = Path(__file__).resolve().parent.parent


def test_no_core_file_imports_shop():
    assert violations() == []


def test_core_imports_with_shop_unimportable():
    code = (
        "import sys; sys.modules['shop'] = None\n"
        "import backend.main, common.io, trust.features, benchmark.make_splits\n"
        "assert not backend.main.STATE['shop_enabled']\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
