"""Spec tests call stubbed functions. While a stub raises NotImplementedError the test is reported as
XFAIL ("TODO"); once implemented it runs for real and must pass. `pytest -rx` lists what's left.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if call.excinfo is not None and call.excinfo.errisinstance(NotImplementedError):
        rep.outcome = "skipped"
        rep.wasxfail = f"not implemented: {call.excinfo.value}"


def pytest_collection_modifyitems(config, items):
    shop_present = (Path(__file__).resolve().parent.parent / "shop").is_dir()
    for item in items:
        if "tests/shop" in str(item.fspath).replace("\\", "/") and not shop_present:
            item.add_marker(pytest.mark.skip(reason="shop/ absent"))
