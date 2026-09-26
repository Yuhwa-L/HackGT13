"""Shop end-to-end pipeline. Owner: A.

Reuses core functions (inference/tta.py, trust/signals.py, trust/temperature_scaling.py,
trust/calibrate.py, trust/thresholds.py, trust/metrics.py, trust/reasons.py), passing shop paths
and num_classes explicitly. Core functions may be refactored ONLY to accept num_classes / input size
as parameters; core tests must keep passing and core defaults must not change.

Outputs in data/shop/ (via shop_path): manifest, predictions, features, thresholds, evaluation.json.
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.parse_args()
    raise SystemExit("[shop.run_pipeline] TODO(A)")


if __name__ == "__main__":
    main()
