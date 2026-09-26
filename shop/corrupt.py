"""Apply the 8 MVP corruptions (config.yaml benchmark.cifar10c_types) x 5 severities at 224 px. Owner: A.

Uses `imagecorruptions`. If it can't be installed, implement the 8 corruptions directly here (TODO(A)).
Writes only via shop.paths.shop_path().
"""
from __future__ import annotations


def corrupt_image(img, corruption: str, severity: int):
    """PIL/uint8 image -> corrupted uint8 image, same size. TODO(A)."""
    raise NotImplementedError("TODO(A): corrupt_image")


def main() -> None:
    raise SystemExit("[shop.corrupt] TODO(A)")


if __name__ == "__main__":
    main()
