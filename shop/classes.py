"""Resolve shop_config.product_classes against torchvision ImageNet category names. Owner: A.

Source of names: torchvision.models.ResNet18_Weights.IMAGENET1K_V1.meta["categories"].
Must FAIL LOUDLY and print close matches (difflib) for any name not matched exactly.
Never guess indices — the team fixes the YAML.
"""
from __future__ import annotations


class UnresolvedClassError(ValueError):
    pass


def resolve_product_classes(names: list[str], categories: list[str]) -> list[int]:
    """Exact-match each name to an index in `categories`; raise UnresolvedClassError listing every
    unresolved name with its close matches. TODO(A).
    """
    raise NotImplementedError("TODO(A): resolve_product_classes")


def main() -> None:
    """CLI: python -m shop.classes -> prints name -> index table or the unresolved report. TODO(A)."""
    raise SystemExit("TODO(A): shop.classes CLI")


if __name__ == "__main__":
    main()
