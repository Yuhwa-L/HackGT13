"""torchvision ResNet-18 (IMAGENET1K_V1) restricted to product classes. Owner: A.

SUBSET-RESTRICTED SOFTMAX: take the 1000-way logits, keep only the product-class columns (indices
from shop/classes.py, in shop_config order), and softmax over that subset. So probabilities sum to 1
across the 30 product classes only. Embeddings = 512-d penultimate layer (after avgpool, before fc).
Standard ImageNet preprocessing (weights.transforms()).
"""
from __future__ import annotations


def load_shop_model(device: str):
    """TODO(A)."""
    raise NotImplementedError("TODO(A): load_shop_model")


def embed_and_subset_logits(model, x, class_indices: list[int]):
    """x (B,3,224,224) -> (emb (B,512), subset_logits (B,len(class_indices))). TODO(A)."""
    raise NotImplementedError("TODO(A): embed_and_subset_logits")
