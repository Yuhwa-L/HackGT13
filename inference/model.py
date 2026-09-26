"""CIFAR ResNet-18 loader (3x3 stem, no maxpool). Owner: A.

Use exactly this recipe (plan.md Section 6):

    import timm, torch, torch.nn as nn
    model = timm.create_model("resnet18", num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)   # CIFAR stem
    model.maxpool = nn.Identity()
    model.load_state_dict(torch.hub.load_state_dict_from_url(
        CHECKPOINT_URL, map_location="cpu", file_name="resnet18_cifar10.pth"))
    model.eval()
    emb = model.forward_head(model.forward_features(x), pre_logits=True)  # (B, 512)
    logits = model.fc(emb)

CHECKPOINT_URL comes from config.model.checkpoint_url. Never download in tests.
"""
from __future__ import annotations


def load_model(device: str, checkpoint_url: str | None = None):
    """Return the eval-mode model on `device`. TODO(A): implement per the recipe above."""
    raise NotImplementedError("TODO(A): load_model")


def normalize(x_uint8, mean: list[float], std: list[float]):
    """(B,32,32,3) uint8 numpy -> (B,3,32,32) float tensor normalized with config.model.mean/std. TODO(A)."""
    raise NotImplementedError("TODO(A): normalize")


def embed_and_logits(model, x):
    """x: normalized (B,3,H,W) tensor -> (emb (B,512), logits (B,num_classes)). TODO(A)."""
    raise NotImplementedError("TODO(A): embed_and_logits")
