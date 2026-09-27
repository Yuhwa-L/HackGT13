"""Frozen torchvision ViT-B/16 (ImageNet weights) with a subset-restricted softmax over the product classes.

A vision transformer here, deliberately different from the core project's CIFAR ResNet-18: the same trust layer
wraps a different architecture on a different task.

Subset-restricted softmax: we keep the 1000-way logits only for the PRODUCT_CLASSES indices and renormalize over those,
i.e. the model answers "which of these products is it?". Everything downstream (signals, trust layer) sees
len(PRODUCT_CLASSES)-way logits. Embeddings are the 768-d class token after the encoder's final LayerNorm, i.e. the
input to the classification head.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import vit_b_16

from shop.classes import product_indices
from shop.config import RAW, TTA_SHIFT
from shop.images import load_photo, prepare  # noqa: F401  (re-exported)

ARCH = "vit_b_16"
EMB_DIM = 768
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
WEIGHTS = RAW / "vit_b_16-c867db91.pth"  # torchvision ViT_B_16_Weights.IMAGENET1K_V1, fetched with curl (shop/README.md)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def load_model(dev):
    model = vit_b_16()
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    return model.eval().requires_grad_(False).to(dev)


def _to_tensor(imgs, dev):
    x = torch.from_numpy(np.ascontiguousarray(imgs)).permute(0, 3, 1, 2).float().div(255)
    return ((x - MEAN) / STD).to(dev)


def _embed(model, x):
    """Class token after the encoder (which ends with its LayerNorm): the 768-d input to model.heads."""
    tokens = model._process_input(x)
    cls = model.class_token.expand(tokens.shape[0], -1, -1)
    return model.encoder(torch.cat([cls, tokens], dim=1))[:, 0]


@torch.inference_mode()
def forward(model, imgs, dev):
    """imgs: (B, 224, 224, 3) uint8 -> subset logits (B, C), TTA subset logits (B, 3, C), embeddings (B, 768).
    TTA views: hflip, +/-TTA_SHIFT px horizontal shift with reflect padding (the core's views, scaled to 224 px)."""
    idx = torch.tensor(product_indices(), device=dev)
    x = _to_tensor(imgs, dev)
    feats = _embed(model, x)
    logits = model.heads(feats)[:, idx]
    w = x.shape[-1]
    padded = F.pad(x, (TTA_SHIFT, TTA_SHIFT, 0, 0), mode="reflect")
    views = [x.flip(-1), padded[..., :w], padded[..., 2 * TTA_SHIFT:2 * TTA_SHIFT + w]]
    tta = torch.stack([model(v)[:, idx] for v in views], dim=1)
    return logits.float().cpu().numpy(), tta.float().cpu().numpy(), feats.float().cpu().numpy()
