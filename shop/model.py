"""Frozen torchvision ResNet-18 (ImageNet weights) with a subset-restricted softmax over the product classes.

Subset-restricted softmax: we keep the 1000-way logits only for the PRODUCT_CLASSES indices and renormalize over those,
i.e. the model answers "which of these products is it?". Everything downstream (signals, trust layer) sees
len(PRODUCT_CLASSES)-way logits. Embeddings are the 512-d penultimate features.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import resnet18

from shop.classes import product_indices
from shop.config import RAW, TTA_SHIFT
from shop.images import load_photo, prepare  # noqa: F401  (re-exported)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
WEIGHTS = RAW / "resnet18-f37072fd.pth"  # torchvision IMAGENET1K_V1, fetched with curl (see shop/README.md)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def load_model(dev):
    model = resnet18()
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    return model.eval().requires_grad_(False).to(dev)


def _to_tensor(imgs, dev):
    x = torch.from_numpy(np.ascontiguousarray(imgs)).permute(0, 3, 1, 2).float().div(255)
    return ((x - MEAN) / STD).to(dev)


@torch.inference_mode()
def forward(model, imgs, dev):
    """imgs: (B, 224, 224, 3) uint8 -> subset logits (B, C), TTA subset logits (B, 3, C), embeddings (B, 512).
    TTA views: hflip, +/-TTA_SHIFT px horizontal shift with reflect padding (the core's views, scaled to 224 px)."""
    idx = torch.tensor(product_indices(), device=dev)
    x = _to_tensor(imgs, dev)
    feats = torch.flatten(model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(
        model.maxpool(model.relu(model.bn1(model.conv1(x))))))))), 1)
    logits = model.fc(feats)[:, idx]
    w = x.shape[-1]
    padded = F.pad(x, (TTA_SHIFT, TTA_SHIFT, 0, 0), mode="reflect")
    views = [x.flip(-1), padded[..., :w], padded[..., 2 * TTA_SHIFT:2 * TTA_SHIFT + w]]
    tta = torch.stack([model(v)[:, idx] for v in views], dim=1)
    return logits.float().cpu().numpy(), tta.float().cpu().numpy(), feats.float().cpu().numpy()
