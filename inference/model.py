"""Owner: A. ResNet-18 for CIFAR-10 (timm resnet18, 3x3 stem, no maxpool), weights from data/raw/resnet18_cifar10.pth.
Normalize inputs with mean (0.4914, 0.4822, 0.4465) and std (0.2023, 0.1994, 0.2010). Expected clean test accuracy: 94.98%.
"""
from pathlib import Path

import torch
import torch.nn as nn
import timm
from torchvision import transforms


def load_model(checkpoint_path, device):
    model = timm.create_model("resnet18", num_classes=10)

    # Match the pretrained checkpoint's CIFAR architecture.
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()

    weights = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=True
    )

    # KEEP: this actually loads the trained weights.
    model.load_state_dict(weights, strict=True)

    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    return model


def get_preprocess():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])