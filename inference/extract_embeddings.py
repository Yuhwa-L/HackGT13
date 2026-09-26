"""Owner: A. 512-d penultimate embeddings via model.forward_head(model.forward_features(x), pre_logits=True).
Out: data/embeddings.npy (manifest samples, same row order as prediction_runs.parquet) and
data/train_embeddings.npy + data/train_labels.npy (CIFAR-10 train set: the kNN / Mahalanobis reference bank).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from inference.model import load_model, get_preprocess
from inference.run_inference import make_image_loader


def embed_batch(model, images, device):
    images = images.to(device)

    with torch.inference_mode():
        feature_maps = model.forward_features(images)
        embeddings = model.forward_head(
            feature_maps,
            pre_logits=True
        )

    return embeddings.cpu().numpy()


def extract_manifest_embeddings(
    model, preprocess, device, raw_dir, save_dir, batch_size
):
    manifest = pd.read_csv(save_dir / "manifest.csv")

    prediction_ids = pd.read_parquet(
        save_dir / "prediction_runs.parquet",
        columns=["sample_id"]
    )

    assert manifest["sample_id"].is_unique
    assert manifest["sample_id"].tolist() == prediction_ids["sample_id"].tolist(), (
        "Manifest order does not match prediction_runs.parquet."
    )

    load_image, _ = make_image_loader(manifest, raw_dir)

    n = len(manifest)
    embeddings = np.empty((n, 512), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        images = torch.stack([
            preprocess(load_image(row))
            for _, row in manifest.iloc[start:end].iterrows()
        ])

        embeddings[start:end] = embed_batch(model, images, device)

        if start == 0 or (start // batch_size + 1) % 50 == 0 or end == n:
            print(f"Manifest embeddings: {end:,}/{n:,}")

    assert embeddings.shape == (n, 512)
    assert np.isfinite(embeddings).all()

    np.save(save_dir / "embeddings.npy", embeddings)
    print("Saved embeddings.npy:", embeddings.shape)


def extract_reference_embeddings(
    model, preprocess, device, raw_dir, save_dir, batch_size
):
    reference_dataset = datasets.CIFAR10(
        root=str(raw_dir),
        train=True,
        download=False,
        transform=preprocess
    )

    reference_loader = DataLoader(
        reference_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    n = len(reference_dataset)
    embeddings = np.empty((n, 512), dtype=np.float32)
    labels_array = np.empty(n, dtype=np.int64)

    offset = 0

    for images, labels in reference_loader:
        end = offset + len(labels)

        embeddings[offset:end] = embed_batch(model, images, device)
        labels_array[offset:end] = labels.numpy()

        offset = end

        if offset % (batch_size * 50) == 0 or offset == n:
            print(f"Reference embeddings: {offset:,}/{n:,}")

    assert offset == n == 50000
    assert embeddings.shape == (50000, 512)
    assert np.isfinite(embeddings).all()
    assert np.array_equal(
        labels_array,
        np.asarray(reference_dataset.targets)
    )

    np.save(save_dir / "train_embeddings.npy", embeddings)
    np.save(save_dir / "train_labels.npy", labels_array)

    print("Saved train_embeddings.npy:", embeddings.shape)
    print("Saved train_labels.npy:", labels_array.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Generate only train_embeddings.npy and train_labels.npy."
    )
    args = parser.parse_args()

    raw_dir = Path("data/raw")
    save_dir = Path("data")
    save_dir.mkdir(parents=True, exist_ok=True)
    batch_size = 128

    device = torch.device(  # NVIDIA GPU first; Apple-silicon GPU (MPS) is ~24x faster than CPU on a Mac
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("Device:", device)

    model = load_model(
        raw_dir / "resnet18_cifar10.pth",
        device
    )
    preprocess = get_preprocess()

    if not args.reference_only:
        extract_manifest_embeddings(
            model, preprocess, device,
            raw_dir, save_dir, batch_size
        )

    extract_reference_embeddings(
        model, preprocess, device,
        raw_dir, save_dir, batch_size
    )


if __name__ == "__main__":
    main()