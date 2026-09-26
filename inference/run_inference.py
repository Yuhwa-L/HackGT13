"""Owner: A. Run every data/manifest.csv sample through the frozen ResNet: original + 3 TTA views (hflip, +2 px / -2 px shift, reflect pad).
Out: data/prediction_runs.parquet (one row per sample_id, including logit_0..logit_9) and data/tta_logits.npy (N x 3 x 10, same row order).
"""
# from pathlib import Path
# import torch
# import torch.nn as nn
# import timm
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# from PIL import Image
# import time

# checkpoint_path = Path("data/raw/resnet18_cifar10.pth")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = timm.create_model("resnet18", num_classes=10)

# model.conv1 = nn.Conv2d(
#     3, 64, kernel_size=3, stride=1, padding=1, bias=False
# )
# model.maxpool = nn.Identity()

# weights = torch.load(
#     checkpoint_path,
#     map_location="cpu",
#     weights_only=True
# )

# print(model.load_state_dict(weights, strict=True))

# model = model.to(device)
# model.eval()
# model.requires_grad_(False)


# preprocess = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=(0.4914, 0.4822, 0.4465),
#         std=(0.2023, 0.1994, 0.2010)
#     )
# ])

# test_dataset = datasets.CIFAR10(
#     root="data/raw",
#     train=False,
#     download=False
# )

# print("Test images:", len(test_dataset))
# print("Classes:", test_dataset.classes)

# def extract_batch(images):
#     images = images.to(device)

#     with torch.inference_mode():
#         # Original-image embeddings and logits.
#         feature_maps = model.forward_features(images)
#         embeddings = model.forward_head(
#             feature_maps,
#             pre_logits=True
#         )
#         logits = model.fc(embeddings)

#         # Three augmented views: flip, right shift, left shift.
#         width = images.shape[-1]
#         padded = F.pad(images, (2, 2, 0, 0), mode="reflect")

#         views = [
#             images.flip(-1),
#             padded[..., :width],
#             padded[..., 4:4 + width]
#         ]

#         tta_logits = torch.stack(
#             [model(view) for view in views],
#             dim=1
#         )

#     # Return CPU tensors so collected outputs don't fill GPU memory.
#     return {
#         "logits": logits.cpu(),
#         "embeddings": embeddings.cpu(),
#         "tta_logits": tta_logits.cpu()
#     }

# # ==============================================================================
# raw_dir = Path("data/raw")
# manifest = pd.read_csv("data/manifest.csv")

# # Original uint8 images, unaffected by test_dataset.transform.
# clean_images = test_dataset.data
# clean_labels = np.asarray(test_dataset.targets)

# natural_images = np.load(
#     raw_dir / "cifar10.1_v6_data.npy",
#     mmap_mode="r"
# )
# natural_labels = np.load(
#     raw_dir / "cifar10.1_v6_labels.npy",
#     mmap_mode="r"
# )

# corruption_names = manifest.loc[
#     manifest["dataset"] == "cifar10c", "corruption"
# ].unique()

# corrupted_images = {
#     name: np.load(
#         raw_dir / "CIFAR-10-C" / f"{name}.npy",
#         mmap_mode="r"
#     )
#     for name in corruption_names
# }

# def load_manifest_image(row):
#     index = int(row["image_index"])
#     dataset = row["dataset"]

#     if dataset == "cifar10_test":
#         image = clean_images[index]
#         expected_label = clean_labels[index]

#     elif dataset == "cifar10c":
#         image = corrupted_images[row["corruption"]][index]
#         expected_label = clean_labels[index % 10000]

#         assert index // 10000 + 1 == int(row["severity"])

#     elif dataset == "cifar10_1":
#         image = natural_images[index]
#         expected_label = natural_labels[index]

#     else:
#         raise ValueError(f"Unknown dataset: {dataset}")

#     assert int(expected_label) == int(row["true_label"])

#     return Image.fromarray(np.array(image, copy=True))

# batch_size = 128
# n = len(manifest)

# all_logits = np.empty((n, 10), dtype=np.float32)
# all_embeddings = np.empty((n, 512), dtype=np.float32)
# all_tta_logits = np.empty((n, 3, 10), dtype=np.float32)

# model.eval()
# start_time = time.time()

# for start in range(0, n, batch_size):
#     end = min(start + batch_size, n)
#     batch_rows = manifest.iloc[start:end]

#     images = torch.stack([
#         preprocess(load_manifest_image(row))
#         for _, row in batch_rows.iterrows()
#     ])

#     batch_outputs = extract_batch(images)

#     all_logits[start:end] = batch_outputs["logits"].numpy()
#     all_embeddings[start:end] = batch_outputs["embeddings"].numpy()
#     all_tta_logits[start:end] = batch_outputs["tta_logits"].numpy()

#     if start == 0 or (start // batch_size + 1) % 50 == 0 or end == n:
#         elapsed = time.time() - start_time
#         print(f"{end:,}/{n:,} images processed — {elapsed:.1f}s")

# print("Inference complete.")

# prediction_rows = manifest.copy().reset_index(drop=True)

# predictions = all_logits.argmax(axis=1)

# prediction_rows["pred_label"] = predictions
# prediction_rows["pred_class"] = [
#     test_dataset.classes[i] for i in predictions
# ]
# prediction_rows["correct"] = (
#     predictions == prediction_rows["true_label"].to_numpy()
# ).astype("int8")

# for i in range(10):
#     prediction_rows[f"logit_{i}"] = all_logits[:, i]

# probabilities = torch.softmax(
#     torch.from_numpy(all_logits).double(),
#     dim=1
# ).numpy()

# prediction_rows["raw_confidence"] = probabilities.max(axis=1)

# assert prediction_rows["sample_id"].is_unique
# assert prediction_rows["sample_id"].tolist() == manifest["sample_id"].tolist()
# assert all_embeddings.shape == (n, 512)
# assert all_tta_logits.shape == (n, 3, 10)

# for array in [all_logits, all_embeddings, all_tta_logits]:
#     assert np.isfinite(array).all()

# save_dir = Path("data")
# save_dir.mkdir(parents=True, exist_ok=True)

# prediction_rows.to_parquet(
#     save_dir / "prediction_runs.parquet",
#     index=False
# )
# np.save(save_dir / "embeddings.npy", all_embeddings)
# np.save(save_dir / "tta_logits.npy", all_tta_logits)

# print(f"Saved {n:,} predictions.")
# print("Embeddings:", all_embeddings.shape)
# print("TTA logits:", all_tta_logits.shape)



# =
# =
# =
# =
# =
# =
# =
# =
# =

from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import datasets

from inference.model import load_model, get_preprocess


def make_image_loader(manifest, raw_dir):
    """Open source datasets and return an image-loading function."""
    raw_dir = Path(raw_dir)

    clean_dataset = datasets.CIFAR10(
        root=str(raw_dir),
        train=False,
        download=False
    )

    clean_images = clean_dataset.data
    clean_labels = np.asarray(clean_dataset.targets)

    natural_images = np.load(
        raw_dir / "cifar10.1_v6_data.npy",
        mmap_mode="r"
    )
    natural_labels = np.load(
        raw_dir / "cifar10.1_v6_labels.npy",
        mmap_mode="r"
    )

    corruption_names = manifest.loc[
        manifest["dataset"] == "cifar10c", "corruption"
    ].unique()

    corrupted_images = {
        name: np.load(
            raw_dir / "CIFAR-10-C" / f"{name}.npy",
            mmap_mode="r"
        )
        for name in corruption_names
    }

    def load_image(row):
        index = int(row["image_index"])
        dataset = row["dataset"]

        if dataset == "cifar10_test":
            image = clean_images[index]
            expected_label = clean_labels[index]

        elif dataset == "cifar10c":
            # The manifest already includes the severity offset.
            image = corrupted_images[row["corruption"]][index]
            expected_label = clean_labels[index % 10000]

            assert index // 10000 + 1 == int(row["severity"])

        elif dataset == "cifar10_1":
            image = natural_images[index]
            expected_label = natural_labels[index]

        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        assert int(expected_label) == int(row["true_label"])

        return Image.fromarray(np.array(image, copy=True))

    return load_image, clean_dataset.classes


def predict_batch(model, images, device):
    images = images.to(device)

    with torch.inference_mode():
        logits = model(images)

        width = images.shape[-1]
        padded = F.pad(images, (2, 2, 0, 0), mode="reflect")

        views = [
            images.flip(-1),
            padded[..., :width],
            padded[..., 4:4 + width]
        ]

        tta_logits = torch.stack(
            [model(view) for view in views],
            dim=1
        )

    return logits.cpu().numpy(), tta_logits.cpu().numpy()


def main():
    raw_dir = Path("data/raw")
    save_dir = Path("data")
    batch_size = 128

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    model = load_model(
        raw_dir / "resnet18_cifar10.pth",
        device
    )
    preprocess = get_preprocess()

    manifest = pd.read_csv(save_dir / "manifest.csv")
    assert len(manifest) > 0
    assert manifest["sample_id"].is_unique

    load_image, classes = make_image_loader(manifest, raw_dir)

    n = len(manifest)
    all_logits = np.empty((n, 10), dtype=np.float32)
    all_tta_logits = np.empty((n, 3, 10), dtype=np.float32)

    start_time = time.time()

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        images = torch.stack([
            preprocess(load_image(row))
            for _, row in manifest.iloc[start:end].iterrows()
        ])

        logits, tta_logits = predict_batch(model, images, device)

        all_logits[start:end] = logits
        all_tta_logits[start:end] = tta_logits

        if start == 0 or (start // batch_size + 1) % 50 == 0 or end == n:
            elapsed = time.time() - start_time
            print(f"{end:,}/{n:,} images — {elapsed:.1f}s")

    prediction_rows = manifest.copy().reset_index(drop=True)
    predictions = all_logits.argmax(axis=1)

    prediction_rows["pred_label"] = predictions
    prediction_rows["pred_class"] = [
        classes[i] for i in predictions
    ]
    prediction_rows["correct"] = (
        predictions == prediction_rows["true_label"].to_numpy()
    ).astype("int8")

    for i in range(10):
        prediction_rows[f"logit_{i}"] = all_logits[:, i]

    probabilities = torch.softmax(
        torch.from_numpy(all_logits).double(),
        dim=1
    ).numpy()

    prediction_rows["raw_confidence"] = probabilities.max(axis=1)

    assert np.isfinite(all_logits).all()
    assert np.isfinite(all_tta_logits).all()
    assert prediction_rows["sample_id"].tolist() == manifest["sample_id"].tolist()

    save_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows.to_parquet(
        save_dir / "prediction_runs.parquet",
        index=False
    )
    np.save(save_dir / "tta_logits.npy", all_tta_logits)

    print(f"Saved {n:,} predictions.")
    print("TTA shape:", all_tta_logits.shape)
    print("Accuracy by dataset:")
    print(prediction_rows.groupby("dataset")["correct"].mean())


if __name__ == "__main__":
    main()