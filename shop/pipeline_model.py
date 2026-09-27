"""Model stage of the shop pipeline, in its own process: torch only, never XGBoost (their OpenMP runtimes can
crash each other on macOS). Started by shop.run_pipeline; also runnable alone: python -m shop.pipeline_model
Writes data/shop/{logits,tta_logits,embeddings,bank_embeddings}.npy in manifest row order.
"""
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from shop.benchmark_data import ARRAY_NAMES as NAMES, N, VERSIONS, build_manifest, list_photos
from shop.config import shop_path
from shop.corrupt import corrupt
from shop.images import load_photo
from shop.model import EMB_DIM, device, forward, load_model

BATCH = 50


def _versions(args):
    path, key = args
    img = load_photo(path)
    return np.stack([corrupt(img, c, s, key) for c, s in VERSIONS])


def run_model(manifest, bank):
    dev, t0 = device(), time.time()
    model = load_model(dev)
    bases = manifest.drop_duplicates("base_image_id")[["path", "base_image_id"]].to_numpy()
    n = len(manifest)
    logits, tta, emb = np.empty((n, N), np.float32), np.empty((n, 3, N), np.float32), np.empty((n, EMB_DIM), np.float32)
    pos = {sid: i for i, sid in enumerate(manifest.sample_id)}
    with ProcessPoolExecutor() as pool:  # corruptions on CPU workers while the GPU runs the model
        for k, ((_, base), imgs) in enumerate(zip(bases, pool.map(_versions, [tuple(b) for b in bases], chunksize=4))):
            rows = [pos[f"{base}_{c}_s{s}"] for c, s in VERSIONS]
            logits[rows], tta[rows], emb[rows] = forward(model, imgs, dev)
            if k % 100 == 0:
                print(f"  {k}/{len(bases)} photos ({time.time() - t0:.0f}s)")
    bank_imgs = np.stack([load_photo(p) for p in bank.path])
    bank_emb = np.concatenate([forward(model, bank_imgs[i:i + BATCH], dev)[2] for i in range(0, len(bank_imgs), BATCH)])
    print(f"model: {n:,} images + {len(bank):,} bank photos in {time.time() - t0:.0f}s on {dev}")
    return logits, tta, emb, bank_emb



def main():
    photos = list_photos()
    bank = photos[photos.role == "bank"].reset_index(drop=True)
    arrays = run_model(build_manifest(photos), bank)
    for name, arr in zip(NAMES, arrays):
        np.save(shop_path(f"{name}.npy"), arr)


if __name__ == "__main__":
    main()
