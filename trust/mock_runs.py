"""Owner: B. Mock versions of A's inference outputs, so B and C can build against the real schemas before A's GPU run.
Run: python -m trust.features --mock   (writes everything under data/mock/, then builds data/mock/features.csv)

Synthetic but self-consistent: embeddings = class prototype + noise, logits = a linear head on the embedding, TTA views =
the embedding re-noised. Corruption shrinks the class signal, adds noise and shifts the embedding off the train bank, harder
for noise than for digital. The numbers are not ResNet numbers; only the schemas and row alignment are the contract.

Files (same names and shapes as A's contract; rows of the .npy files follow prediction_runs.parquet row order):
  prediction_runs.parquet  sample_id, base_image_id, split, true_label, pred_label, correct, raw_confidence, top2_prob,
                           entropy, margin, corruption, severity, logit_0..logit_9 (float32)
  tta_logits.npy           N x 3 x 10 float32 (hflip, +2 px, -2 px)
  embeddings.npy           N x 512 float32
  train_embeddings.npy     M x 512 float32 + train_labels.npy (M,) int64: the kNN / Mahalanobis reference bank
"""
import numpy as np
import pandas as pd

from benchmark.make_splits import build_manifest
from trust.signals import softmax, softmax_stats

DIM = 512
N_CLASSES = 10
RADIUS = 3.6   # prototype norm: sets clean accuracy (~95%)
SCALE = 1.6    # logit scale: > 1 makes the head overconfident, like the real ResNet
TTA_NOISE = 0.35
# family -> (signal lost per severity step, extra noise per step, off-bank shift per step)
FAMILY_SHIFT = {"clean": (0, 0, 0), "natural": (0.15, 0.10, 0.3), "noise": (0.12, 0.10, 0.9),
                "blur": (0.08, 0.05, 0.6), "weather": (0.06, 0.05, 0.7), "digital": (0.05, 0.04, 0.5)}


def _head(emb, protos):
    return SCALE * (emb @ protos.T - 0.5 * (protos ** 2).sum(axis=1))


def make_mock(n_base=200, n_bank=10000, seed=0):
    """Returns manifest, prediction_runs (shuffled row order, like any file keyed by sample_id), tta_logits,
    embeddings, train_embeddings, train_labels."""
    rng = np.random.default_rng(seed)
    manifest = build_manifest(n_base, seed)
    protos = rng.normal(size=(N_CLASSES, DIM))
    protos *= RADIUS / np.linalg.norm(protos, axis=1, keepdims=True)
    shift_dir = {c: v / np.linalg.norm(v) for c in manifest["corruption"].unique()
                 for v in [rng.normal(size=DIM)]}

    # Each base image keeps one noise draw across all its corrupted versions.
    base_ids, base_pos = np.unique(manifest["base_image_id"].to_numpy(), return_inverse=True)
    base_noise = rng.normal(size=(len(base_ids), DIM))

    y = manifest["true_label"].to_numpy()
    sev = manifest["severity"].to_numpy().astype(float)
    sev[manifest["family"].to_numpy() == "natural"] = 1.0
    lost, noise, shift = (np.array([FAMILY_SHIFT[f][j] for f in manifest["family"]]) for j in range(3))
    direction = np.stack([shift_dir[c] for c in manifest["corruption"]])
    emb = ((1 - lost * sev)[:, None] * protos[y] + (1 + noise * sev)[:, None] * base_noise[base_pos]
           + (shift * sev)[:, None] * direction)
    views = emb[:, None, :] + TTA_NOISE * (1 + sev)[:, None, None] * rng.normal(size=(len(emb), 3, DIM))

    logits = _head(emb, protos).astype(np.float32)
    tta_logits = _head(views, protos).astype(np.float32)
    bank_labels = np.repeat(np.arange(N_CLASSES), n_bank // N_CLASSES)
    bank = (protos[bank_labels] + rng.normal(size=(len(bank_labels), DIM))).astype(np.float32)

    st = softmax_stats(logits)
    probs_sorted = np.sort(softmax(logits), axis=1)
    runs = manifest[["sample_id", "base_image_id", "split", "true_label"]].copy()
    runs["pred_label"] = st["pred"]
    runs["correct"] = (st["pred"] == y).astype(np.int64)
    runs["raw_confidence"] = st["raw_confidence"].astype(np.float32)
    runs["top2_prob"] = probs_sorted[:, -2].astype(np.float32)
    runs["entropy"] = st["entropy"].astype(np.float32)
    runs["margin"] = st["margin"].astype(np.float32)
    runs["corruption"] = manifest["corruption"]
    runs["severity"] = manifest["severity"]
    runs[[f"logit_{c}" for c in range(N_CLASSES)]] = logits

    order = rng.permutation(len(runs))
    return (manifest, runs.iloc[order].reset_index(drop=True), tta_logits[order], emb[order].astype(np.float32),
            bank, bank_labels.astype(np.int64))


def write_mock(out_dir, **kwargs):
    manifest, runs, tta_logits, emb, bank, bank_labels = make_mock(**kwargs)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    runs.to_parquet(out_dir / "prediction_runs.parquet", index=False)
    np.save(out_dir / "tta_logits.npy", tta_logits)
    np.save(out_dir / "embeddings.npy", emb)
    np.save(out_dir / "train_embeddings.npy", bank)
    np.save(out_dir / "train_labels.npy", bank_labels)
    print(f"wrote mock A outputs to {out_dir}: {len(runs):,} rows, bank {len(bank):,} x {bank.shape[1]}")
