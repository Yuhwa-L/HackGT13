"""Owner: B. Assemble data/features.csv: sample_id, raw_confidence, entropy, margin, tta_agree, tta_pconf, tta_std,
knn_dist, maha_pred, failure. Corruption, family and severity are for analysis only, never model inputs.

Reads (from A): prediction_runs.parquet (sample_id, pred_label, correct, logit_0..logit_9), tta_logits.npy (N x 3 x 10)
and embeddings.npy (N x 512) in prediction_runs row order, train_embeddings.npy + train_labels.npy (reference bank).
Joins on sample_id and writes rows in manifest.csv order. Prints a sanity report (accuracy, feature means by
family x severity, single-feature failure AUROC) to share with C.

Run: python -m trust.features                 real files in data/
     python -m trust.features --mock          mock A outputs in data/mock/ (see trust/mock_runs.py), then features
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trust import signals

DATA = Path(__file__).resolve().parents[1] / "data"
N_CLASSES = 10
LOGITS = [f"logit_{c}" for c in range(N_CLASSES)]
FEATURES = ["raw_confidence", "entropy", "margin", "tta_agree", "tta_pconf", "tta_std", "knn_dist", "maha_pred"]
COLUMNS = ["sample_id", *FEATURES, "failure"]
SAFER_WHEN_HIGH = {"raw_confidence", "margin", "tta_agree", "tta_pconf"}  # flipped to a risk score for AUROC


def load_inputs(data_dir):
    manifest = pd.read_csv(data_dir / "manifest.csv")
    runs = pd.read_parquet(data_dir / "prediction_runs.parquet", columns=["sample_id", "pred_label", "correct", *LOGITS])
    tta = np.load(data_dir / "tta_logits.npy", mmap_mode="r")
    emb = np.load(data_dir / "embeddings.npy", mmap_mode="r")
    bank = np.load(data_dir / "train_embeddings.npy")
    bank_labels = np.load(data_dir / "train_labels.npy").astype(np.int64)

    n = len(runs)
    assert runs["sample_id"].is_unique, "duplicate sample_id in prediction_runs"
    missing = ~manifest["sample_id"].isin(runs["sample_id"])
    extra = ~runs["sample_id"].isin(manifest["sample_id"])
    assert not missing.any() and not extra.any(), \
        f"prediction_runs vs manifest: {missing.sum()} manifest rows missing, {extra.sum()} unknown sample_ids"
    assert tta.shape == (n, 3, N_CLASSES), f"tta_logits.npy is {tta.shape}, expected {(n, 3, N_CLASSES)}"
    assert emb.ndim == 2 and len(emb) == n, f"embeddings.npy is {emb.shape}, expected {n} rows"
    assert bank.shape[1] == emb.shape[1] and len(bank_labels) == len(bank), "train bank shape mismatch"
    assert set(np.unique(bank_labels)) == set(range(N_CLASSES)), "train bank must cover all 10 classes"
    return manifest, runs, tta, emb, bank, bank_labels


def build_features(manifest, runs, tta, emb, bank, bank_labels):
    """Features in manifest row order. Signals are computed in runs order, since the .npy rows follow it."""
    logits = runs[LOGITS].to_numpy()
    sm = signals.softmax_stats(logits)
    true = manifest.set_index("sample_id")["true_label"].loc[runs["sample_id"]].to_numpy()
    bad_pred = (sm["pred"] != runs["pred_label"].to_numpy()).sum()
    assert bad_pred == 0, f"{bad_pred} rows: pred_label != argmax(logits)"
    correct = runs["correct"].to_numpy().astype(np.int64)  # A may store it as bool
    assert (correct == (sm["pred"] == true)).all(), "correct != (pred_label == manifest true_label)"

    t = time.time()
    tta_st = signals.tta_stats(logits, tta)
    knn = signals.knn_dist(emb, bank)
    means, precision = signals.fit_mahalanobis(bank, bank_labels)
    maha = signals.maha_pred(emb, sm["pred"], means, precision)
    print(f"signals for {len(runs):,} rows against a {len(bank):,}-row bank: {time.time() - t:.1f} s")

    feats = pd.DataFrame({"sample_id": runs["sample_id"].to_numpy(),
                          **{k: sm[k] for k in ("raw_confidence", "entropy", "margin")}, **tta_st,
                          "knn_dist": knn, "maha_pred": maha, "failure": 1 - correct})
    feats = feats.set_index("sample_id").loc[manifest["sample_id"]].reset_index()[COLUMNS]
    assert np.isfinite(feats[FEATURES].to_numpy()).all(), "NaN or inf in features"
    return feats


def auroc(risk, failure):
    """P(risk of a failure > risk of a success), ties counted half. NaN if only one class."""
    failure = np.asarray(failure, dtype=bool)
    n1, n0 = failure.sum(), (~failure).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = pd.Series(risk).rank().to_numpy()
    return (ranks[failure].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def report(feats, manifest):
    df = feats.merge(manifest[["sample_id", "dataset", "family", "severity"]], on="sample_id")
    df["accuracy"] = 1 - df["failure"]
    pd.set_option("display.width", 200)
    print("\naccuracy by dataset:", df.groupby("dataset")["accuracy"].mean().round(4).to_dict())
    for col in ["accuracy", "raw_confidence", "tta_pconf", "knn_dist", "maha_pred"]:
        print(f"\nmean {col} by family x severity")
        print(df.pivot_table(index="family", columns="severity", values=col).round(3).to_string())

    rows = {}
    for fam, g in [("all", df), *df.groupby("family")]:
        rows[fam] = {f: auroc(-g[f] if f in SAFER_WHEN_HIGH else g[f], g["failure"]) for f in FEATURES}
    print("\nsingle-feature failure AUROC (higher = better at ranking failures)")
    print(pd.DataFrame(rows).T.round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--mock", action="store_true", help="write mock A outputs to <data-dir>/mock first and use them")
    ap.add_argument("--mock-n-base", type=int, default=200)
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    data_dir = args.data_dir
    if args.mock:
        from trust.mock_runs import write_mock
        data_dir = data_dir / "mock"
        write_mock(data_dir, n_base=args.mock_n_base)

    manifest, *inputs = load_inputs(data_dir)
    feats = build_features(manifest, *inputs)
    out = data_dir / "features.csv"
    feats.to_csv(out, index=False, float_format="%.10g")
    print(f"wrote {out}: {len(feats):,} rows")
    if not args.no_report:
        report(feats, manifest)


if __name__ == "__main__":
    main()
