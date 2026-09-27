"""Convert the core CIFAR-10 run into trust.fit_any's input format: the check that the generic entry point matches
trust.run_trust. Keeps the benchmark's splits (by photo) and uses the corruption family as the condition. The held-out
family's train / val / cal rows are dropped, so fit_any trains on exactly what run_trust's fold for that family sees and
its test report's "<family>" row is comparable with evaluation.json's held-out numbers.

Reads prediction_runs.parquet, tta_logits.npy, embeddings.npy, train_embeddings.npy, train_labels.npy (not in git;
python -m inference.run_inference and python -m inference.extract_embeddings make them).
Run: python -m trust.cifar_to_fit_any --out <folder> [--data data] [--hold-out blur]
     python -m trust.fit_any --data <folder> --out <layer folder>
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from trust.run_trust import FAMILIES, LOGITS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hold-out", default="blur", choices=FAMILIES, help="family kept out of train / val / cal")
    args = ap.parse_args()
    runs = pd.read_parquet(args.data / "prediction_runs.parquet",
                           columns=["sample_id", "base_image_id", "true_label", "true_class", "family", "split", *LOGITS])
    keep = ((runs.split == "test") | (runs.family != args.hold_out)).to_numpy()
    rows = runs[keep].rename(columns={"base_image_id": "group_id", "true_label": "label", "family": "condition"})
    rows["condition"] = rows.condition.replace("natural", "cifar10_1")
    args.out.mkdir(parents=True, exist_ok=True)
    # %.17g writes each float32 logit's exact value, so fit_any's signals match the core pipeline's
    rows[["sample_id", "group_id", "label", *LOGITS, "split", "condition"]].to_csv(
        args.out / "predictions.csv", index=False, float_format="%.17g")
    names = runs.drop_duplicates("true_label").sort_values("true_label").true_class
    (args.out / "classes.txt").write_text("\n".join(names) + "\n")
    for name in ("tta_logits.npy", "embeddings.npy"):   # same row order as prediction_runs.parquet
        np.save(args.out / name, np.load(args.data / name, mmap_mode="r")[keep])
    shutil.copyfile(args.data / "train_embeddings.npy", args.out / "reference_embeddings.npy")
    shutil.copyfile(args.data / "train_labels.npy", args.out / "reference_labels.npy")
    print(f"wrote {args.out}: {keep.sum():,} of {len(runs):,} predictions ({args.hold_out} held out of train / val / cal)")


if __name__ == "__main__":
    main()
