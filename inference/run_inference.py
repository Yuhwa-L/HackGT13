"""Pipeline step 3: original + 3 TTA views -> logits, embeddings. Owner: A.

Outputs (data/):
  prediction_runs.parquet   PREDICTION_RUNS_COLUMNS
  embeddings.npy            (N,512) float32
  embeddings_ids.npy        (N,) sample_id order
  tta_logits.npy            (N,3,10) float32, view order = config.tta.views

Requirements: batched, resumable (skip sample_ids already written), tqdm progress bar.
"""
from __future__ import annotations

from common import cli

STEP, OWNER = "run_inference", "A"


def run(cfg, device: str, limit: int | None) -> None:
    """TODO(A): read manifest, load images per dataset, run model on original + TTA views,
    compute softmax_stats (trust/signals.py), write outputs via common/io.py. Resume support."""
    raise NotImplementedError


def main() -> None:
    args, cfg = cli.parse(cli.base_parser(__doc__))
    if args.mock:
        print(f"[{STEP}] --mock: predictions/embeddings come from `make mock`; nothing to do.")
        return
    cli.not_implemented(STEP, OWNER)


if __name__ == "__main__":
    main()
