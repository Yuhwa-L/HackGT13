"""Owner: A. Run every data/manifest.csv sample through the frozen ResNet: original + 3 TTA views (hflip, +2 px / -2 px shift, reflect pad).
Out: data/prediction_runs.parquet (one row per sample_id, including logit_0..logit_9) and data/tta_logits.npy (N x 3 x 10, same row order).
"""
