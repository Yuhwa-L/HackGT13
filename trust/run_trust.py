"""Owner: C. Entry point: python -m trust.run_trust
Reads data/manifest.csv, data/prediction_runs.parquet, data/features.csv; runs leave-one-family-out (headline: blur) + CIFAR-10.1.
Writes data/thresholds.json, data/evaluation.json, data/scores.parquet, data/models/.
"""
