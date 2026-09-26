"""Owner: A. 512-d penultimate embeddings via model.forward_head(model.forward_features(x), pre_logits=True).
Out: data/embeddings.npy (manifest samples, same row order as prediction_runs.parquet) and
data/train_embeddings.npy + data/train_labels.npy (CIFAR-10 train set: the kNN / Mahalanobis reference bank).
"""
