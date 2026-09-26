"""Owner: B. Build data/manifest.csv, one row per evaluated image: sample_id, base_image_id, dataset, image_index,
true_label, true_class, corruption, family, severity, split.
Split by base_image_id, never by row: train 50 / val 15 / cal 15 / test 20, stratified by class, fixed seed.
CIFAR-10.1 rows: dataset=cifar10_1, family=natural, split=test.
"""
