#!/usr/bin/env bash
# Manual-run only. Never invoked by setup or tests.
# Confirm every URL below before running. Writes into data/raw/.
#
#   CIFAR-10       via torchvision.datasets.CIFAR10(download=True)          ~170 MB
#   CIFAR-10-C     Zenodo record 2535967, CIFAR-10-C.tar                    ~2.9 GB
#   CIFAR-10.1 v6  modestyachts/CIFAR-10.1 GitHub repo:
#                  cifar10.1_v6_data.npy, cifar10.1_v6_labels.npy           ~6 MB
#
# TODO(B): fill in and verify the commands below, then run: bash benchmark/download.sh
set -euo pipefail

RAW="${RAW:-data/raw}"
mkdir -p "$RAW"

echo "TODO(B): CIFAR-10 -> python -c 'import torchvision; torchvision.datasets.CIFAR10(\"$RAW\", train=True, download=True); torchvision.datasets.CIFAR10(\"$RAW\", train=False, download=True)'"
echo "TODO(B): CIFAR-10-C -> curl -L -o $RAW/CIFAR-10-C.tar <zenodo 2535967 URL> && tar -xf $RAW/CIFAR-10-C.tar -C $RAW"
echo "TODO(B): CIFAR-10.1 -> curl -L -o $RAW/cifar10.1_v6_data.npy <github raw URL>; same for labels"
echo "Nothing downloaded. Edit this script once URLs are confirmed."
