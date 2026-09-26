# Requirements

This lists everything the project needs, for every role. Find the section for what you want to do and install what it lists.

## Adding a requirement

- **Python package:** add `package>=version  # why` under your role in [requirements.txt](requirements.txt).
- **Anything else** (system tool, dataset, model, Node package): add a row to the matching table below, with a copy-paste command in "How to get".
- Tell the team so they re-run the install.

## 1. Run the demo (any laptop; no GPU, no datasets)

The demo runs from precomputed files committed to git: `data/demo_cache.json`, `data/evaluation.json` and `data/thresholds.json`.

| What | Version | How to get | Notes |
|---|---|---|---|
| Git | any | https://git-scm.com/downloads | `git clone https://github.com/Yuhwa-L/HackGT13.git` |
| Python | 3.11–3.13 | https://www.python.org/downloads/ | Verified on 3.13.2 |
| Nothing else | | | The server uses only Python's standard library, and the page loads no fonts, libraries or CDNs, so it runs offline |

**Start the demo:** `python -m backend.main` from the repo root. It opens http://localhost:8000 (use `--port` to change it, `--no-browser` to skip opening a tab).

**Rebuild the demo data** after a new trust-layer run (needs the section 2 setup and `data/raw/`): `python -m trust.run_trust && python -m backend.build_demo_cache`.

## 2. Run the pipeline (team)

### 2.1 System

| What | Needed by | How to get | Notes |
|---|---|---|---|
| Python 3.11–3.13 | everyone | https://www.python.org/downloads/ | Verified on 3.13.2 |
| libomp | C (XGBoost), macOS only | `brew install libomp` | Without it, XGBoost fails with `libxgboost.dylib could not be loaded` |
| GPU or Apple silicon | A (inference) | Colab T4, or an M-series Mac (PyTorch uses MPS) | On an M1 Pro, 336k forward passes took 131 s, so the full 410k-row benchmark with TTA takes about 11 min. On CPU only, use 2,000 base images |
| About 5 GB free disk | A, B | — | Datasets take about 1.5 GB, embeddings about 1 GB |

On Windows, run the shell commands in Git Bash or WSL.

### 2.2 Python packages

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On Colab, skip the venv and run `!pip install -r requirements.txt`. The file sets minimum versions rather than exact pins, so Colab keeps its preinstalled CUDA build of torch.

### 2.3 Datasets and checkpoint (saved to `data/raw/`, which git ignores)

| What | Needed by | Size | Source |
|---|---|---|---|
| CIFAR-10 (python version) | A, B | 163 MB | https://www.cs.toronto.edu/~kriz/cifar.html |
| CIFAR-10-C: labels + the 8 MVP types | A, B | 2.9 GB streamed, 1.2 GB kept (about 10–15 min) | https://zenodo.org/records/2535967 |
| CIFAR-10.1 v6 | A, B | 6 MB | https://github.com/modestyachts/CIFAR-10.1 |
| ResNet-18 CIFAR-10 checkpoint | A | 45 MB | https://huggingface.co/edadaltocg/resnet18_cifar10 |

Run these from the repo root:

```bash
mkdir -p data/raw && cd data/raw

# CIFAR-10 (python version)
curl -L https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz | tar -xzf -

# CIFAR-10.1 v6
curl -LO https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_data.npy
curl -LO https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_labels.npy

# ResNet-18 checkpoint (use curl, not torch.hub: see Known issues)
curl -L -o resnet18_cifar10.pth https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/main/pytorch_model.bin

# CIFAR-10-C: streams the 2.9 GB tar and keeps only the labels and the 8 MVP types
curl -L "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1" | tar -xf - \
  CIFAR-10-C/labels.npy \
  CIFAR-10-C/gaussian_noise.npy CIFAR-10-C/impulse_noise.npy \
  CIFAR-10-C/defocus_blur.npy CIFAR-10-C/motion_blur.npy \
  CIFAR-10-C/fog.npy CIFAR-10-C/brightness.npy \
  CIFAR-10-C/contrast.npy CIFAR-10-C/jpeg_compression.npy

cd ../..
```

The result should look like this:

```
data/raw/
├── cifar-10-batches-py/     data_batch_1..5, test_batch, batches.meta
├── CIFAR-10-C/              labels.npy + 8 files, each (50000, 32, 32, 3) uint8
├── cifar10.1_v6_data.npy    (2000, 32, 32, 3) uint8
├── cifar10.1_v6_labels.npy
└── resnet18_cifar10.pth
```

### 2.4 Known issues

| Problem | Fix |
|---|---|
| `CERTIFICATE_VERIFY_FAILED` from `torch.hub.load_state_dict_from_url` (python.org Python on macOS) | Download the checkpoint with curl as in 2.3, then load it with `torch.load("data/raw/resnet18_cifar10.pth", map_location="cpu")` |
| `XGBoost Library (libxgboost.dylib) could not be loaded` on macOS | `brew install libomp` |
