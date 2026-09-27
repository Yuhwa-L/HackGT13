# Requirements and setup

What the project is and why it matters: see the [README](README.md). This page covers how to run and rebuild it.

| I want to… | Go to | Needs | Time |
|---|---|---|---|
| Open the demo with our results | [1. Run the demo](#1-run-the-demo) | git and Python 3.9+ | about 1 minute |
| Use the Shopping assistant tab | [2. Shopping assistant tab](#2-shopping-assistant-tab) | the Python packages; optionally 330 MB of ViT weights and an OpenAI key | about 5 minutes |
| Fit the trust layer to my own model | [docs/USE_WITH_YOUR_MODEL.md](docs/USE_WITH_YOUR_MODEL.md) | the Python packages ([3.2](#32-python-packages)) | minutes |
| Regenerate every result | [3. Rebuild everything](#3-rebuild-everything-from-scratch) | Python 3.11–3.13, curl, ~6 GB of disk; a GPU strongly recommended | about 45 minutes on Apple silicon |

`pip install -r requirements.txt` installs Python packages only. It never downloads data; [3.3](#33-datasets-and-checkpoint) does that. All shell commands are for macOS or Linux; on Windows use WSL or Git Bash. Run every command from the repository folder.

## 1. Run the demo

The demo runs from precomputed files committed to git, so it needs no GPU, no datasets and no packages. The server uses only Python's standard library, and the page loads no fonts, libraries or CDNs, so it runs offline.

| What | Version | How to get |
|---|---|---|
| Git | any | https://git-scm.com/downloads |
| Python | 3.9+ (verified on 3.13.2) | https://www.python.org/downloads/ |

```bash
git clone https://github.com/Yuhwa-L/Trust-Issues.git
cd Trust-Issues
python3 -m backend.main          # Windows: py -m backend.main
```

- It opens http://localhost:8000. Use `--port` to change the port and `--no-browser` to skip opening a tab.
- Press Ctrl+C to stop it. It writes nothing, so you can restart it any time.
- The page reads `data/demo_cache.json`, `data/evaluation.json` and `data/thresholds.json`. The same answers are available from the API:

```bash
curl -X POST localhost:8000/predict -d '{"sample_id": "c10_01977", "corruption": "defocus_blur", "severity": 4}'
```

## 2. Shopping assistant tab

This tab shows the same trust layer on a second model (a ViT-B/16). About 5 minutes to set up:

```bash
# 1. Python packages
brew install libomp                  # macOS only: XGBoost needs it
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # ~1 GB, 2–5 minutes

# 2. For photo upload and the webcam: the ViT-B/16 weights (330 MB)
mkdir -p data/shop/raw
curl -L -o data/shop/raw/vit_b_16-c867db91.pth https://download.pytorch.org/models/vit_b_16-c867db91.pth

# 3. For the AI assistant: an OpenAI API key, in a .env file in the repository folder (gitignored, never committed)
echo "LLM_API_KEY=sk-..." >> .env    # >> adds the line without overwriting an existing .env

# 4. Start the demo and open the tab
python -m backend.main               # then open http://localhost:8000/#shop
```

- **Each step is optional in its own way:**
  - Without step 1, the tab is hidden.
  - Without step 2, only upload and the camera are disabled (the button says "Upload unavailable"). The demo story buttons and sample photos still work.
  - Without step 3, the chat uses pre-written replies that follow the same rules (it says "Offline: fixed replies").
- **Model settings in `.env`:** the model defaults to `gpt-6-luna`. `LLM_MODEL=` changes it, `LLM_BASE_URL=` points at any OpenAI-compatible provider, and `LLM_TIMEOUT=` (seconds, default 5) sets how long to wait before using the pre-written reply.
- **Set a spending limit** in the OpenAI dashboard. Each chat reply is one short request.
- **Offline demo:** `DEMO_OFFLINE=1 python -m backend.main` forces the pre-written replies and needs no network.
- **Wait for the model:** after starting, the upload button says "Model loading…" for a few seconds.
- **Receipts** only verify on the server run that issued them, so don't restart the server between buying and pressing "Verify receipt".

How the tab works and its full results: [shop/README.md](shop/README.md).

## 3. Rebuild everything from scratch

This regenerates every result on your machine. It needs about 3 GB of downloads (1.5 GB kept), ~1 GB of Python packages, and about 45 minutes on a Mac with Apple silicon.

### 3.1 System

| What | Needed for | How to get | Notes |
|---|---|---|---|
| Python 3.11–3.13 | everything | https://www.python.org/downloads/ | Verified on 3.13.2 |
| curl | downloads | preinstalled on macOS and most Linux | |
| libomp | XGBoost, macOS only | `brew install libomp` | Without it, XGBoost fails with `libxgboost.dylib could not be loaded` |
| GPU or Apple silicon | inference | an NVIDIA GPU, an M-series Mac (PyTorch uses MPS), or Colab T4 | Picked automatically. Without one, see the CPU note in 3.4 |
| About 6 GB free disk | datasets and embeddings | — | Datasets take about 1.5 GB, embeddings about 1 GB |

### 3.2 Python packages

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Every later command assumes this environment is active. In a new terminal, run `source .venv/bin/activate` again.

On Colab, skip the venv and run `!pip install -r requirements.txt`. The file sets minimum versions rather than exact pins, so Colab keeps its preinstalled CUDA build of torch.

### 3.3 Datasets and checkpoint

These go into `data/raw/`, which git ignores (about 3 GB transferred, 1.5 GB kept, 15–30 minutes).

| What | Size | Source |
|---|---|---|
| CIFAR-10 (python version) | 163 MB | https://www.cs.toronto.edu/~kriz/cifar.html |
| CIFAR-10-C: labels + the 8 corruptions we use | 2.9 GB streamed, 1.2 GB kept (about 10–15 min) | https://zenodo.org/records/2535967 |
| CIFAR-10.1 v6 | 6 MB | https://github.com/modestyachts/CIFAR-10.1 |
| ResNet-18 CIFAR-10 checkpoint | 45 MB | https://huggingface.co/edadaltocg/resnet18_cifar10 |

```bash
mkdir -p data/raw && cd data/raw

# CIFAR-10 (163 MB). The Toronto server is sometimes slow; if it crawls, retry later.
curl -L https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz | tar -xzf -

# CIFAR-10.1 (6 MB)
curl -LO https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_data.npy
curl -LO https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_labels.npy

# The pretrained ResNet-18 (45 MB). Use curl, not torch.hub (see Troubleshooting)
curl -L -o resnet18_cifar10.pth https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/main/pytorch_model.bin

# CIFAR-10-C: streams the 2.9 GB archive and keeps only the labels and the 8 corruptions we use (1.2 GB)
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

### 3.4 Run the pipeline

```
CIFAR-10 test photos + CIFAR-10-C (8 corruptions × 5 severities) + CIFAR-10.1
  │ benchmark.make_splits        manifest.csv: 412,000 rows, split by photo
  │ inference.run_inference      ResNet-18 on each photo + 3 flipped/shifted copies
  │ inference.extract_embeddings 512-d embeddings, plus 50,000 training photos as reference
  │ trust.features               8 signals per prediction
  │ trust.run_trust              trains, calibrates and evaluates the trust layer
  │ backend.build_demo_cache     30 story photos + 10 random photos, all 41 versions each
  ▼ backend.main                 the demo at http://localhost:8000
```

Times are for an M1 Pro using its GPU (about 19 minutes in total):

```bash
python -m benchmark.make_splits --n-base 10000   # 2 s     list of 412,000 photo versions and their splits
python -m inference.run_inference                # 12 min  ResNet-18 predictions, original + 3 flips/shifts
python -m inference.extract_embeddings           # 4 min   embeddings, plus 50,000 training photos as reference
python -m trust.features                         # 2 min   8 signals per prediction
python -m trust.run_trust                        # 25 s    train, calibrate and evaluate the trust layer
python -m backend.build_demo_cache               # 13 s    the demo's photos and results
python -m backend.main                           #         open the demo on your fresh results
```

- **No GPU:** the full run takes about 6 hours on a CPU. To try it faster, use `--n-base 2000` in the first command (about 1.5 hours). The numbers will then differ from the README because the test set is smaller.
- **Reproducible:** everything is seeded, so the same settings always give the same results.
- **Only the trust layer changed?** Rerun just the last two steps: `python -m trust.run_trust && python -m backend.build_demo_cache`.

### 3.5 Check the code

Optional, about 20 seconds, and most checks need no data:

```bash
python -m trust.test_trust          # 15 checks: metrics, thresholds, calibration, leakage, file formats, fit_any
python -m backend.test_backend      # 3 checks: demo cache, image encoding, server and /predict
python -m trust.test_features       # 9 checks: signals vs scipy/sklearn references, row alignment, bad inputs
python -m benchmark.test_benchmark  # 8 checks: loaders, split leakage, pixel-exact image loading (needs data/raw/)
python -m shop.test_shop            # 11 checks: the gate, checkout tiers, profiles, LLM output enforcement, receipts
```

### 3.6 Rebuild the shopping assistant

Optional; about 25 minutes on an M3 MacBook, mostly the ViT. It needs the ViT weights from [section 2](#2-shopping-assistant-tab), step 2.

```bash
python -m shop.download_data      # ~2 min: streams 3 ImageNetV2 archives (1.26 GB each), keeps only the 30 product types
python -m shop.run_pipeline       # ~20 min: corruptions, ViT, signals, cross-fitted trust layer; --reuse skips the ViT
python -m shop.catalog            # product catalog and example shoppers
python -m shop.build_shop_cache   # the tab's photos and results
python -m shop.eval_real_photos ~/my_photos   # optional: score your own photos, named <class>_<condition>.jpg
```

Rebuilds are reproducible: the same data gives identical results (only the demo cache's build timestamp changes). Details: [shop/README.md](shop/README.md).

### 3.7 What's in git

The results the demo needs are committed: `data/evaluation.json`, `thresholds.json`, `scores.parquet`, `demo_cache.json`, `models/`, and `data/shop/`. The large intermediate files (the photo list, predictions, signals and about 1 GB of embeddings) are not. Anyone can regenerate them with 3.4, and the team keeps a copy on the [team drive](https://drive.google.com/drive/folders/1kpS_yfUTMG8SPe4nEjUM1_m-JKSkldwR?usp=sharing).

## 4. Troubleshooting

| Message | Fix |
|---|---|
| `Port 8000 is already in use` | The demo is already running: open http://localhost:8000, or start with `--port 8001` |
| `XGBoost Library (libxgboost.dylib) could not be loaded` | macOS: `brew install libomp` |
| `CERTIFICATE_VERIFY_FAILED` from `torch.hub.load_state_dict_from_url` (python.org Python on macOS) | Download the checkpoint with curl as in 3.3, then load it with `torch.load("data/raw/resnet18_cifar10.pth", map_location="cpu")` |
| `run_inference` prints `Device: cpu` and is very slow | No GPU found; use a GPU machine or `--n-base 2000` |
| `sample_ids differ` from `trust.run_trust` | Files from different runs got mixed in `data/`; rerun 3.4 from the first command |
| `No such file or directory: data/raw/...` | A download in 3.3 is missing or incomplete; rerun that line |
| The Shopping assistant tab doesn't appear | The terminal prints the reason ("Shopping assistant tab disabled: …"); usually the packages aren't installed: `pip install -r requirements.txt` |
| "Upload unavailable" in the Shopping assistant tab | The ViT weights are missing; see section 2, step 2 |
| "Offline: fixed replies" in the chat | No `LLM_API_KEY` in `.env`, no network, or a reply took longer than `LLM_TIMEOUT` (5 s); fix the key or raise the timeout |
| The camera doesn't start | Allow camera access in the browser's address bar (it works on `localhost`), and close other apps using the camera |

## Adding a requirement

- **Python package:** add `package>=version  # why` under your role in [requirements.txt](requirements.txt).
- **Anything else** (system tool, dataset, model): add a row to the matching table above, with a copy-paste command.
- Tell the team so they re-run the install.
