# Trust Issues

**A trust layer that says when a confident image classifier is about to be wrong.** Built at HackGT 13 (Oracle of the Deep track). The same layer also runs a **shopping assistant** that decides when an AI may buy for you ([below](#the-shopping-assistant-trust-in-agentic-commerce)).

Image classifiers report high confidence even when they are wrong, and it gets worse when photos are blurred, noisy or compressed. We put a trust layer on top of a frozen ResNet-18. For every prediction it returns one calibrated probability that the prediction is right (`p_correct`), a decision (TRUST, CAUTION or REJECT), and plain-language reasons. We tested it only on corruption types it never saw during training.

## The problem, measured

The ResNet-18 we used is 94.98% accurate on clean CIFAR-10 test photos. On heavily blurred versions of the same photos it is right **60.3%** of the time but still claims **87.5%** confidence. A cutoff tuned on clean photos to allow 5% errors lets through **20.0%** errors on blurred ones.

## What runs

```
CIFAR-10 test photos + CIFAR-10-C (8 corruptions × 5 severities) + CIFAR-10.1
  │ python -m benchmark.make_splits --n-base 10000   manifest.csv: 412,000 rows, split by photo
  │ python -m inference.run_inference                 ResNet-18 on each photo + 3 flipped/shifted copies
  │ python -m inference.extract_embeddings            512-d embeddings, plus 50,000 training photos as reference
  │ python -m trust.features                          8 signals per prediction
  │ python -m trust.run_trust                         trains, calibrates and evaluates the trust layer
  │ python -m backend.build_demo_cache                30 story photos + 10 random photos, all 41 versions each
  ▼ python -m backend.main                            the demo at http://localhost:8000
```

On an M1 Pro with its GPU the whole chain takes about 19 minutes: inference 11.9 min, embeddings 3.9 min, signals 2.4 min, trust layer 25 s, demo cache 13 s.

## How the trust layer decides

1. **Signals.** Eight numbers describe each prediction:
   - **Confidence:** raw confidence, entropy and top-2 margin of the ResNet's softmax.
   - **Stability:** whether the prediction survives a horizontal flip and ±2 px shifts (`tta_agree`, `tta_pconf`, `tta_std`).
   - **Familiarity:** how far the photo's embedding is from clean training photos (distance to the 10th-nearest neighbor, and Mahalanobis distance to the predicted class).
2. **Failure model.** XGBoost (300 trees) learns from these signals whether the ResNet was wrong. Clean photos get 25% of the training weight.
3. **Calibration.** Isotonic regression, fit on separate validation photos, turns the model's score into `p_correct`, a probability that means what it says.
4. **Decision.** Two cutoffs are chosen on a third set of calibration photos. REJECT means `p_correct` is below the cutoff that keeps accepted predictions at no more than 5% errors (0.742). TRUST means it is at or above the 1% cutoff (0.972). Everything in between is CAUTION.
5. **Reasons.** XGBoost's SHAP values are summed into three groups (stability, familiarity, confidence). The groups that push risk up the most become the reasons shown.

## How we tested it

- **Split by photo.** 5,000 training, 1,500 validation, 1,500 calibration and 2,000 test photos. All 41 versions of a photo stay in the same split.
- **Unseen corruption types.** Train on clean photos plus three corruption families, test on the fourth. We repeat this for noise, blur, weather and digital. Blur is the headline.
- **Real-world shift.** CIFAR-10.1 (2,000 newly collected photos) is never used for training.
- **Baselines.** Raw confidence; temperature scaling fit on clean validation data and on corrupted validation data; the flip/shift signal alone ("TTA only"); logistic regression.
- **Statistics.** 1,000 bootstrap resamples over photos give 95% intervals for every "better than" claim.

## Results

Held-out **blur** (20,000 test predictions, ResNet accuracy 79.7%):

| Method | Finds failures (AUROC ↑) | Risk–coverage (AURC ↓) | Calibration error (ECE ↓) | Error among the 20% most trusted |
|---|---|---|---|---|
| Raw confidence | 0.863 | 0.0557 | 0.137 | 1.0% |
| Temperature scaling, clean val | 0.863 | 0.0562 | 0.088 | 1.2% |
| Temperature scaling, corrupted val | 0.861 | 0.0572 | 0.030 | 1.3% |
| TTA only | 0.879 | 0.0507 | 0.096 | 0.5% |
| Logistic regression, all signals | 0.875 | 0.0514 | 0.036 | 0.35% |
| **Trust layer** | **0.884** | **0.0486** | **0.024** | **0.2%** |

What this supports:

- **Fixing the average is not enough.** Temperature scaling on corrupted data is almost as well calibrated as the trust layer (ECE 0.030 vs 0.024), but it is no better than raw confidence at telling *which* predictions will fail (AUROC 0.861). The trust layer ranks failures better than raw confidence (+0.021, 95% CI [+0.017, +0.026]), temperature scaling (+0.023) and TTA only (+0.005, [+0.002, +0.008]).
- **More automation at the same error.** With both cutoffs tuned on the same calibration photos for 5% errors, raw confidence accepts 59.3% of blurred photos (4.2% wrong). The trust layer accepts 63.7% (4.6% wrong).
- **Decisions.**
  - TRUST covers 42.0% of blurred photos at 1.1% error.
  - REJECT catches 592 of the 797 severe-blur predictions made with over 85% confidence that were wrong (74%).
  - Of all REJECTed predictions, 48% were wrong; the rest are the cost of caution.
- **A drift signal.** The share of predictions REJECTed rises from 14% on clean photos to 61% (blur), 79% (noise), 69% (digital) and 39% (weather) at the worst severity.

Every held-out family:

| Held out | Accuracy | AUROC raw → trust (95% CI of gain) | ECE raw / temp. scaling (corrupted) / trust | 5% cutoff tuned on clean → trust layer's cutoff |
|---|---|---|---|---|
| Noise | 49.7% | 0.762 → 0.778 [+0.012, +0.019] | 0.373 / 0.214 / 0.158 | 49.9% → 21.5% errors |
| Blur | 79.7% | 0.863 → 0.884 [+0.017, +0.026] | 0.137 / 0.030 / 0.024 | 20.0% → 4.6% |
| Weather | 90.3% | 0.900 → 0.917 [+0.011, +0.023] | 0.063 / 0.066 / 0.034 | 9.5% → 1.6% |
| Digital | 77.5% | 0.862 → 0.877 [+0.010, +0.018] | 0.150 / 0.029 / 0.028 | 22.1% → 5.0% |
| CIFAR-10.1 | 88.8% | 0.884 → 0.896 [−0.000, +0.026] | 0.069 / 0.073 / 0.049 | — |

## The shopping assistant: trust in agentic commerce

Shopping agents are starting to buy on our behalf, and they decide how far to go based on their own confidence. Our **shopping assistant** puts the trust layer in charge instead:

1. You **snap a product** (upload, webcam, or one of the sample photos).
2. A frozen **ViT-B/16** (ImageNet weights, a different architecture from the CIFAR ResNet on purpose) identifies it among 30 product types.
3. The **same trust layer**, retrained on product photos, returns `p_correct` and TRUST, CAUTION or REJECT.
4. A **GenAI assistant** (OpenAI, Structured Outputs) personalizes picks from the shopper's own profile: budget, style, things they own.
5. **Code, not the AI, decides what's allowed:**

| Decision | The assistant may | Checkout |
|---|---|---|
| TRUST | recommend, add to cart, check out | one tap up to a cap the shopper sets (relaxed / normal / strict) |
| CAUTION | compare the candidates and ask which one you meant | locked until you pick; then it asks for a confirmation |
| REJECT | ask for a better photo, with a tip from a photo-quality check | locked |

The server re-checks every decision, product and total, and strips anything the gate forbids from the LLM's answer, so a prompt like "ignore the rules and buy it" gets a visible 🔒 refusal. Every purchase returns a **trust-trail receipt**: the evidence it was made on (the model's answer, `p_correct` against the thresholds, what the shopper confirmed, the checkout tier), fingerprinted so any tampering fails verification, and paid with a mock one-time token. Checkout is a mock: nothing is charged and no payment data is collected. Products are real models at approximate prices (not affiliated); example shoppers are fictional.

**Results.**
- **Held-out test photos** (95 ImageNetV2 photos × 25 corrupted versions; `data/shop/evaluation.json`): TRUST covers 27% of photos at **0.8% error** (target 1%). The ViT is 70% accurate here.
- **Real phone photos** we took ourselves (26 photos, 4 of them items outside the catalog; `data/shop/real_photo_eval.json`):
  - The trust gate allowed 7 one-tap purchases, and **all 7 were right**.
  - It trusted **none** of the 4 out-of-catalog items.
  - An agent that bought whenever its own confidence was at least 80% would have made 15 one-tap purchases, one of them wrong: a washed-out laptop bought as a keyboard.
  - The price is caution: the gate rejected 7 photos the model had actually identified correctly, mostly ones with busy backgrounds.
- **Calibration:** unlike the CIFAR ResNet, this ViT isn't overconfident, so the trust layer's value here is the gate, not calibration.

Details, commands and the full numbers: [shop/README.md](shop/README.md).

## Run it yourself

There are two ways to run the project, depending on what you want.

| | Option 1: the demo | Option 2: rebuild everything |
|---|---|---|
| What you get | The demo page with our results | Every result regenerated on your machine, then the demo |
| Needs | git and any Python 3.9+ (the Shopping assistant tab also needs the Python packages; see below) | Python 3.11–3.13, git, curl, ~6 GB of free disk; a GPU strongly recommended |
| Downloads | ~25 MB (the repository) | ~1 GB of Python packages + ~3 GB of data transferred (1.5 GB kept) |
| Time | about 1 minute | about 45 minutes on a Mac with Apple silicon |

`pip install -r requirements.txt` installs Python packages only. It never downloads data; step 3 below does that.

### Option 1: run the demo

The demo uses only Python's standard library and loads nothing from the internet, so there is nothing to install.

**Try it with our data.** The repository includes the results of our full run, so the demo works right after cloning. They come from a pretrained ResNet-18 (94.98% accurate on clean CIFAR-10 test photos), kept frozen and never retrained, run on all 10,000 CIFAR-10 test photos in 41 versions each (clean, plus 8 corruptions at 5 severity levels) and on 2,000 CIFAR-10.1 photos. Our trust layer scored every prediction, and each corrupted photo is judged by a trust layer that never saw that type of corruption.

```bash
git clone https://github.com/Yuhwa-L/Trust-Issues.git
cd Trust-Issues
python3 -m backend.main          # Windows: py -m backend.main
```

It opens http://localhost:8000 in your browser. Press Ctrl+C to stop it; it writes nothing, so you can restart it any time.

**Set up the Shopping assistant tab** (about 5 minutes; macOS or Linux, on Windows use WSL). Run these in the repository folder:

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
  - Without step 1, the tab is simply hidden.
  - Without step 2, only upload and the camera are disabled (the button says "Upload unavailable").
  - Without step 3, the chat uses pre-written replies that follow the same rules (it says "Offline: fixed replies").
- **Model settings in `.env`:** the model defaults to `gpt-6-luna`. `LLM_MODEL=` changes it, `LLM_BASE_URL=` points at any OpenAI-compatible provider, and `LLM_TIMEOUT=` (seconds, default 5) sets how long to wait before using the pre-written reply.
- **Set a spending limit** in the OpenAI dashboard. Each chat reply is one short request.
- **Offline demo:** `DEMO_OFFLINE=1 python -m backend.main` forces the pre-written replies and needs no network.
- **Wait for the model:** after starting, the upload button says "Model loading…" for a few seconds.

What you see:

1. **The one-minute story.** One dog photo from the test split, defocus-blurred step by step:
   - Clean: "dog" (right), TRUST.
   - Severity 3: still "dog" (right), but `p_correct` 89%, so CAUTION.
   - Severity 4: "cat" at 91% confidence (wrong); `p_correct` 26%, so REJECT, with the reasons shown.

   The photo was picked to show all three decisions.
2. **Any photo, any corruption.** The gallery holds 30 picked photos and 10 random ones. A corruption picker and severity slider cover all 41 versions of each photo. For every version you see the confidence ladder (raw → temperature scaling → `p_correct`) against the REJECT/CAUTION/TRUST cutoffs, plus the reasons and the API response.
3. **Research results.** The headline numbers and charts on unseen corruptions, drawn from `data/evaluation.json`.
4. **Shopping assistant** (after the setup above).
   - The first visit asks for your shopping profile: budget, style, things you own, and how careful checkout should be. Three fictional example shoppers are also available.
   - The four "Demo story" buttons walk through TRUST (one-tap), a big order (confirm once), CAUTION (pick the item) and REJECT (retake tip), using a water-bottle photo.
   - You can also upload a photo, use the webcam, or pick a sample and degrade it with a corruption.
   - Every purchase returns a receipt with its trust trail. "Verify receipt" and "Tamper test" show that it can't be altered.
   - The card at the bottom shows how well the gate works on held-out test photos and on our own phone photos.

The page reads precomputed results from `data/demo_cache.json`. The same answers are available from the API in the service's response format:

```bash
curl -X POST localhost:8000/predict -d '{"sample_id": "c10_01977", "corruption": "defocus_blur", "severity": 4}'
```

### Option 2: rebuild everything from scratch

All commands are for macOS or Linux. On Windows, use WSL. Run every command from the repository folder.

**1. Get the code and system tools.**

```bash
git clone https://github.com/Yuhwa-L/Trust-Issues.git
cd Trust-Issues
brew install libomp              # macOS only: XGBoost needs it
```

**2. Create a Python environment and install the packages** (about 1 GB, 2–5 minutes).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Every later command assumes this environment is active. In a new terminal, run `source .venv/bin/activate` again.

**3. Download the data** into `data/raw/` (about 3 GB transferred, 1.5 GB kept, 15–30 minutes).

```bash
mkdir -p data/raw && cd data/raw

# CIFAR-10 (163 MB). The Toronto server is sometimes slow; if it crawls, retry later.
curl -L https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz | tar -xzf -

# CIFAR-10.1 (6 MB)
curl -LO https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_data.npy
curl -LO https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_labels.npy

# The pretrained ResNet-18 (45 MB)
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

**4. Run the pipeline.** Times are for an M1 Pro using its GPU. The code uses an NVIDIA GPU or an Apple-silicon GPU automatically and falls back to the CPU otherwise.

```bash
python -m benchmark.make_splits --n-base 10000   # 2 s     list of 412,000 photo versions and their splits
python -m inference.run_inference                # 12 min  ResNet-18 predictions, original + 3 flips/shifts
python -m inference.extract_embeddings           # 4 min   embeddings, plus 50,000 training photos as reference
python -m trust.features                         # 2 min   8 signals per prediction
python -m trust.run_trust                        # 25 s    train, calibrate and evaluate the trust layer
python -m backend.build_demo_cache               # 13 s    the demo's photos and results
```

Without a GPU the full run takes about 6 hours. To try it on a CPU, use `--n-base 2000` in the first command instead (about 1.5 hours); the numbers will then differ from the tables above because the test set is smaller. Everything is seeded, so the same settings always give the same results.

**5. Check the code** (optional, about 20 seconds, needs no data):

```bash
python -m trust.test_trust        # 12 checks: metrics, thresholds, calibration, leakage, file formats
python -m backend.test_backend    # 3 checks: demo cache, image encoding, server and /predict
python -m trust.test_features     # 9 checks: signals vs scipy/sklearn references, row alignment, bad inputs
python -m benchmark.test_benchmark  # 8 checks: loaders, split leakage, pixel-exact image loading (needs data/raw/)
python -m shop.test_shop          # 11 checks: the gate, checkout tiers, profiles, LLM output enforcement, receipts
```

**6. Open the demo** on your freshly built results:

```bash
python -m backend.main
```

**7. Rebuild the shopping assistant** (optional; about 25 minutes on an M3 MacBook, mostly the ViT; needs step 2 of the Shopping assistant setup above for the weights):

```bash
python -m shop.download_data      # ~2 min: streams 3 ImageNetV2 archives (1.26 GB each), keeps only the 30 product types
python -m shop.run_pipeline       # ~20 min: corruptions, ViT, signals, cross-fitted trust layer; --reuse skips the ViT
python -m shop.catalog            # product catalog and example shoppers
python -m shop.build_shop_cache   # the tab's photos and results
python -m shop.eval_real_photos ~/my_photos   # optional: score your own photos, named <class>_<condition>.jpg
```

Details and the full results are in [shop/README.md](shop/README.md). Rebuilds are reproducible: the same data gives identical results (only the demo cache's build timestamp changes).

### If something goes wrong

| Message | Fix |
|---|---|
| `XGBoost Library (libxgboost.dylib) could not be loaded` | macOS: `brew install libomp` |
| `Port 8000 is already in use` | The demo is already running: open http://localhost:8000, or start with `--port 8001` |
| `run_inference` prints `Device: cpu` and is very slow | No GPU found; use a GPU machine or `--n-base 2000` |
| `sample_ids differ` from `trust.run_trust` | Files from different runs got mixed in `data/`; rerun step 4 from the first command |
| `No such file or directory: data/raw/...` | A download in step 3 is missing or incomplete; rerun that line |
| The Shopping assistant tab doesn't appear | The terminal prints the reason ("Shopping assistant tab disabled: …"); usually the packages aren't installed: `pip install -r requirements.txt` |
| "Upload unavailable" in the Shopping assistant tab | The ViT weights are missing; see step 2 of the Shopping assistant setup |
| "Offline: fixed replies" in the chat | No `LLM_API_KEY` in `.env`, no network, or a reply took longer than `LLM_TIMEOUT` (5 s); fix the key or raise the timeout |
| The camera doesn't start | Allow camera access in the browser's address bar (it works on `localhost`), and close other apps using the camera |

The results the demo needs (`data/evaluation.json`, `thresholds.json`, `scores.parquet`, `demo_cache.json`, `models/`) are committed to git. The large intermediate files (the photo list, predictions, signals and about 1 GB of embeddings) are not. Anyone can regenerate them with step 4, and the team keeps a copy on the [team drive](https://drive.google.com/drive/folders/1kpS_yfUTMG8SPe4nEjUM1_m-JKSkldwR?usp=sharing).

## Repository

| Path | What it does |
|---|---|
| `benchmark/` | CIFAR loaders and the manifest with per-photo splits |
| `inference/` | ResNet-18 loading, inference with flips/shifts, embeddings |
| `trust/` | signals and features, temperature scaling, the failure model, calibration, cutoffs, metrics, the evaluation runner |
| `backend/` | demo cache builder and the local demo server |
| `frontend/` | the demo page (plain HTML/JS, hand-drawn SVG charts) |
| `shop/` | the shopping assistant: the ViT pipeline, the trust gate, checkout policy, the GenAI assistant, and the tab (optional: delete it and everything else still works) |
| `docs/` | team plans, pitch notes and the pitch plan |
| `data/` | results used by the demo; `data/raw/` holds downloads (not in git) |


## Data and references

- Data: CIFAR-10 (Krizhevsky, 2009); CIFAR-10-C (Hendrycks & Dietterich, ICLR 2019); CIFAR-10.1 (Recht et al., 2018).
- Model: pretrained ResNet-18 checkpoint `edadaltocg/resnet18_cifar10` on Hugging Face; for the shopping assistant, torchvision's ViT-B/16 (ImageNet weights) and ImageNetV2 photos (Recht et al., 2019).
- Methods: temperature scaling on perturbed data (Tomani et al., CVPR 2021), failure-detection evaluation (Jaeger et al., ICLR 2023), kNN distance on deep features (Sun et al., ICML 2022), Trust Score (Jiang et al., NeurIPS 2018).
