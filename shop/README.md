# Snap to Shop: a trust-gated shopping agent (Visa wrapper)

The trust layer is the product; this tab shows it in agentic commerce. A shopper picks a product photo; a frozen ImageNet
ResNet-18 identifies it (restricted to 30 product classes); the same trust layer as the core project, retrained on
product photos, turns that into `p_correct` and TRUST / CAUTION / REJECT. A GenAI assistant (OpenAI `gpt-6-luna`) then
helps the shopper, and **code** decides what it may do and how much checkout friction applies.

| Decision | Assistant may | Checkout |
|---|---|---|
| TRUST | recommend, add to cart, checkout | One-tap up to $150, one confirmation above |
| CAUTION | compare the candidates, ask which one, recommend | Locked until the shopper picks a candidate; then confirm once (twice above $75) |
| REJECT | ask for a better photo | Locked |

Dollar caps are **demo settings** (`shop/config.py`). Products are **real, widely sold models**
(`shop/catalog_data.py`, listed from general knowledge, no web lookups) with **approximate list prices that may be
out of date**; we're not affiliated with any brand. Shoppers are **fictional**. Checkout is a **mock**: nothing is
charged and no payment data is collected anywhere.

## Run the demo

```bash
python -m backend.main          # the core demo; the "Snap to Shop" tab appears when data/shop/ is present
```

LLM (optional; the tab works offline): put the key in a repo-root `.env` (gitignored):
```
LLM_API_KEY=sk-...
LLM_MODEL=gpt-6-luna                         # default
LLM_BASE_URL=https://api.openai.com/v1       # default; any OpenAI-compatible endpoint works
```
**Set a spending limit in the OpenAI dashboard.** Each assist call is one short request (`reasoning_effort="none"`,
Structured Outputs, 5 s timeout). `DEMO_OFFLINE=1 python -m backend.main` forces the offline path; any LLM error or
timeout also falls back to it, and the tab shows "Offline mode".

Demo story (buttons at the top of the tab): 1 TRUST one-tap · 2 TRUST order over the cap → confirm once ·
3 CAUTION → pick the item → confirm · 4 REJECT → retake tip, checkout locked.

## Live photo upload

"Upload or snap a photo" in the tab sends your photo (downsized to ~800 px in the browser) to the server, which runs
the same ResNet, signals and trust layer on it in about 50 ms and returns TRUST / CAUTION / REJECT like any demo
photo. The assistant, retake coach and checkout work the same way. Photos of things outside the 30 product types
should come back CAUTION or REJECT. Uploads stay in server memory only (the last 100), and nothing is written to disk.

Needs torch and the ResNet-18 weights at `data/shop/raw/resnet18-f37072fd.pth` (the curl command below). Without them
the button says "Upload unavailable", and everything else still works.

**Why a separate model process:** on macOS, torch and XGBoost each ship their own OpenMP runtime, and in one process
whichever runs parallel code second can segfault. So the server (XGBoost, trust layer) never imports torch. The ResNet
runs in `python -m shop.live_worker`, a torch-only child that exchanges images and outputs over stdin/stdout. The
pipeline is split the same way (`shop.pipeline_model` runs as its own process). Live scores match the pipeline's
cached scores exactly (checked on 65 cached versions).

## Who decides what

- `shop/gate.py`: allowed actions per decision; CAUTION upgrades to `trust_confirmed` only for a class the model proposed.
- `shop/checkout_policy.py`: the checkout tier from the effective decision and the order total.
- `shop/assistant.py`: the LLM writes the reply, ranking and `why_for_you`; server-side enforcement then replaces
  forbidden actions, drops unknown or out-of-class products and strips the cart when the gate forbids it.
- `shop/api.py`: recomputes the decision and the order total from the cache and catalog; never trusts the client.

## Rebuild from scratch (~10 min on an M-series Mac)

```bash
pip install -r requirements.txt          # includes the shop/ extras at the bottom
curl -L -o data/shop/raw/resnet18-f37072fd.pth https://download.pytorch.org/models/resnet18-f37072fd.pth
python -m shop.download_data      # streams 3 x 1.26 GB ImageNetV2 tars, keeps the 30 classes (~2 min)
python -m shop.run_pipeline       # corruptions, model (own process), signals, trust layer (~2 min); --reuse skips the model
python -m shop.catalog            # real-product catalog + fictional profiles
python -m shop.build_shop_cache   # data/shop/shop_cache.json (+ quality_ref.json)
# Rebuilds are byte-for-byte reproducible (every corruption is seeded, including skimage's impulse noise).
python -m shop.test_shop          # gate, checkout policy, assistant enforcement
```

Photos: **ImageNetV2** (Recht et al. 2019), a public re-collection of ImageNet validation-style images, 10 per class per
variant. The gated `imagenet-1k` needs a Hugging Face login, so we used this instead. After removing duplicates across
variants: 475 benchmark photos (split by photo 238 / 71 / 71 / 95) and 146 disjoint reference-bank photos (at least 3
per class). Each photo: clean + 8 corruptions (`imagecorruptions`) × severities 1 / 3 / 5 = 25 versions, 11,875 rows.

## Results (test split, 95 photos × 25 versions; demo-scale, not a benchmark claim)

Source: `data/shop/evaluation.json`.

| | |
|---|---|
| ResNet accuracy (30-way) | 51% overall; 77% clean, 30% at severity 5 |
| At severity 5 | raw confidence 46% vs accuracy 30%; p_correct 29% |
| TRUST | 12.9% of test images, **1.0% error** (target 1%) |
| CAUTION | 2.4%, 8.9% error |
| REJECT | 84.8%, 57% error |
| Failure AUROC | raw confidence 0.816, trust layer 0.821 |
| ECE | raw 0.072, p_correct 0.079 (worse: only 71 validation photos for calibration) |

Say it plainly: with this little data the layer is conservative (it REJECTs most corrupted photos) and its calibration
is no better than raw confidence; what it does well is keep the TRUST tier at its 1% error target. The thresholds are
also volatile: they're set on only 71 calibration photos, so small changes move them (after making impulse noise
reproducible, tau_reject went from 0.56 to 0.80 and CAUTION shrank from 16% to 2% of test images). Scope cuts vs `plan.md`:
one split, no leave-one-family-out, no bootstrap, 3 severities.

## Isolation

Nothing outside `shop/` imports it except the guarded `try: from shop.api import ShopAPI` in `backend/main.py`.
Everything `shop/` writes goes through `shop_path()` (asserts `data/shop/`). Delete `shop/` and the core demo,
`/predict` and all core tests still work; `/shop.js` becomes a no-op. The only core-code change is an optional
`n_classes` argument on `trust.signals.bank_signals` (default unchanged).
