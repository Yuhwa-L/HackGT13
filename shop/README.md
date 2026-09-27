# The shopping assistant: the trust layer in agentic commerce (Visa challenge)

The trust layer is the product; this tab shows it in agentic commerce. A shopper picks a product photo; a frozen ImageNet
ViT-B/16 vision transformer identifies it (restricted to 30 product classes; a different architecture from the core
project's CIFAR ResNet-18, on purpose); the same trust layer as the core project, retrained on
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
python -m backend.main          # the core demo; the "Shopping assistant" tab appears when data/shop/ is present
# open it directly: http://localhost:8000/#shop
```

LLM (optional; the tab works offline): put the key in a repo-root `.env` (gitignored):
```
LLM_API_KEY=sk-...
LLM_MODEL=gpt-6-luna                         # default
LLM_BASE_URL=https://api.openai.com/v1       # default; any OpenAI-compatible endpoint works
```
**Set a spending limit in the OpenAI dashboard.** Each assist call is one short request (`reasoning_effort="none"`,
Structured Outputs, 5 s timeout). `DEMO_OFFLINE=1 python -m backend.main` forces the offline path; any LLM error or
timeout also falls back to it. The tab then shows "Offline: fixed replies", and the first time you type it explains
that the replies are pre-written and won't change with what you ask. The rules (gate, checkout tiers) are identical.

Demo story (buttons at the top of the tab): 1 TRUST one-tap · 2 TRUST order over the cap → confirm once ·
3 CAUTION → pick the item → confirm · 4 REJECT → retake tip, checkout locked.

## Live photo upload

"Upload or snap a photo" in the tab sends your photo (downsized to ~800 px in the browser) to the server, which runs
the same ViT, signals and trust layer on it in well under a second and returns TRUST / CAUTION / REJECT like any demo
photo. The assistant, retake coach and checkout work the same way. Photos of things outside the 30 product types
should come back CAUTION or REJECT. Uploads stay in server memory only (the last 100), and nothing is written to disk.

Needs torch and the ViT-B/16 weights at `data/shop/raw/vit_b_16-c867db91.pth` (330 MB) (the curl command below). Without them
the button says "Upload unavailable", and everything else still works.

**Why a separate model process:** on macOS, torch and XGBoost each ship their own OpenMP runtime, and in one process
whichever runs parallel code second can segfault. So the server (XGBoost, trust layer) never imports torch. The ViT
runs in `python -m shop.live_worker`, a torch-only child that exchanges images and outputs over stdin/stdout. The
pipeline is split the same way (`shop.pipeline_model` runs as its own process). Live scores match the pipeline's
cached scores exactly (checked on 65 cached versions).

## The tab

- **Snap a product:** upload a photo, use the webcam ("Use camera", which works on localhost with no network), or tap a sample.
  Samples can be degraded with 8 corruptions at 3 severities to show the gate reacting.
- **One screen, three panels:**
  - the photo with its decision badge;
  - the trust check: a plain-language "why", the model's own confidence next to the trust layer's `p_correct`, and on
    CAUTION or REJECT an inline **retake tip** from a pixel-based photo-quality check, with "Try a clearer shot";
  - the assistant chat.
- **Below the panels:** personalized picks, and a checkout panel whose tier follows the gate and the shopper's
  careful-checkout setting.
- **Your own profile by default** (name, budget, style, things you own, careful checkout), saved in the browser.
  Three fictional example shoppers stay available.

## Trust-trail receipts

Every mock purchase returns a receipt carrying the evidence it was made on:
- what the model said, and its own confidence;
- `p_correct` against the thresholds in force;
- the decision, and what the shopper confirmed;
- the checkout tier and the confirmations given;
- the careful-checkout setting, the items, and the time.

The evidence is fingerprinted with an HMAC-SHA256 under a per-server secret (`shop/receipt.py`).
`POST /api/shop/verify` checks a receipt, so changing any field (say, inflating `p_correct`) makes it fail. In the tab,
"Verify receipt" and "Tamper test" show both outcomes. Payment is a **mock one-time token** bound to that single order.
No card or account data exists anywhere in the system: it only shows where a real network token would sit, and why an
agent purchase should carry an auditable trust record.

## Who decides what

- `shop/gate.py`: allowed actions per decision; CAUTION upgrades to `trust_confirmed` only for a class the model proposed.
- `shop/checkout_policy.py`: the checkout tier from the effective decision and the order total.
- `shop/assistant.py`: the LLM writes the reply, ranking and `why_for_you`; server-side enforcement then replaces
  forbidden actions, drops unknown or out-of-class products and strips the cart when the gate forbids it.
- `shop/api.py`: recomputes the decision and the order total from the cache and catalog; never trusts the client.

## Rebuild from scratch (~25 min on an M3 MacBook; the ViT stage dominates, faster on a discrete GPU)

```bash
pip install -r requirements.txt          # includes the shop/ extras at the bottom
curl -L -o data/shop/raw/vit_b_16-c867db91.pth https://download.pytorch.org/models/vit_b_16-c867db91.pth
python -m shop.download_data      # streams 3 x 1.26 GB ImageNetV2 tars, keeps the 30 classes (~2 min)
python -m shop.run_pipeline       # corruptions, ViT (own process, ~20 min on M3), signals, trust layer; --reuse skips the model
python -m shop.catalog            # real-product catalog + fictional profiles
python -m shop.build_shop_cache   # data/shop/shop_cache.json (+ quality_ref.json)
# Rebuilds are byte-for-byte reproducible (every corruption is seeded, including skimage's impulse noise).
python -m shop.test_shop          # gate, checkout policy, assistant enforcement
python -m shop.threshold_stability # optional: fixed split vs cross-fitting, 10 reshuffles (~30 s)
```

Photos: **ImageNetV2** (Recht et al. 2019), a public re-collection of ImageNet validation-style images, 10 per class per
variant. The gated `imagenet-1k` needs a Hugging Face login, so we used this instead. After removing duplicates across
variants: 475 benchmark photos (split by photo 238 / 71 / 71 / 95) and 146 disjoint reference-bank photos (at least 3
per class). Each photo: clean + 8 corruptions (`imagecorruptions`) × severities 1 / 3 / 5 = 25 versions, 11,875 rows.

## Results (test split, 95 photos × 25 versions; demo-scale, not a benchmark claim)

Source: `data/shop/evaluation.json`. Model: ViT-B/16 (ImageNet weights), 30-way subset softmax. The trust layer is
**cross-fitted**: 5 folds over the 380 non-test photos give out-of-fold scores for isotonic calibration and both
thresholds, and the final model is fit on all 380. The 95 test photos are used only for these numbers.

| | |
|---|---|
| ViT accuracy (30-way) | 70% overall; 85% clean, 55% at severity 5 |
| At severity 5 | raw confidence 53% vs accuracy 55%; p_correct 52% |
| TRUST | 27.3% of test images, **0.8% error** (target 1%) |
| CAUTION | 8.6%, 8.3% error |
| REJECT | 64.0%, 46% error |
| Failure AUROC | raw confidence 0.859, trust layer 0.864 |
| AURC (lower is better) | raw confidence 0.0999, trust layer 0.0976 |
| ECE | raw 0.045, p_correct 0.059 |

**Why cross-fitting** (`python -m shop.threshold_stability`): across 10 random reshuffles of the non-test photos, the
old fixed split (71 calibration photos) moved the TRUST threshold with std 0.050, and TRUST-tier test error was
0.8% ± 0.7%. Cross-fitting cut that to std 0.015 and 0.6% ± 0.3%, with the same average behavior.

Say it plainly. Unlike the core project's CIFAR ResNet, this ViT is **not overconfident** under these corruptions (at
severity 5 it claims 53% and is right 55% of the time), so raw confidence is already well calibrated and p_correct
doesn't beat it on calibration. The trust layer's value here is the gate: the one-tap TRUST tier holds its 1% error
target, and failure ranking improves slightly. Scope cuts vs `plan.md`: no leave-one-family-out, no bootstrap,
3 severities.

## Real phone photos

`python -m shop.eval_real_photos <folder>` scores a folder of your own photos through the exact upload path. The
truth comes from each filename: `<class>_<condition>.jpg`, or `external<N>.jpg` for items outside the catalog. It writes
only aggregate results to `data/shop/real_photo_eval.json`, never the photos.

On 26 iPhone photos we took (8 product types in good, busy-background, washed-out, dark and blurry conditions, plus 4
out-of-catalog items):

| | |
|---|---|
| Model top-1 accuracy (catalog items) | 17/22 |
| TRUST (one-tap) purchases | 7, **all correct** |
| CAUTION: the right item was among the offered choices | 3/3 |
| Out-of-catalog items trusted | **0/4** |
| One-tap buys an agent would make at raw confidence ≥ 80% | 15, **1 wrong** (a washed-out laptop bought as a keyboard, 92% sure) |
| Correct identifications the gate still rejected | 7 (mostly busy backgrounds; the gate is conservative on real shift) |

The quality check flagged 2 of the 3 washed-out photos. It called the dark photos "hazy", not "dark": low-light
detection is its weak spot.

## Isolation

Nothing outside `shop/` imports it except the guarded `try: from shop.api import ShopAPI` in `backend/main.py`.
Everything `shop/` writes goes through `shop_path()` (asserts `data/shop/`). Delete `shop/` and the core demo,
`/predict` and all core tests still work; `/shop.js` becomes a no-op. The only core-code change is an optional
`n_classes` argument on `trust.signals.bank_signals` (default unchanged).
