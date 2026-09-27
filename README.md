# Trust Issues

**AI models are often confidently wrong. Trust Issues tells you when.**

Trust Issues is a trust layer that sits on top of any classifier. For every prediction it gives you:

- **`p_correct`**: an honest chance that the prediction is right. When it says 70%, it is right about 70% of the time.
- **A decision**: **TRUST** (act on it automatically), **CAUTION** (have a person check it) or **REJECT** (don't use it).
- **Reasons** in plain words, like "the answer changed when the photo was shifted slightly".

The model underneath is never changed or retrained. 

## The problem

An image model (ResNet-18) is 95% accurate on clean photos. On heavily blurred photos it is right only **60.3%** of the time, yet it still claims **87.5%** confidence. Its confidence can't tell you which answers to trust.

## One photo, three decisions

From the demo: a dog photo, blurred step by step. The trust layer never saw blur during training.

| Photo | Model says | Model's confidence | Trust layer (`p_correct`) | Decision |
|---|---|---|---|---|
| Clean | dog ✓ | over 99% | over 99% | **TRUST** |
| Some blur | dog ✓ | 99% | 89% | **CAUTION** |
| Heavy blur | **cat ✗** | 91% | 26% | **REJECT** |

Why REJECT: the answer flipped when the photo was nudged slightly, and the photo looks unlike the ones the model learned from.

## How it works

For every prediction, the layer checks three things:

1. **Confidence:** how sure is the model?
2. **Stability:** does the answer stay the same when the photo is slightly flipped or shifted?
3. **Familiarity:** does the photo look like the ones the model learned from?

A small second model learns from these checks when the classifier tends to be wrong, and turns them into `p_correct`. Two cutoffs then pick the decision: TRUSTed answers should be wrong at most 1% of the time, and everything that isn't REJECTed at most 5%. The reasons are the checks that raised the risk the most.

Under the hood: XGBoost, isotonic calibration, and SHAP values for the reasons.

## Results

We tested the layer only on damage it never saw. There are four kinds of damage (noise, blur, weather, digital): we train on three, test on the fourth, and rotate. No test photo is ever used for training.

On blur it never saw:

- **It keeps its promise.** A "5% errors" rule based on the model's own confidence lets **20%** errors through. The trust layer's rule lets through **4.6%**.
- **It approves more, just as safely.** When both rules are tuned on the same photos, both stay under 5% errors, but the trust layer auto-approves 64% of photos against 59%.
- **It catches confident mistakes.** On heavy blur, it rejects 74% of the wrong answers the model gave with over 85% confidence.
- **It raises an alarm.** Its reject rate climbs from 14% on clean photos to 61% on the worst blur, so you can see when your inputs go bad.
- **It beats the standard fix.** Temperature scaling gets the average confidence right, but it is no better at spotting *which* answers are wrong (AUROC 0.861 against our 0.884).

Errors actually let through by a "5% errors" rule, set the usual way (the model's own confidence, tuned on clean photos) and by the trust layer:

| Damage it never saw | Model right | Usual rule | Trust layer |
|---|---|---|---|
| Noise | 49.7% | 49.9% | 21.5% |
| Blur | 79.7% | 20.0% | 4.6% |
| Weather | 90.3% | 9.5% | 1.6% |
| Digital | 77.5% | 22.1% | 5.0% |

On all four, the trust layer is also better at spotting wrong answers (AUROC up 0.015 to 0.021, with every 95% confidence interval above zero). All the numbers are in the Research results tab and `data/evaluation.json`.

## Run the demo

**The demo uses trust layers we already fitted:** one for the ResNet-18 on CIFAR-10, and one for the ViT-B/16 on product photos. Their results are precomputed and committed to this repository, so the demo trains nothing, needs no GPU and works offline. To build a trust layer for your own model, follow the [three steps below](#works-with-any-classifier).

You need git and Python 3.9 or newer, and there is nothing to install:

```bash
git clone https://github.com/Yuhwa-L/Trust-Issues.git
cd Trust-Issues
python3 -m backend.main          # Windows: py -m backend.main
```

It opens http://localhost:8000 in your browser. Press Ctrl+C to stop it.

| Tab | What you see |
|---|---|
| **Live demo** | One dog photo, blurred step by step, gets all three decisions ([above](#one-photo-three-decisions)). Then try any of 40 test photos with 8 kinds of damage at 5 strengths. |
| **Research results** | The evidence from 412,000 predictions: confidence vs. reality, whether a "5% errors" promise holds, and the drift alarm. |
| **Shopping assistant** | The same layer on a different model, used for an AI shopping assistant ([below](#proof-the-same-layer-on-a-second-model)). This tab needs a few Python packages: see [REQUIREMENTS.md](REQUIREMENTS.md#2-shopping-assistant-tab). |

Setup beyond this, rebuilding every result from scratch, and troubleshooting are in [REQUIREMENTS.md](REQUIREMENTS.md).

## Works with any classifier

The trust layer only reads a model's outputs, so it works with any classification model, from any framework and with any architecture: a CNN, a transformer, a gradient-boosted model or a logistic regression. You fit one trust layer per model, on that model's own labeled outputs, in three steps:

1. **Export your model's outputs** on labeled examples it hasn't trained on ([how, with PyTorch and scikit-learn code](docs/USE_WITH_YOUR_MODEL.md#2-export-your-models-outputs)). Only the raw scores (logits) are required. Two optional extras make the layer stronger: the model's scores on slightly changed inputs, and its embeddings.
2. **Fit the trust layer.** It takes seconds: 3 s on 332,000 predictions, plus about 80 s once if you give it embeddings. It also prints a report on a held-back test split, so you see how well it works on your model before you rely on it.
   ```bash
   python -m trust.fit_any --data my_model_outputs --out my_trust_layer
   ```
3. **Score new predictions.** Each one gets `p_correct`, TRUST / CAUTION / REJECT, and its reasons.
   ```bash
   python -m trust.fit_any --score my_trust_layer --data new_outputs --out scores.csv
   ```

**Step-by-step guide:** [docs/USE_WITH_YOUR_MODEL.md](docs/USE_WITH_YOUR_MODEL.md), with the input format and troubleshooting. It needs the [Python packages](REQUIREMENTS.md#32-python-packages).

**Already running on two very different models** with the same code: a ResNet-18 (a CNN) on CIFAR-10 photos, and a ViT-B/16 (a vision transformer) on product photos. On the CIFAR data, `fit_any` reproduces the results above (AUROC 0.8839 and ECE 0.0236 on unseen blur). See [the proof](#proof-the-same-layer-on-a-second-model).

## Proof: the same layer on a second model

To show the layer isn't tied to one model, we ran the same code on a completely different one: a **ViT-B/16 vision transformer** that recognizes 30 kinds of products, powering an **AI shopping assistant**. The trust decision controls what the assistant may do:

- **TRUST:** one-tap checkout.
- **CAUTION:** it asks which item you meant before buying.
- **REJECT:** checkout locks, and it asks for a better photo.

| | ResNet-18 on CIFAR-10 (unseen blur) | ViT-B/16 on product photos |
|---|---|---|
| TRUST tier: share auto-approved, and share of those that were wrong (target 1%) | 42% at 1.1% | 27% at 0.8% |
| Tells right from wrong (AUROC): model → trust layer | 0.863 → 0.884 | 0.859 → 0.864 |
| Gap between confidence and reality (ECE): model → trust layer | 0.137 → 0.024 | 0.045 → 0.059 |

We also tried it on 26 real phone photos we took. All 7 one-tap purchases it allowed were the right item, and it trusted none of the 4 items the store doesn't sell. An agent that bought whenever it was at least 80% sure would have bought a laptop as a keyboard. Details: [shop/README.md](shop/README.md).

## Repository

| Path | What it does |
|---|---|
| `backend/`, `frontend/` | The demo server and page: standard-library Python and plain HTML/JS, with no internet needed |
| `trust/` | The trust layer: signals, failure model, calibration, cutoffs and metrics; `fit_any.py` fits it to any classifier |
| `benchmark/`, `inference/` | The CIFAR benchmark (412,000 photo versions, split by photo) and the ResNet-18 runs |
| `shop/` | The second use case: the ViT and the shopping assistant. It's optional: delete it and everything else still works |
| `data/` | The precomputed results the demo reads |
| `docs/` | The [guide for your own model](docs/USE_WITH_YOUR_MODEL.md), team plans and pitch notes |

## Data and references

- Data: CIFAR-10 (Krizhevsky, 2009); CIFAR-10-C (Hendrycks & Dietterich, ICLR 2019); CIFAR-10.1 (Recht et al., 2018); ImageNetV2 (Recht et al., 2019).
- Models: pretrained ResNet-18 checkpoint `edadaltocg/resnet18_cifar10` on Hugging Face; torchvision's ViT-B/16 with ImageNet weights.
- Methods: temperature scaling on perturbed data (Tomani et al., CVPR 2021), failure-detection evaluation (Jaeger et al., ICLR 2023), kNN distance on deep features (Sun et al., ICML 2022), Trust Score (Jiang et al., NeurIPS 2018).
