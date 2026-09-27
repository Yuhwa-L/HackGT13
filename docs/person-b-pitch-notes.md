# Pitch notes: benchmark and signals (from B)

For the Devpost and slides. Every number comes from the full run. The **Source** column says where to re-check it:
- `eval` = `data/evaluation.json`
- `ablation` = `data/signal_ablation.json` (`python -m trust.signal_ablation`)
- `features` = the sanity report printed by `python -m trust.features`

## 1. Paste-ready paragraph: the benchmark

> We built a leakage-safe robustness benchmark from CIFAR-10. Each of the 10,000 test photos appears 41 times: once clean and under 8 corruption types (2 each from the noise, blur, weather and digital families) at 5 severities. That makes 410,000 images, plus 2,000 photos from CIFAR-10.1, a separately collected test set with real-world shift. Splits are made **per photo, not per image**: all 41 versions of a photo are in the same split (50% train, 15% validation, 15% calibration, 20% test). So the trust layer is never tested on a blurred copy of a photo it trained on. To test shift the layer has never seen, we hold out one whole corruption family at a time, train on the other three, and rotate through all four. CIFAR-10.1 is never trained on.

## 2. Paste-ready paragraph: the signals

> For every prediction we extract signals that go beyond the softmax. **Confidence:** the top probability, entropy and margin. **Stability:** does the prediction survive a horizontal flip and ±2-pixel shifts? **Familiarity:** how far is the image's 512-d ResNet embedding from the 50,000 clean training images (10th-nearest-neighbour cosine distance, and Mahalanobis distance to the predicted class)? The trust layer never sees the corruption type, family or severity, because a real deployment wouldn't know them.

## 3. What each signal adds (held-out family, test split)

Failure-ranking AUROC. Higher is better; 0.5 is chance.

| Features the model gets | Noise | Blur (headline) | Weather | Digital |
|---|---|---|---|---|
| Raw confidence alone | 0.762 | 0.863 | 0.900 | 0.862 |
| Stability alone (`tta_pconf`) | 0.768 | 0.879 | 0.912 | 0.868 |
| Confidence signals only (model) | 0.768 | 0.865 | 0.900 | 0.867 |
| + familiarity | 0.768 | 0.869 | 0.905 | 0.867 |
| + stability | 0.779 | 0.881 | 0.915 | 0.877 |
| **All 8 (our trust layer)** | **0.778** | **0.884** | **0.917** | **0.877** |

Source: ablation. The last row equals eval → `folds.<family>.heldout_test.methods.trust_layer.auroc`.

**How to say it:**
- **Stability is the signal that matters.** Whether a prediction survives a flip or a 2-pixel shift is the best single signal in all four families, better than the model's own confidence. Adding it to the confidence signals gives most of the gain: blur 0.865 → 0.881.
- **Familiarity adds a little on top** (blur 0.881 → 0.884) and is the "drift alarm" in the story. The average distance to the training images rises with corruption severity (source: features):
  - blur: 0.019 clean → 0.057 at severity 5
  - digital: 0.019 → 0.069
  - CIFAR-10.1: 0.029
- **Be honest about size.** These are modest ranking gains, as the literature predicts (Jaeger et al., ICLR 2023). The big wins are calibration under shift and the kept promise (§4).

## 4. Headline numbers that involve the benchmark

| Claim | Number | Source |
|---|---|---|
| ResNet-18 clean accuracy | 94.98% on all 10,000 test photos | features / A's checkpoint |
| Accuracy collapses under shift (severity 5) | noise 23%, digital 54%, blur 61% | features |
| On unseen blur, raw confidence averages 93% while accuracy is 80% | mean_conf 0.934 vs accuracy 0.797 | eval `folds.blur.heldout_test` |
| The broken promise: a "5% error" rule on raw confidence, set on clean data, gets **20.0%** error on unseen blur | 0.1998 | eval `folds.blur.broken_promise_heldout` |
| The trust layer's 5% rule holds: **4.6%** error at 63.7% coverage | 0.0460 / 0.6366 | same |
| Calibration on unseen blur: ECE 0.137 raw → **0.024** trust layer | | eval `folds.blur.heldout_test.methods` |
| CIFAR-10.1 (never trained on) | ResNet accuracy 88.75% | eval `cifar10_1` |

## 5. Likely judge questions on B's part

- **"Isn't a blurred photo in test just a copy of one in train?"** No. We split by photo, so all 41 versions share one split. It's checked by an assert on every build and by `python -m benchmark.test_benchmark`.
- **"Does it only work on corruptions it trained on?"** Every number above is on a corruption family the layer never saw. We rotate the held-out family (noise, blur, weather, digital) and also test on CIFAR-10.1, which is real-world shift.
- **"Does the model cheat by knowing the corruption type?"** No. Corruption, family and severity are never model inputs; they are only used to group the results.
- **"Which signals actually help?"** Stability first, familiarity a little (§3). We also tried the Trust Score (Jiang et al., 2018). It didn't help: on noise it slightly hurt, with an AUROC change of −0.0015 and a 95% CI of [−0.0021, −0.0010]. We left it out, which is how the project doc says to treat optional signals.
- **"Is it reproducible?"** Everything is seeded. We regenerated the whole pipeline from scratch on a second laptop (an M3). Correctness matched on all 84,000 test rows, and confidences matched to within 5×10⁻¹¹.
- **"Where does the weakness show?"** Under noise, the ResNet is right less than half the time (49.7% on held-out noise test). The trust layer still ranks failures better than raw confidence there, but its probabilities remain overconfident. Say this openly.

## 6. One-line version for a slide

> 412,000 test images, split by photo so nothing leaks; tested on corruption families it never saw. The strongest warning sign is a prediction that flips under a 2-pixel shift.
