# Use Trust Issues with your own model

Trust Issues fits on top of any classifier you already have. You export the model's outputs on labeled examples, and
`trust.fit_any` fits a trust layer to them. Your classifier is never retrained. For every new prediction, the layer
returns:

- `p_correct`: a calibrated probability that the prediction is right;
- a decision: TRUST, CAUTION or REJECT;
- plain-language reasons.

This guide goes from a trained model to scored predictions in six steps.

## 1. What you need

- **A trained classifier** that gives one score per class. Any framework and any architecture work: a CNN, a
  transformer, a gradient-boosted model, a logistic regression.
- **Labeled examples the model hasn't trained on.** The layer learns what your model's mistakes look like, so it needs
  mistakes to learn from.
  - Aim for a few thousand examples: about 1,000 or more end up in the cutoff-setting split, and about 100 or more wrong
    predictions in the training split.
  - Ideally, include the conditions you expect in real use: bad lighting, blur, new data sources, other microphones.
    Conditions it never sees are the hardest ones to warn about.
- **Python 3.9+** with this repository's `requirements.txt` (numpy, pandas, scikit-learn, xgboost, scipy). Fitting
  needs no GPU. On macOS, XGBoost also needs `brew install libomp`.

## 2. Export your model's outputs

You need up to three things per example:

| What | Shape | Required | Turns on |
|---|---|---|---|
| Logits: the raw scores before softmax | N × K | yes | confidence signals |
| TTA logits: logits on V small changes that shouldn't change the answer | N × V × K | no | stability signals |
| Embeddings: the vector the final layer reads (the second-to-last layer) | N × D | no | familiarity signals |

Familiarity also needs reference embeddings and labels: the same vectors for clean training examples (M × D, and M
labels). A few thousand examples covering every class is enough.

### PyTorch

```python
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

model.eval()
head = model.fc                       # the final nn.Linear: model.fc (torchvision ResNet), model.heads.head (ViT), ...
captured = {}
head.register_forward_hook(lambda mod, inputs, out: captured.update(emb=inputs[0].detach()))


def small_changes(x):                 # V versions that shouldn't change the answer: a flip and ±2 px shifts (our CIFAR setup)
    w, padded = x.shape[-1], F.pad(x, (2, 2, 0, 0), mode="reflect")
    return [x.flip(-1), padded[..., :w], padded[..., 4:4 + w]]


@torch.no_grad()
def export(loader, views=True):       # loader yields (inputs, labels) in a fixed order: shuffle=False
    Z, T, E, Y = [], [], [], []
    for x, y in loader:
        x = x.to(device)
        Z.append(model(x).cpu())
        E.append(captured["emb"].cpu())   # read it before the extra passes below overwrite it
        if views:
            T.append(torch.stack([model(v) for v in small_changes(x)], dim=1).cpu())
        Y.append(y)
    return [torch.cat(a).numpy() if a else None for a in (Z, T, E, Y)]


logits, tta, emb, labels = export(eval_loader)                 # your labeled examples
meta = pd.read_csv("eval_meta.csv")   # one row per example, in loader order: sample_id, group_id (and condition)
df = meta.assign(label=labels, **{f"logit_{k}": logits[:, k] for k in range(logits.shape[1])})
df.to_csv("my_outputs/predictions.csv", index=False)
np.save("my_outputs/tta_logits.npy", tta)
np.save("my_outputs/embeddings.npy", emb)

ref_logits, _, ref_emb, ref_labels = export(clean_train_loader, views=False)   # e.g. 10,000 clean training examples
np.save("my_outputs/reference_embeddings.npy", ref_emb)
np.save("my_outputs/reference_labels.npy", ref_labels)
```

For a `timm` model, `model.forward_head(model.forward_features(x), pre_logits=True)` gives the same embedding without
a hook.

### scikit-learn (and anything with `predict_proba`)

Use log probabilities as the logits. Softmax turns them back into the same probabilities.

```python
import numpy as np
import pandas as pd

proba = clf.predict_proba(X_eval)                  # N x K, columns in clf.classes_ order
logits = np.log(np.clip(proba, 1e-12, 1))          # clip: a probability of exactly 0 would give -inf
labels = np.searchsorted(clf.classes_, y_eval)     # the true class as an index into clf.classes_
df = pd.DataFrame({"sample_id": ids, "group_id": groups, "label": labels,
                   **{f"logit_{k}": logits[:, k] for k in range(logits.shape[1])}})
df.to_csv("my_outputs/predictions.csv", index=False)
with open("my_outputs/classes.txt", "w") as f:
    f.write("\n".join(map(str, clf.classes_)))
```

Logits alone already give you a calibrated `p_correct` and cutoffs. For stability signals, run `predict_proba` on your
own small changes and stack the log probabilities into an N × V × K array.

### "Small changes" for other kinds of data

A small change is one a person would say doesn't change the right answer. Use the same V changes for every example, and
again when you score new data.

| Data | Small changes |
|---|---|
| Images | a horizontal flip; shifts of a few pixels |
| Text | a paraphrase; swapping a word for a synonym; dropping a filler word |
| Audio | light background noise; a time shift of a few milliseconds; a ±1 dB volume change |
| Tables | tiny noise on measured numbers, within their measurement error |

## 3. The input folder

Put these files in one folder. Rows of every `.npy` file must follow `predictions.csv` row order.

| File | Required | Shape | Contents |
|---|---|---|---|
| `predictions.csv` | yes | N rows | one row per prediction (columns below) |
| `classes.txt` | no | K lines | class names, one per line, in class order |
| `tta_logits.npy` | no | N × V × K | logits on the V small changes |
| `embeddings.npy` | no | N × D | second-to-last-layer vectors |
| `reference_embeddings.npy` | with `embeddings.npy` | M × D | the same vectors for clean training examples |
| `reference_labels.npy` | with `embeddings.npy` | M | their true classes, 0..K-1; every class must appear |

Columns of `predictions.csv`:

| Column | Required | Meaning |
|---|---|---|
| `sample_id` | yes | a unique id for the row |
| `group_id` | yes | the original item. Every version of one item (a clean photo and its blurred copies, the same sentence paraphrased) shares a `group_id`, so all of them land in the same split and nothing leaks from training to testing. |
| `label` | yes | the true class, 0..K-1, in logit column order |
| `logit_0` … `logit_{K-1}` | yes | the model's raw scores before softmax, for any K ≥ 2 |
| `split` | no | `train`, `val`, `cal` or `test`. If it's missing, rows are split by `group_id` into 50/15/15/20, stratified by label, with a fixed seed. |
| `condition` | no | `clean`, or the name of the shift or corruption. Clean rows get 25% of the training weight, and the report breaks results down by condition. |

An example with K = 3:

```csv
sample_id,group_id,label,logit_0,logit_1,logit_2,split,condition
photo_0001,item_017,2,-1.20,0.35,3.10,train,clean
photo_0001_dark,item_017,2,-0.42,0.91,1.18,train,low_light
photo_0002,item_018,0,2.75,-0.10,-1.60,test,clean
```

What each split is for: `train` fits the failure model, `val` calibrates `p_correct`, `cal` sets the TRUST and REJECT
cutoffs, and `test` is used only for the report. To measure how the layer handles a condition it has never seen, keep
that condition out of `train`, `val` and `cal`, and put it only in `test`.

## 4. Fit the layer

```bash
python -m trust.fit_any --data my_outputs --out my_trust_layer
```

It checks the folder first and stops with a message that names the file, the line and the fix. Then it fits and prints
a report on the test split. Here is the report for the CIFAR-10 ResNet-18 run, with blur kept out of training (the
conversion script is `trust/cifar_to_fit_any.py`):

```
332,000 predictions, 10 classes; signals: confidence, stability (3 views), familiarity (50,000 reference rows)
cutoffs from cal: REJECT below 0.742, TRUST at or above 0.972

Test split: 84,000 predictions, 75.1% of them right.
                                                      raw confidence   trust layer
Tells right from wrong (AUROC, higher = better)                0.864         0.886
Stated confidence vs reality (ECE, lower = better)             0.174         0.013
TRUST: share of test, share of those wrong               25.0%, 1.0%   39.9%, 1.2%
CAUTION: share of test, share of those wrong             31.3%, 8.4%  19.1%, 12.3%
REJECT: share of test, share of those wrong             43.6%, 50.4%  41.0%, 53.7%
(both use cutoffs set on cal: TRUST at most 1% wrong, TRUST + CAUTION at most 5% wrong)

condition            rows   right   AUROC raw/trust   ECE raw/trust    TRUST (wrong)   REJECT
clean               2,000   95.2%     0.915 / 0.923   0.033 / 0.039     70.0% (0.4%)    14.5%
noise              20,000   49.7%     0.762 / 0.790   0.373 / 0.083     14.1% (4.0%)    70.3%
blur               20,000   79.7%     0.863 / 0.884   0.137 / 0.024     42.0% (1.1%)    36.3%
weather            20,000   90.3%     0.900 / 0.916   0.063 / 0.030     60.3% (0.6%)    21.1%
digital            20,000   77.5%     0.862 / 0.877   0.150 / 0.029     39.1% (1.5%)    40.3%
cifar10_1           2,000   88.8%     0.884 / 0.896   0.069 / 0.049     51.7% (1.1%)    26.9%
```

How to read it:

- **AUROC: how well it tells right answers from wrong ones.** 1 is perfect, and 0.5 is guessing. The trust layer's
  column should beat raw confidence.
- **ECE: how far stated confidence is from reality.** 0 is perfect. If your model is already well calibrated, the trust
  layer may not beat it here. Our ViT-B/16 was one such model.
- **The decision rows** give each decision's share of the test set, and how many of those answers were wrong. Raw
  confidence gets the same rule, with its own cutoffs set on the same `cal` rows, so the two columns compare fairly.
  TRUST aims for at most 1% wrong. TRUST and CAUTION together aim for at most 5% wrong.
- **The condition rows** show where the layer works and where it doesn't. The cutoffs hold on average over the kind of
  data in `cal`, not for every condition. Above, noise (the hardest condition) ran its TRUST tier at 4.0% wrong. If one
  condition matters to you, check its row and give it enough `cal` rows.
- **Warnings** flag thin data: fewer than about 1,000 rows in `cal`, fewer than about 100 wrong predictions in `train`,
  or a target that no cutoff could meet. The layer still fits, but treat its numbers with care.

What to expect from each signal, measured on CIFAR blur (a kind of damage it never saw):

| Signals you export | Tells right from wrong (AUROC) | Stated vs reality (ECE) |
|---|---|---|
| none: raw confidence | 0.863 | 0.137 |
| logits only | 0.865 | 0.027 |
| logits + TTA logits | 0.881 | 0.025 |
| logits + TTA logits + embeddings | 0.884 | 0.024 |

With logits alone, the layer mostly fixes calibration and sets safe cutoffs. Telling right from wrong improves when you
add the stability and familiarity signals.

The layer folder holds everything that scoring needs:

| File | Contents |
|---|---|
| `xgb.json` | the failure model |
| `isotonic.json` | the calibration map that turns its score into `p_correct` |
| `trust_config.json` | the features used, K, the class names, the cutoffs (`tau_reject`, `tau_trust`), the targets and the split seed |
| `reference.npz` | the reference embeddings, and the correct training rows the reasons compare against |
| `report.json` | the numbers above, and any warnings |

Fitting takes seconds: 3 s for the 332,000 CIFAR predictions from logits and TTA logits. Computing the familiarity
signals adds a nearest-neighbor search, which took about 80 s on a laptop against 50,000 reference embeddings.

## 5. Score new predictions

Export new predictions the same way. They need `sample_id` and the logit columns, plus `tta_logits.npy` and
`embeddings.npy` if the layer uses them, with the same V and D as before. They need no labels and no reference files,
because the layer carries its own.

```bash
python -m trust.fit_any --score my_trust_layer --data new_outputs --out scores.csv
```

From Python, with numpy arrays:

```python
from trust.fit_any import load_layer, score

layer = load_layer("my_trust_layer")                  # load once
out = score(layer, logits, tta_logits, embeddings)    # per batch; pass only what the layer was fit with
```

Each row of the output has `pred` (the class index), `pred_class` (if you gave `classes.txt`), `raw_confidence`,
`p_correct`, `decision` and `reasons`. The reasons are a JSON list like
`[{"signal": "stability", "text": "Prediction changed under 2 of 3 small input changes", "shap": 1.64}]`.

| Decision | What it means | What to do |
|---|---|---|
| TRUST | `p_correct` is at or above `tau_trust`. On `cal`, answers at this level were wrong at most 1% of the time. | Act on it automatically. |
| CAUTION | Between the two cutoffs. Together with TRUST, it stayed at most 5% wrong on `cal`. | Use it, but flag it for review or ask a person to confirm. |
| REJECT | `p_correct` is below `tau_reject`. | Don't act on it. Send it to a person or a stronger model, or ask for a better input. |

Two more habits:

- **Watch the REJECT rate over time.** When it climbs, your inputs have probably changed (a dirty lens, a new data
  source), often before anyone measures the accuracy drop.
- **Refit when the model changes.** The layer learns one model's mistakes. After you retrain or replace the classifier,
  export its outputs again and refit. Fitting takes seconds.

## 6. Troubleshooting

| Message or symptom | Likely cause | Fix |
|---|---|---|
| `logit columns must be numbered logit_0 .. logit_{K-1} with no gaps` | a logit column is missing or misnamed | write one column per class, numbered from 0 |
| `tta_logits.npy must be N x V x K ...` or `embeddings.npy must be N x D ...` | the array's rows or shape don't match `predictions.csv` | save one row per prediction, in `predictions.csv` row order |
| `group_id '...' (line N) has rows in more than one split` | versions of one item sit in different splits, so testing leaks into training | give every version of an item the same `split`, or drop the `split` column |
| `label must be a class index from 0 to K-1` | labels are names, or start at 1 | map each label to its logit column's index |
| `row(s) have a logit that is empty, not a number or infinite` | NaN or infinity, often `log(0)` from `predict_proba` | clip probabilities before the log (section 2) |
| `sample_id must be unique` | a duplicated id | make every row's id unique, for example by adding the condition to it |
| `the train split needs both right and wrong predictions` | the model never errs on `train`, or always does | add harder examples; the layer learns from mistakes |
| warning: `only N wrong predictions in train` | too few mistakes to learn from | add data from the conditions where your model struggles |
| warning: `only N rows in cal` | the cutoffs will be noisy | add data, or give `cal` more groups through the `split` column |
| warning: `no cutoff kept TRUST at or under 1% wrong on cal` | even the most confident answers were wrong more than 1% of the time on `cal` | add `cal` data; until then nothing gets TRUST, and CAUTION is the top tier |
| `reference_labels.npy must include every class` | the reference set is missing some classes | add clean training examples of every class |
| `embeddings.npy needs reference_embeddings.npy and reference_labels.npy` | the reference files are missing | export embeddings for clean training data too (section 2) |
| when scoring: `this layer uses stability signals: pass tta_logits` | the layer was fit with TTA logits | export the same V small changes for the new data |
| when scoring: `logits must be N x K (the layer's classes)` | a different model, or a different class list | score with a layer fit on this model's outputs |
| `XGBoost Library (libxgboost.dylib) could not be loaded` | macOS without OpenMP | `brew install libomp` |
