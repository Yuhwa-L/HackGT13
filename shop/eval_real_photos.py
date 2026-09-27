"""Score a folder of real phone photos through the exact live-upload path, and report how the trust gate did.

Filenames carry the truth: "<class>_<condition>.jpg", e.g. water_bottle_dark.jpeg, comp_keyboard_good.jpeg. The class
part must match a product class (spaces as underscores; short aliases below). Files starting with "external" are
items outside the catalog: for those, any TRUST is a failure. The photos themselves are never copied into the repo;
only the results JSON is written.

Run: python -m shop.eval_real_photos ~/Downloads/real_photo_set   -> prints a table, writes data/shop/real_photo_eval.json
"""
import base64
import json
import sys
from pathlib import Path

from shop.config import PRODUCT_CLASSES, shop_path
from shop.live import LiveScorer

ALIASES = {"comp_keyboard": "computer keyboard", "keyboard": "computer keyboard", "phone": "cellular telephone",
           "jeans": "jean", "loafer": "Loafer", "watch": "digital watch", "mug": "coffee mug"}
RAW_CONFIDENT = 0.8  # "a shopping agent that trusts its own confidence" buys when raw confidence >= this


def truth(stem):
    if stem.lower().startswith("external"):
        return None, stem.split("_", 1)[-1] if "_" in stem else "external"
    for cls in sorted(PRODUCT_CLASSES + list(ALIASES), key=len, reverse=True):
        key = cls.lower().replace(" ", "_")
        if stem.lower().startswith(key + "_") or stem.lower() == key:
            return ALIASES.get(cls, cls), stem[len(key) + 1:] or "unlabeled"
    raise ValueError(f"can't tell the class of {stem!r}; name it <class>_<condition> or external<N>")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    folder = Path(sys.argv[1]).expanduser()
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    scorer = LiveScorer()
    scorer.warm_up()
    rows = []
    for f in files:
        cls, cond = truth(f.stem)
        mime = "png" if f.suffix.lower() == ".png" else "webp" if f.suffix.lower() == ".webp" else "jpeg"
        it = scorer.score(f"data:image/{mime};base64," + base64.b64encode(f.read_bytes()).decode())
        cands = [c["class"] for c in it["candidates"]]
        rows.append({"file": f.name, "true_class": cls, "condition": cond, "in_catalog": cls is not None,
                     "pred_class": it["pred_class"], "correct": cls is not None and it["pred_class"] == cls,
                     "raw_confidence": it["raw_confidence"], "p_correct": it["p_correct"], "decision": it["decision"],
                     "true_in_candidates": cls in cands, "quality_issue": it["quality"]["label"]})
    s = summarize(rows)
    shop_path("real_photo_eval.json").write_text(json.dumps({"summary": s, "photos": rows}, indent=1))
    print(f"{'file':34s} {'truth':18s} {'model says':18s} {'raw':>5s} {'p_corr':>6s}  decision  quality")
    for r in rows:
        mark = "ok " if (r["correct"] if r["in_catalog"] else r["decision"] != "trust") else "BAD" if r["decision"] == "trust" else "   "
        print(f"{r['file']:34s} {str(r['true_class'] or '(not a product)'):18s} {r['pred_class']:18s} "
              f"{r['raw_confidence']:5.2f} {r['p_correct']:6.2f}  {r['decision']:8s}  {r['quality_issue'] or '-'}  {mark}")
    print(json.dumps(s, indent=1))


def summarize(rows):
    inc = [r for r in rows if r["in_catalog"]]
    ext = [r for r in rows if not r["in_catalog"]]
    trust = [r for r in inc if r["decision"] == "trust"]
    caution = [r for r in inc if r["decision"] == "caution"]
    naive = [r for r in rows if r["raw_confidence"] >= RAW_CONFIDENT]
    return {
        "photos": len(rows), "in_catalog": len(inc), "outside_catalog": len(ext),
        "model_top1_accuracy_in_catalog": round(sum(r["correct"] for r in inc) / max(len(inc), 1), 3),
        "decisions_in_catalog": {d: sum(r["decision"] == d for r in inc) for d in ("trust", "caution", "reject")},
        "trust_correct": f"{sum(r['correct'] for r in trust)}/{len(trust)}",
        "caution_true_item_offered": f"{sum(r['true_in_candidates'] for r in caution)}/{len(caution)}",
        "outside_catalog_trusted": f"{sum(r['decision'] == 'trust' for r in ext)}/{len(ext)}",
        "wrong_one_tap_purchases": {
            "trust_gate": sum(r["decision"] == "trust" and not r["correct"] for r in rows),
            f"raw_confidence_at_least_{RAW_CONFIDENT}": sum(not r["correct"] for r in naive),
        },
        "one_tap_purchases": {"trust_gate": sum(r["decision"] == "trust" for r in rows),
                              f"raw_confidence_at_least_{RAW_CONFIDENT}": len(naive)},
    }


if __name__ == "__main__":
    main()
