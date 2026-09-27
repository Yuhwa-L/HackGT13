"""data/shop/scores.parquet + the photos -> data/shop/shop_cache.json for the shopping-assistant tab.

Picks ~18 photos from the TEST split (the trust layer never trained on them), one per class, with all 25 versions each
(clean + 8 corruptions x severities 1/3/5) as 224 px JPEG data URIs. The story photo is one whose clean version is
TRUST and correct, with a CAUTION version whose candidates include the true class, and a REJECT version.

Run: python -m shop.build_shop_cache   (after python -m shop.run_pipeline)
"""
import base64
import io
import json
from datetime import datetime, timezone

import pandas as pd
from PIL import Image

from shop.config import PRODUCT_CLASSES, shop_path
from shop.corrupt import corrupt
from shop.images import load_photo
from shop.photo_quality import assess, reference
from shop.benchmark_data import list_photos

N_PHOTOS = 18
# Story photo preference: a clean, cheap product (one-tap fits the $150 cap) whose CAUTION step reads naturally
# ("water bottle or perfume?" for a glass bottle). The first class with a full TRUST/CAUTION/REJECT story wins.
STORY_PREF = ["water bottle", "digital watch", "running shoe", "sunglasses", "coffee mug", "backpack", "laptop"]


def jpeg_uri(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def story_parts(g):
    """(trust, caution, reject) rows for one photo's versions, or None."""
    clean = g[g.corruption == "clean"].iloc[0]
    if clean.decision != "trust" or not clean.correct:
        return None
    cands = g[(g.decision == "caution")].copy()
    cands = cands[[len(json.loads(c)) >= 2 and clean.true_class in [x["class"] for x in json.loads(c)] for c in cands.candidates]]
    rej = g[g.decision == "reject"].sort_values("severity", ascending=False)
    if cands.empty or rej.empty:
        return None
    return clean, cands.sort_values("severity").iloc[0], rej.iloc[0]


def main():
    s = pd.read_parquet(shop_path("scores.parquet"))
    test = s[s.split == "test"]
    groups = {b: g for b, g in test.groupby("base_image_id")}
    ok = {b: story_parts(g) for b, g in groups.items()}
    ok = {b: v for b, v in ok.items() if v is not None}
    assert ok, "no test photo has TRUST, CAUTION and REJECT versions; check the pipeline's thresholds"
    story_base = min(ok, key=lambda b: (STORY_PREF.index(groups[b].true_class.iloc[0]) if groups[b].true_class.iloc[0] in STORY_PREF
                                        else len(STORY_PREF), b))
    # One photo per class: prefer full-story photos, then any photo whose clean version is TRUST.
    chosen, seen = [story_base], {groups[story_base].true_class.iloc[0]}
    ranked = sorted(groups, key=lambda b: (b not in ok, not bool((groups[b].corruption == "clean").any() and
                                          groups[b][groups[b].corruption == "clean"].decision.iloc[0] == "trust"), b))
    for b in ranked:
        c = groups[b].true_class.iloc[0]
        if len(chosen) < N_PHOTOS and c not in seen:
            chosen.append(b)
            seen.add(c)

    paths = {f"prod_{r.md5[:10]}": r.path for r in list_photos().itertuples()}
    # Photo-quality reference: clean TRAIN photos only (never the test photos shown in the demo)
    train_bases = s[(s.split == "train") & (s.corruption == "clean")].base_image_id
    qref = reference([load_photo(paths[b]) for b in train_bases])
    shop_path("quality_ref.json").write_text(json.dumps(qref, indent=1))
    photos = []
    for b in chosen:
        g = groups[b]
        img = load_photo(paths[b])
        versions = []
        for r in g.sort_values(["corruption", "severity"], key=lambda col: col.map(lambda x: (x != "clean", x)) if col.name == "corruption" else col).itertuples():
            version = corrupt(img, r.corruption, r.severity, b)
            q = assess(version, qref)
            versions.append({
                "photo_id": r.sample_id, "base_image_id": b, "true_class": r.true_class, "corruption": r.corruption,
                "family": r.family, "severity": int(r.severity), "pred_class": r.pred_class, "correct": bool(r.correct),
                "raw_confidence": round(float(r.raw_confidence), 4), "temp_clean": round(float(r.temp_clean), 4),
                "temp_corrupted": round(float(r.temp_corrupted), 4), "p_correct": round(float(r.p_correct), 4),
                "decision": r.decision, "reasons": json.loads(r.reasons), "candidates": json.loads(r.candidates),
                "quality": {k: q[k] for k in ("issue", "label", "tip")}, "image": jpeg_uri(version)})
        photos.append({"base_image_id": b, "true_class": g.true_class.iloc[0], "versions": versions})

    t, c, r = ok[story_base]
    story = {k: {"base_image_id": story_base, "photo_id": row.sample_id} for k, row in (("trust", t), ("caution", c), ("reject", r))}
    cache = {"meta": {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "photos": len(photos),
                      "classes": len(PRODUCT_CLASSES), "source": "ImageNetV2 test-split photos; fictional catalog"},
             "story": story, "photos": photos}
    out = shop_path("shop_cache.json")
    out.write_text(json.dumps(cache))
    print(f"wrote {out}: {len(photos)} photos x {len(photos[0]['versions'])} versions, "
          f"{out.stat().st_size / 1e6:.1f} MB; story photo {story_base} ({groups[story_base].true_class.iloc[0]})")


if __name__ == "__main__":
    main()
