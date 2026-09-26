"""Owner: D. Build data/demo_cache.json, everything the demo page shows, precomputed so the demo cannot fail at the expo.

Picks story images from the test split (the trust layer never trained on them): per class, the base images whose
clean version is TRUST, whose moderate corruption is still right but CAUTION, and whose heavy corruption is
confidently wrong and REJECT. Adds a few test photos at random (curated=false) so the demo can show it is not cherry-picked. For each it stores all 41 versions (clean + 8 corruptions x 5 severities) in the
/predict response format of the project doc (section 11), plus the 32x32 pixels as a PNG data URI.
Also stores per-family curves (accuracy, confidence, p_correct, decision shares by severity) for the drift chart.

Reads data/scores.parquet + data/evaluation.json (C), data/manifest.csv (B), pixels via benchmark.load_cifar10c (data/raw).
Run: python -m backend.build_demo_cache [--per-class 3]
"""
import argparse
import base64
import json
import struct
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
FAMILIES = ["noise", "blur", "weather", "digital"]
N_VERSIONS = 41          # clean + 8 corruptions x 5 severities
CONFIDENT = 0.8          # "confidently wrong" = raw confidence at least this


def png_data_uri(img):
    """(H, W, 3) uint8 -> data:image/png;base64 URI, standard library only. The page upscales it pixelated."""
    h, w, _ = img.shape
    chunk = lambda tag, data: struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    rows = b"".join(b"\x00" + np.ascontiguousarray(img[y], dtype=np.uint8).tobytes() for y in range(h))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def response(r):
    """One scores.parquet row -> the section-11 /predict response (plus truth and context for the page)."""
    f = lambda v: round(float(v), 4)
    return {
        "target_model": {"prediction": r.pred_class, "raw_confidence": f(r.raw_confidence), "entropy": f(r.entropy),
                         "margin": f(r.margin)},
        "baselines": {"temp_scaled_clean": f(r.temp_clean), "temp_scaled_corrupted": f(r.temp_corrupted)},
        "trust_layer": {"p_correct": f(r.p_correct), "decision": r.decision, "reasons": json.loads(r.reasons)},
        "truth": {"true_class": r.true_class, "correct": bool(r.correct)},
        "context": {"corruption": r.corruption, "family": r.family, "severity": int(r.severity), "fold": r.fold},
    }


def find_stories(rows):
    """(base_image_id, corruption) pairs that play the clean -> CAUTION -> REJECT story, with a drama score."""
    clean = rows[rows.corruption == "clean"].set_index("base_image_id")
    good = set(clean.index[(clean.decision == "trust") & (clean.correct == 1)])
    out = []
    for (base, corr), g in rows[(rows.corruption != "clean") & rows.base_image_id.isin(good)].groupby(["base_image_id", "corruption"]):
        g = g.set_index("severity")
        mid = g.loc[g.index.isin([2, 3]) & (g.decision == "caution") & (g.correct == 1)]
        hi = g.loc[g.index.isin([4, 5]) & (g.decision == "reject") & (g.correct == 0) & (g.raw_confidence >= CONFIDENT)]
        if len(mid) and len(hi):
            h = hi.index.max()
            out.append(dict(base=base, corruption=corr, moderate=int(mid.index.min()), heavy=int(h),
                            true_class=clean.loc[base, "true_class"], drama=float(hi.raw_confidence.loc[h] - hi.p_correct.loc[h])))
    return pd.DataFrame(out, columns=["base", "corruption", "moderate", "heavy", "true_class", "drama"])


def family_curves(rows):
    """Per held-out family: severity 0 (clean test images) to 5 -> accuracy, mean confidences, decision shares."""
    agg = lambda g: dict(n=int(len(g)), accuracy=float(g.correct.mean()), raw_confidence=float(g.raw_confidence.mean()),
                         p_correct=float(g.p_correct.mean()),
                         **{d: float((g.decision == d).mean()) for d in ("trust", "caution", "reject")})
    clean = agg(rows[rows.corruption == "clean"])
    return {fam: [dict(severity=0, **clean)] + [dict(severity=int(s), **agg(g)) for s, g in
                                              rows[rows.family == fam].groupby("severity")]
            for fam in FAMILIES if (rows.family == fam).any()}


def build(scores, manifest, evaluation, load_pixels, per_class=3, n_random=10):
    rows = scores[scores.dataset != "cifar10_1"]
    complete = rows.groupby("base_image_id").size()
    rows = rows[rows.base_image_id.isin(complete.index[complete == N_VERSIONS])]
    stories = find_stories(rows).sort_values("drama", ascending=False)
    best = stories.drop_duplicates("base")                    # each image's most dramatic corruption
    picked = best.groupby("true_class").head(per_class)
    # classes without enough story images are filled with other test images whose clean version is TRUST
    short = {c: per_class - n for c, n in picked.true_class.value_counts().reindex(sorted(rows.true_class.unique()), fill_value=0).items() if n < per_class}
    fill = []
    for c, k in short.items():
        cand = rows[(rows.corruption == "clean") & (rows.true_class == c) & ~rows.base_image_id.isin(picked.base)]
        fill += list(cand.sort_values(["decision", "p_correct"], ascending=[False, False]).base_image_id[:k])
    ids = list(picked.base) + fill
    # plus a few test photos nobody picked, so a judge can check the demo is not cherry-picked
    others = sorted(set(rows.base_image_id) - set(ids))
    random_ids = [str(b) for b in np.random.default_rng(0).choice(others, size=min(n_random, len(others)), replace=False)]
    curated = set(ids)
    ids += random_ids

    sel = rows[rows.base_image_id.isin(ids)].merge(manifest[["sample_id", "image_index"]], on="sample_id", validate="1:1")
    pixels = load_pixels(sel)
    sel = sel.assign(image=[png_data_uri(p) for p in pixels])
    story_of = picked.set_index("base")
    images = []
    for base in ids:
        g, versions = sel[sel.base_image_id == base], {}
        for r in g.itertuples():
            versions.setdefault(r.corruption, {})[str(int(r.severity))] = dict(response(r), image=r.image)
        st = story_of.loc[base] if base in story_of.index else None
        images.append(dict(image_id=base, true_class=g.true_class.iloc[0], versions=versions, curated=base in curated,
                           story=None if st is None else dict(corruption=st.corruption, moderate=int(st.moderate), heavy=int(st.heavy))))
    images.sort(key=lambda im: (not im["curated"], im["true_class"], im["image_id"]))

    blur = picked[picked.corruption.isin(["defocus_blur", "motion_blur"])]   # the headline family tells the story best
    top = (blur if len(blur) else picked).iloc[0] if len(picked) else None
    corruptions = rows[rows.corruption != "clean"].drop_duplicates("corruption").set_index("corruption").family
    meta = evaluation.get("meta", {})
    return {
        "meta": dict(created=datetime.now(timezone.utc).isoformat(timespec="seconds"), n_images=len(images), n_random=len(random_ids),
                     source_rows=meta.get("n_rows"), source_created=meta.get("created"), base_images=meta.get("base_images"),
                     headline_fold=evaluation.get("headline_fold"),
                     thresholds={F: fo["thresholds"] for F, fo in evaluation.get("folds", {}).items()}),
        "corruptions": {c: corruptions[c] for fam in FAMILIES for c in corruptions.index if corruptions[c] == fam},
        "story": None if top is None else dict(image_id=top.base, corruption=top.corruption,
                                               beats=[dict(label="Clean", severity=0), dict(label="Moderate", severity=int(top.moderate)),
                                                      dict(label="Heavy", severity=int(top.heavy))]),
        "images": images,
        "family_curves": family_curves(rows),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--random", type=int, default=10, help="extra test photos picked at random (not curated)")
    args = ap.parse_args()
    from benchmark.load_cifar10c import load_images   # B's single place that turns manifest rows into pixels

    d = args.data_dir
    cache = build(pd.read_parquet(d / "scores.parquet"), pd.read_csv(d / "manifest.csv"),
                  json.loads((d / "evaluation.json").read_text()), load_images, args.per_class, args.random)
    out = d / "demo_cache.json"
    out.write_text(json.dumps(cache, separators=(",", ":"), allow_nan=False))
    st = cache["story"]
    print(f"wrote {out}: {cache['meta']['n_images']} images ({cache['meta']['n_random']} random) x {N_VERSIONS} versions, {out.stat().st_size / 1e6:.1f} MB")
    if st:
        print(f"story: {st['image_id']} ({next(i['true_class'] for i in cache['images'] if i['image_id'] == st['image_id'])}), "
              f"{st['corruption']} severities {[b['severity'] for b in st['beats']]}")


if __name__ == "__main__":
    main()
