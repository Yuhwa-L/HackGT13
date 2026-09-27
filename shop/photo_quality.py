"""Photo-quality check for the retake coach: is the photo blurry, noisy, too dark, washed out or hazy?

Four cheap image statistics on the 224 px grayscale photo, compared against percentiles of clean benchmark photos
(data/shop/quality_ref.json, written by build_shop_cache). It looks only at the pixels, so it works on uploads too.
"""
import json

import numpy as np
from scipy.ndimage import laplace, median_filter

from shop.config import shop_path

ISSUES = {  # issue -> (label for sentences, retake tip)
    "blurry": ("blurry", "Hold the phone steady, tap the item to focus, and step back a little if you're too close."),
    "noisy": ("grainy", "Add light: grain comes from dim scenes. Move near a window or turn on a lamp."),
    "dark": ("too dark", "Turn on a light or move near a window, and avoid shooting into shadow."),
    "bright": ("washed out", "Avoid direct sun or flash glare; shoot in even, indirect light."),
    "hazy": ("hazy or low-contrast", "Wipe the lens and shoot the item against a plain, contrasting background."),
}
GENERIC_TIP = "Fill the frame with just the item, on a plain background, in even light."
METRICS = ("brightness", "contrast", "sharpness", "noise")


def metrics(img):
    g = np.asarray(img, np.float64) @ np.array([0.299, 0.587, 0.114])
    return {"brightness": float(g.mean()), "contrast": float(g.std()),
            "sharpness": float(np.log(laplace(g).var() / max(g.var(), 1e-6) + 1e-6)),  # contrast-invariant
            "noise": float(np.median(np.abs(g - median_filter(g, size=3))) * 1.4826)}


def reference(images):
    """Percentiles of each metric over clean photos."""
    m = np.array([[metrics(im)[k] for k in METRICS] for im in images])
    return {k: {q: float(np.percentile(m[:, j], int(q[1:]))) for q in ("p5", "p50", "p95")} for j, k in enumerate(METRICS)}


def load_reference():
    return json.loads(shop_path("quality_ref.json").read_text())


def assess(img, ref):
    """{"issue", "label", "tip", "metrics"}: the most out-of-range issue vs clean photos, or issue None."""
    m = metrics(img)
    r = {k: ref[k] for k in METRICS}

    def below(k):  # how far under the clean 5th percentile, in units of (median - p5)
        return (r[k]["p5"] - m[k]) / max(r[k]["p50"] - r[k]["p5"], 1e-9)

    def above(k):
        return (m[k] - r[k]["p95"]) / max(r[k]["p95"] - r[k]["p50"], 1e-9)

    scores = {"blurry": below("sharpness"), "noisy": above("noise"), "dark": below("brightness"),
              "bright": above("brightness"), "hazy": below("contrast")}
    issue, score = max(scores.items(), key=lambda kv: kv[1])
    if score <= 0:
        return {"issue": None, "label": None, "tip": GENERIC_TIP, "metrics": m}
    label, tip = ISSUES[issue]
    return {"issue": issue, "label": label, "tip": tip, "metrics": m}
