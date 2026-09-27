"""Live scoring of an uploaded photo: the same model, signals and trust layer as shop/run_pipeline.py, for one image.

The ViT-B/16 runs in a child process (shop/live_worker.py) so torch and XGBoost never share a process (their OpenMP
runtimes crash each other on macOS). Loads in the background at server start from committed artifacts:
models/xgb.json, models/isotonic.json, thresholds.json, live_reference.npz, quality_ref.json, plus the ViT-B/16
weights in data/shop/raw/ (see shop/README.md) and torch. Without those, uploads are disabled with a clear message;
the rest of the tab still works.
"""
import base64
import hashlib
import io
import json
import subprocess
import sys
import threading

import numpy as np
import xgboost as xgb
import pandas as pd
from PIL import Image, ImageOps

from shop import live_worker
from shop.config import PRODUCT_CLASSES, shop_path
from shop.images import prepare
from shop.photo_quality import assess, load_reference
from trust import signals
from trust.calibrate import apply_isotonic
from trust.temperature_scaling import scaled_confidence, softmax
from trust.thresholds import decide
from trust.train_failure_model import FEATURES, p_correct_score, reasons

MAX_PIXELS = 40_000_000  # refuse decompression bombs
N = len(PRODUCT_CLASSES)


class LiveScorer:
    def __init__(self):
        self._lock, self._ready, self.error = threading.Lock(), threading.Event(), None

    def warm_up(self):
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            from shop.config import ROOT
            self.proc = subprocess.Popen([sys.executable, "-W", "ignore", "-m", "shop.live_worker"], cwd=ROOT,
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            state, info = self._recv()  # blocks until the worker is ready (or reports / dies)
            if state != "ready":
                raise RuntimeError(info)
            self.dev = info
            z = np.load(shop_path("live_reference.npz"))
            assert list(z["features"]) == FEATURES, "live_reference.npz is stale; rerun python -m shop.run_pipeline"
            self.bank, self.bank_y = z["bank_emb"], z["bank_labels"]
            self.reference = pd.DataFrame(z["reference"], columns=FEATURES)
            self.means, self.prec = signals.fit_mahalanobis(self.bank, self.bank_y, n_classes=N)
            self.xgb = xgb.XGBClassifier()
            self.xgb.load_model(shop_path("models", "xgb.json"))
            self.iso = json.loads(shop_path("models", "isotonic.json").read_text())
            self.th = json.loads(shop_path("thresholds.json").read_text())
            self.qref = load_reference()
        except Exception as e:  # torch missing, weights missing, stale artifacts: uploads off, demo unaffected
            self.error = f"{type(e).__name__}: {e}"
            print(f"Snap-to-Shop live upload disabled ({self.error})")
        finally:
            self._ready.set()

    def _recv(self):
        try:
            return live_worker.recv(self.proc.stdout)
        except EOFError:
            raise RuntimeError(f"the model process exited (code {self.proc.wait()})") from None

    def status(self):
        if not self._ready.is_set():
            return "loading"
        return "error" if self.error else "ready"

    def score(self, data_url):
        """A data: URL (JPEG/PNG/WebP) -> a shop_cache-style item. Raises ValueError on a bad image."""
        if not self._ready.wait(timeout=60) or self.error:
            raise RuntimeError(self.error or "the live model is still loading")
        raw = _decode(data_url)
        try:
            Image.MAX_IMAGE_PIXELS = MAX_PIXELS
            im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
            img = prepare(im)
        except Exception as e:
            raise ValueError(f"couldn't read that image ({type(e).__name__}); try a JPEG or PNG") from e
        with self._lock:  # one request at a time over the pipe
            try:
                live_worker.send(self.proc.stdin, img)
                state, out = self._recv()
            except (EOFError, BrokenPipeError, OSError) as e:
                self.error = f"the model process stopped ({type(e).__name__}); restart the server"
                raise RuntimeError(self.error) from e
        if state != "ok":
            raise RuntimeError(out)
        Z, tta, emb = out
        sm = signals.softmax_stats(Z)
        pred = sm["pred"]
        X = pd.DataFrame({**{k: sm[k] for k in ("raw_confidence", "entropy", "margin")}, **signals.tta_stats(Z, tta),
                          **signals.bank_signals(emb, pred, self.bank, self.bank_y, n_classes=N),
                          "maha_pred": signals.maha_pred(emb, pred, self.means, self.prec)})
        score = p_correct_score(self.xgb, X)
        p = float(apply_isotonic(self.iso, score)[0])
        decision = str(decide(np.array([p]), self.th["tau_reject"], self.th["tau_trust"])[0])
        probs = softmax(Z.astype(np.float64))[0]
        top = np.argsort(-probs)[:3]
        cands = [{"class": PRODUCT_CLASSES[k], "prob": round(float(probs[k]), 4)}
                 for j, k in enumerate(top) if j == 0 or probs[k] >= 0.05]
        q = assess(img, self.qref)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=85)
        pid = "upload-" + hashlib.sha256(raw).hexdigest()[:12]
        return {"photo_id": pid, "base_image_id": pid, "true_class": None, "corruption": "upload", "family": "upload",
                "severity": 0, "pred_class": PRODUCT_CLASSES[int(pred[0])], "correct": None,
                "raw_confidence": round(float(sm["raw_confidence"][0]), 4),
                "temp_clean": round(float(scaled_confidence(Z, self.th["T_clean"])[0]), 4),
                "temp_corrupted": round(float(scaled_confidence(Z, self.th["T_corrupted"])[0]), 4),
                "p_correct": round(p, 4), "decision": decision,
                "reasons": json.loads(reasons(self.xgb, X, np.array([decision]), self.reference)[0]),
                "candidates": cands, "quality": {k: q[k] for k in ("issue", "label", "tip")},
                "image": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}


def _decode(data_url):
    if not isinstance(data_url, str) or not data_url.startswith("data:image/") or ";base64," not in data_url:
        raise ValueError("send the photo as a data:image/...;base64 URL")
    try:
        return base64.b64decode(data_url.split(",", 1)[1], validate=True)
    except Exception as e:
        raise ValueError("the image data isn't valid base64") from e
