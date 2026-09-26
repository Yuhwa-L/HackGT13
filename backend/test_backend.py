"""Owner: D. Self-checks for the demo cache builder and the server. Run from the repo root: python -m backend.test_backend"""
import base64
import json
import struct
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from backend.build_demo_cache import CONFIDENT, N_VERSIONS, build, png_data_uri
from backend.main import make_server
from trust.run_trust import run
from trust.test_trust import write_mock

SECTION_11 = {"target_model", "baselines", "trust_layer"}


def fake_pixels(rows):   # stands in for benchmark.load_cifar10c.load_images: deterministic per image_index
    return np.stack([np.random.default_rng(int(i)).integers(0, 256, (32, 32, 3), dtype=np.uint8) for i in rows.image_index])


def plant_story(scores, base, corruption, raw_heavy):
    """The synthetic mock is never confidently wrong, so it has no demo story; write one into a test image's rows."""
    m = scores.base_image_id == base
    scores.loc[m & (scores.corruption == "clean"), ["decision", "correct"]] = ["trust", 1]
    scores.loc[m & (scores.corruption == corruption) & (scores.severity == 2), ["decision", "correct"]] = ["caution", 1]
    scores.loc[m & (scores.corruption == corruption) & (scores.severity == 5),
               ["decision", "correct", "raw_confidence", "p_correct"]] = ["reject", 0, raw_heavy, 0.05]


def make_cache(tmp):
    """Mock inputs -> the real trust layer -> the real cache builder (only the pixels are fake)."""
    d = Path(tmp)
    man = write_mock(d, n_base=400, n_c101=40)
    man["image_index"] = np.arange(len(man))
    man.to_csv(d / "manifest.csv", index=False)
    run(d)
    scores, ev = pd.read_parquet(d / "scores.parquet"), json.loads((d / "evaluation.json").read_text())
    tests = sorted(set(scores.base_image_id[scores.dataset != "cifar10_1"]))
    plant_story(scores, tests[0], "defocus_blur", 0.90)
    plant_story(scores, tests[1], "gaussian_noise", 0.99)    # more dramatic, but blur is the headline family
    return build(scores, man, ev, fake_pixels, per_class=3), scores, tests[:2]


def test_png_roundtrip():
    img = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    png = base64.b64decode(png_data_uri(img).split(",", 1)[1])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    pos, chunks = 8, {}
    while pos < len(png):
        n = struct.unpack(">I", png[pos:pos + 4])[0]
        tag, data = png[pos + 4:pos + 8], png[pos + 8:pos + 8 + n]
        assert struct.unpack(">I", png[pos + 8 + n:pos + 12 + n])[0] == zlib.crc32(tag + data), tag
        chunks[tag], pos = data, pos + 12 + n
    assert struct.unpack(">IIBBBBB", chunks[b"IHDR"]) == (32, 32, 8, 2, 0, 0, 0)
    raw = np.frombuffer(zlib.decompress(chunks[b"IDAT"]), np.uint8).reshape(32, 1 + 32 * 3)
    assert (raw[:, 0] == 0).all() and np.array_equal(raw[:, 1:].reshape(32, 32, 3), img)


def test_cache_contents():
    with tempfile.TemporaryDirectory() as tmp:
        cache, scores, planted = make_cache(tmp)
        json.dumps(cache, allow_nan=False)
        imgs = cache["images"]
        assert len(imgs) == 40 and len({im["image_id"] for im in imgs}) == 40
        assert sum(im["curated"] for im in imgs) == 30 and cache["meta"]["n_random"] == 10
        assert all(im["story"] is None for im in imgs if not im["curated"])
        by_id = scores.set_index("sample_id")
        test_bases = set(scores.base_image_id[(scores.dataset != "cifar10_1")])
        for im in imgs:
            assert im["image_id"] in test_bases
            vs = [(c, s, r) for c, by in im["versions"].items() for s, r in by.items()]
            assert len(vs) == N_VERSIONS and set(im["versions"]["clean"]) == {"0"}
            for c, s, r in vs:
                assert SECTION_11 <= set(r) and r["image"].startswith("data:image/png;base64,")
                assert r["trust_layer"]["decision"] in {"trust", "caution", "reject"} and 0 <= r["trust_layer"]["p_correct"] <= 1
                assert r["context"]["corruption"] == c and r["context"]["severity"] == int(s)
        # a response equals its scores.parquet row
        im, r = imgs[0], imgs[0]["versions"]["clean"]["0"]
        row = scores[(scores.base_image_id == im["image_id"]) & (scores.corruption == "clean")].iloc[0]
        assert np.isclose(r["trust_layer"]["p_correct"], row.p_correct, atol=1e-4) and r["trust_layer"]["decision"] == row.decision
        assert r["target_model"]["prediction"] == row.pred_class and r["truth"]["correct"] == bool(row.correct)
        # every stored story really plays clean TRUST -> moderate CAUTION (right) -> heavy REJECT (confidently wrong)
        stories = [im for im in imgs if im["story"]]
        assert {im["image_id"] for im in stories} == set(planted)
        assert cache["story"] == dict(image_id=planted[0], corruption="defocus_blur", beats=[
            dict(label="Clean", severity=0), dict(label="Moderate", severity=2), dict(label="Heavy", severity=5)])
        for im in stories:
            st, v = im["story"], im["versions"]
            clean, mid, hi = v["clean"]["0"], v[st["corruption"]][str(st["moderate"])], v[st["corruption"]][str(st["heavy"])]
            assert clean["trust_layer"]["decision"] == "trust" and clean["truth"]["correct"]
            assert mid["trust_layer"]["decision"] == "caution" and mid["truth"]["correct"]
            assert hi["trust_layer"]["decision"] == "reject" and not hi["truth"]["correct"]
            assert hi["target_model"]["raw_confidence"] >= CONFIDENT
        # drift curves: 4 families, severities 0-5, decision shares sum to 1
        fc = cache["family_curves"]
        assert set(fc) == {"noise", "blur", "weather", "digital"}
        for pts in fc.values():
            assert [p["severity"] for p in pts] == [0, 1, 2, 3, 4, 5]
            assert all(np.isclose(p["trust"] + p["caution"] + p["reject"], 1) for p in pts)
        assert set(cache["meta"]["thresholds"]) == {"noise", "blur", "weather", "digital"}


def test_server():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "data").mkdir()
        cache, _, _ = make_cache(root / "data")
        (root / "data" / "demo_cache.json").write_text(json.dumps(cache))
        (root / "frontend").mkdir()
        (root / "frontend" / "index.html").write_text("<p>demo</p>")
        (root / "frontend" / "app.js").write_text("// app")
        server = make_server(port=0, root=root)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_address[1]}"

        def call(path, body=None):
            req = urllib.request.Request(base + path, data=None if body is None else body.encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req) as r:
                    return r.status, r.headers.get("Content-Type"), r.read()
            except urllib.error.HTTPError as e:
                return e.code, e.headers.get("Content-Type"), e.read()

        try:
            code, ctype, body = call("/")
            assert code == 200 and ctype.startswith("text/html") and body == b"<p>demo</p>"
            assert call("/data/demo_cache.json")[0] == 200 and call("/app.js")[0] == 200
            assert call("/data/manifest.csv")[0] == 404 and call("/../backend/main.py")[0] == 404   # whitelist only
            im = cache["images"][0]
            corr = next(c for c in im["versions"] if c != "clean")
            code, _, body = call("/predict", json.dumps({"sample_id": im["image_id"], "corruption": corr, "severity": 3}))
            got = json.loads(body)
            want = {k: v for k, v in im["versions"][corr]["3"].items() if k != "image"}
            assert code == 200 and got == want and SECTION_11 <= set(got) and "image" not in got
            code, _, body = call("/predict", json.dumps({"sample_id": im["image_id"], "corruption": corr, "severity": 0}))
            assert code == 200 and json.loads(body)["context"]["corruption"] == "clean"          # severity 0 = clean image
            assert call("/predict", "not json")[0] == 400 and call("/predict", "{}")[0] == 400
            assert call("/predict", json.dumps({"sample_id": "nope", "severity": 2, "corruption": corr}))[0] == 404
            assert call("/other", "{}")[0] == 404
            # starting a second demo on a busy port fails with a clear message instead of a traceback
            second = subprocess.run([sys.executable, "-m", "backend.main", "--port", str(server.server_address[1]), "--no-browser"],
                                    capture_output=True, text=True, timeout=60, cwd=Path(__file__).resolve().parents[1])
            assert second.returncode == 1 and "already in use" in second.stderr and "Traceback" not in second.stderr
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
