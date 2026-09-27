"""Owner: D. Local demo server, standard library only: python -m backend.main [--port 8000] [--no-browser]

Serves the demo page (frontend/), the data files it reads, and POST /predict answered from data/demo_cache.json in the
project doc's section-11 format:  {"sample_id": "c10_00069", "corruption": "defocus_blur", "severity": 5}
Binds to 127.0.0.1 and serves only the files listed in FILES; no internet needed at the expo.
/predict reads the cache once at startup: restart the server after rebuilding it (the page itself reloads the files).
"""
import argparse
import json
import sys
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:  # optional Visa wrapper (shop/); the core demo is unchanged without it (plan.md section 11)
    from shop.api import ShopAPI
except ImportError:
    ShopAPI = None

ROOT = Path(__file__).resolve().parents[1]
FILES = {"/": ("frontend/index.html", "text/html; charset=utf-8"),
         "/app.js": ("frontend/app.js", "text/javascript; charset=utf-8"),
         "/data/demo_cache.json": ("data/demo_cache.json", "application/json"),
         "/data/evaluation.json": ("data/evaluation.json", "application/json"),
         "/data/thresholds.json": ("data/thresholds.json", "application/json")}
MAX_BODY = 10_000


def load_index(cache_path):
    """(image_id, corruption, severity) -> section-11 response, without the pixels."""
    cache = json.loads(Path(cache_path).read_text())
    return {(im["image_id"], corr, int(sev)): {k: v for k, v in resp.items() if k != "image"}
            for im in cache["images"] for corr, by_sev in im["versions"].items() for sev, resp in by_sev.items()}


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, root, index, shop=None, **kwargs):
        self.root, self.index, self.shop = root, index, shop
        super().__init__(*args, **kwargs)

    def log_message(self, *args):   # keep the terminal quiet during the demo
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _shop(self, method, body=b""):
        """True if the shop wrapper answered this request."""
        if self.path.split("?")[0] == "/shop.js" and self.shop is None:
            self._send(200, b"// Snap-to-Shop not installed", "text/javascript; charset=utf-8")
            return True
        res = self.shop.handle(method, self.path, body) if self.shop else None
        if res is not None:
            self._send(res[0], res[2], res[1])
        return res is not None

    def do_GET(self):
        if self._shop("GET"):
            return
        entry = FILES.get(self.path.split("?")[0])
        path = entry and self.root / entry[0]
        if not path or not path.exists():
            return self._send(404, b"not found", "text/plain")
        self._send(200, path.read_bytes(), entry[1])

    def do_POST(self):
        if self.path.startswith("/api/shop/"):
            size = int(self.headers.get("Content-Length", 0))
            if size > MAX_BODY:
                return self._json(413, {"error": "request too large"})
            if self._shop("POST", self.rfile.read(size)):
                return
        if self.path != "/predict":
            return self._json(404, {"error": "unknown endpoint; use POST /predict"})
        try:
            size = int(self.headers.get("Content-Length", 0))
            if size > MAX_BODY:
                return self._json(413, {"error": "request too large"})
            req = json.loads(self.rfile.read(size) or b"{}")
            severity = int(req.get("severity", 0))
            key = (str(req["sample_id"]), "clean" if severity == 0 else str(req.get("corruption", "clean")), severity)
        except (KeyError, ValueError, TypeError, AttributeError):
            return self._json(400, {"error": 'send JSON like {"sample_id": "c10_00069", "corruption": "defocus_blur", "severity": 5}'})
        if self.index is None:
            return self._json(503, {"error": "data/demo_cache.json is missing; run python -m backend.build_demo_cache"})
        result = self.index.get(key)
        if result is None:
            return self._json(404, {"error": "that image / corruption / severity is not in the demo cache"})
        self._json(200, result)


def make_server(port=8000, root=ROOT):
    cache = root / "data" / "demo_cache.json"
    index = load_index(cache) if cache.exists() else None
    shop = ShopAPI.load() if ShopAPI else None
    return ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, root=root, index=index, shop=shop))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    try:
        server = make_server(args.port)
    except OSError:   # most likely the demo is already running in another terminal
        sys.exit(f"Port {args.port} is already in use. If the demo is already running, open http://localhost:{args.port}/ "
                 f"or stop it with Ctrl+C there; otherwise start this one with --port {args.port + 1}.")
    url = f"http://localhost:{server.server_address[1]}/"
    print(f"demo running at {url}  (Ctrl+C to stop; safe to restart any time, nothing is written)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
