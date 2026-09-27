"""The model (ViT-B/16) half of live scoring, in its own process (torch only, never XGBoost).

On macOS, torch and XGBoost each bring their own OpenMP runtime; loaded into one process, whichever runs parallel code
second can segfault. So the server process (XGBoost, trust layer) never imports torch, and this worker, started as
`python -m shop.live_worker` (not multiprocessing, which would re-import the server's modules), never imports XGBoost.
Protocol: length-prefixed pickles over stdin/stdout. In: a (224, 224, 3) uint8 array (or None to quit).
Out: ("ready", device) once, then ("ok", (logits, tta_logits, embeddings)) or ("error", message) per request.
"""
import pickle
import struct
import sys


def send(out, msg):
    data = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
    out.write(struct.pack("<Q", len(data)) + data)
    out.flush()


def recv(inp):
    head = inp.read(8)
    if len(head) < 8:
        raise EOFError
    return pickle.loads(inp.read(struct.unpack("<Q", head)[0]))


def main():
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # anything printed by libraries must not corrupt the protocol
    try:
        import numpy as np

        from shop.model import WEIGHTS, device, forward, load_model
        if not WEIGHTS.exists():
            raise FileNotFoundError(f"model weights missing at {WEIGHTS}; see shop/README.md")
        dev = device()
        model = load_model(dev)
        forward(model, np.zeros((1, 224, 224, 3), np.uint8), dev)  # warm up the GPU path
        send(out, ("ready", str(dev)))
    except Exception as e:
        send(out, ("error", f"{type(e).__name__}: {e}"))
        return
    while True:
        try:
            img = recv(inp)
        except EOFError:
            return
        if img is None:
            return
        try:
            send(out, ("ok", forward(model, img[None], dev)))
        except Exception as e:
            send(out, ("error", f"{type(e).__name__}: {e}"))


if __name__ == "__main__":
    try:
        main()
    except (BrokenPipeError, KeyboardInterrupt):  # the server went away: exit quietly
        pass
