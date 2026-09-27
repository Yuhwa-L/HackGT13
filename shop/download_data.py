"""Stream the three ImageNetV2 tars and keep only the product classes (~1.26 GB each streamed, ~100 MB kept).

Run: python -m shop.download_data          -> data/shop/raw/<variant>/<imagenet index>/<file>.jpeg
ImageNetV2 folders are named by ImageNet class index. Nothing else in the tars is written to disk.
"""
import ssl
import tarfile
import threading
import urllib.request

import certifi

from shop.classes import product_indices
from shop.config import BANK_VARIANT, BENCHMARK_VARIANTS, IMAGENETV2_URL, shop_path


def fetch(variant, keep):
    n = 0
    ctx = ssl.create_default_context(cafile=certifi.where())  # python.org Python on macOS lacks system CAs
    with urllib.request.urlopen(IMAGENETV2_URL.format(variant), timeout=60, context=ctx) as resp:
        with tarfile.open(fileobj=resp, mode="r|*") as tar:  # named .tar.gz but actually plain tar
            for m in tar:
                parts = m.name.split("/")
                if not m.isfile() or len(parts) < 3 or parts[-2] not in keep:
                    continue
                out = shop_path("raw", variant, parts[-2], parts[-1])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(tar.extractfile(m).read())
                n += 1
    print(f"{variant}: kept {n} images")


def main():
    keep = {str(i) for i in product_indices()}
    threads = [threading.Thread(target=fetch, args=(v, keep)) for v in [*BENCHMARK_VARIANTS, BANK_VARIANT]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
