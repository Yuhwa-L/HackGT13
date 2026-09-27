"""Photo loading for the shop pipeline and live uploads: PIL + NumPy only (no torch), so any process can use it."""
import numpy as np
from PIL import Image

from shop.config import IMAGE_SIZE


def prepare(im):
    """A PIL image -> (224, 224, 3) uint8: short side 256, center crop 224 (standard ImageNet eval).
    An image that is already 224 x 224 is used as is, never resampled."""
    im = im.convert("RGB")
    w, h = im.size
    if (w, h) == (IMAGE_SIZE, IMAGE_SIZE):
        return np.asarray(im, dtype=np.uint8)
    s = 256 / min(w, h)
    im = im.resize((max(IMAGE_SIZE, round(w * s)), max(IMAGE_SIZE, round(h * s))), Image.BILINEAR)
    w, h = im.size
    left, top = (w - IMAGE_SIZE) // 2, (h - IMAGE_SIZE) // 2
    return np.asarray(im.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE)), dtype=np.uint8)


def load_photo(path):
    return prepare(Image.open(path))
