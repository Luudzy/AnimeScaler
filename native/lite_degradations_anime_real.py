"""Stronger real-world anime degradation profile for isolated experiments."""
import io
import random
import numpy as np
from PIL import Image, ImageFilter


def degrade(high, rng, scale=2):
    size = (round(high.width / scale), round(high.height / scale))
    low = high
    if rng.random() < .4:
        low = low.filter(ImageFilter.GaussianBlur(rng.uniform(.2, 1.0)))
    low = low.resize(size, rng.choice([Image.Resampling.BICUBIC, Image.Resampling.LANCZOS,
                                       Image.Resampling.BILINEAR, Image.Resampling.BOX]))
    if rng.random() < .65:
        y, cb, cr = low.convert('YCbCr').split()
        chroma = (max(1, size[0] // 2), max(1, size[1] // 2))
        cb = cb.resize(chroma, Image.Resampling.BOX).resize(size, Image.Resampling.BILINEAR)
        cr = cr.resize(chroma, Image.Resampling.BOX).resize(size, Image.Resampling.BILINEAR)
        low = Image.merge('YCbCr', (y, cb, cr)).convert('RGB')
    if rng.random() < .65:
        buf = io.BytesIO()
        low.save(buf, format='JPEG', quality=rng.randint(60, 92), subsampling=2)
        buf.seek(0)
        low = Image.open(buf).convert('RGB')
    if rng.random() < .45:
        arr = np.asarray(low, dtype=np.float32)
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(.8, 3.0), arr.shape)
        low = Image.fromarray(np.uint8(np.clip(np.round(arr + noise), 0, 255)))
    return low

def degrade_balanced(high, rng, scale=2):
    """Lower-strength restoration degradation for sharp AnimeVideo-style finetuning."""
    size = (round(high.width / scale), round(high.height / scale))
    low = high
    if rng.random() < .22:
        low = low.filter(ImageFilter.GaussianBlur(rng.uniform(.12, .55)))
    low = low.resize(size, rng.choice([Image.Resampling.BICUBIC, Image.Resampling.LANCZOS,
                                       Image.Resampling.BILINEAR]))
    if rng.random() < .35:
        y, cb, cr = low.convert('YCbCr').split()
        chroma = (max(1, size[0] // 2), max(1, size[1] // 2))
        cb = cb.resize(chroma, Image.Resampling.BOX).resize(size, Image.Resampling.BILINEAR)
        cr = cr.resize(chroma, Image.Resampling.BOX).resize(size, Image.Resampling.BILINEAR)
        low = Image.merge('YCbCr', (y, cb, cr)).convert('RGB')
    if rng.random() < .42:
        buf = io.BytesIO(); low.save(buf, format='JPEG', quality=rng.randint(76, 96), subsampling=2)
        buf.seek(0); low = Image.open(buf).convert('RGB')
    if rng.random() < .28:
        arr = np.asarray(low, dtype=np.float32)
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(.35, 1.6), arr.shape)
        low = Image.fromarray(np.uint8(np.clip(np.round(arr + noise), 0, 255)))
    return low
