import numpy as np
from PIL import Image
import random

MAX_ROTATION = 5

def transform_image(img, zoom_factor, max_rotation=MAX_ROTATION):
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    w, h = pil.size

    # Zoom
    nw, nh = int(w / zoom_factor), int(h / zoom_factor)
    pil = pil.crop(((w - nw) // 2, (h - nh) // 2, (w + nw) // 2, (h + nh) // 2))
    pil = pil.resize((w, h), Image.LANCZOS)

    # Random rotation
    if random.random() < 0.3:
        angle = random.uniform(-max_rotation, max_rotation)
        bigger = Image.new("RGB", (int(w*1.2), int(h*1.2)), (0, 0, 0))
        bigger.paste(pil, ((bigger.width - w)//2, (bigger.height - h)//2))
        pil = bigger.rotate(angle, resample=Image.BICUBIC, expand=True)

        left = (pil.width - w) // 2
        top = (pil.height - h) // 2
        pil = pil.crop((left, top, left+w, top+h))

    return np.array(pil, dtype=np.float32)
