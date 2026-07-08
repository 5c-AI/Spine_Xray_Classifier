import numpy as np
from PIL import Image
def load(path, tf):
    img = Image.open(path).convert("RGB")
    t = tf(image=np.array(img), bboxes=[], labels=[])["image"]
    return t, img.size  # (w,h)
