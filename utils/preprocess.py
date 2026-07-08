import cv2, numpy as np
from PIL import Image
from ultralytics.data.augment import classify_transforms
def build_tf(img): return classify_transforms(size=img)
def load(path, tf):
    bgr = cv2.imread(path)
    if bgr is None: return None
    return tf(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
