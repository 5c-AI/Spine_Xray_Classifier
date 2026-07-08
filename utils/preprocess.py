import cv2
def load(path, tf, proc):
    bgr = cv2.imread(path)
    if bgr is None: return None
    aug = tf(image=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))["image"]
    return proc(aug, return_tensors="pt")["pixel_values"][0]
