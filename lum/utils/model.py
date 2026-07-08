import sys
from ultralytics import YOLO
from utils import config as C
from utils.preprocess import dets_from_result
sys.path.append(C.LUM_DIR)
import lumbarization_detection as L  # find_d12_anchor, get_vertebrae_below_d12
class Model:
    def __init__(self):
        self.model = YOLO(C.MODEL_PATH)
    def predict(self, ap_images):
        per, any_pos = {}, False
        if not ap_images: return {"lumbarization": False, "per_image": per}
        results = self.model.predict(source=ap_images, conf=0.5, verbose=False)  # batched
        for p, res in zip(ap_images, results):
            dets = dets_from_result(res); d12 = L.find_d12_anchor(dets)
            n = len(L.get_vertebrae_below_d12(dets, d12)) if d12 is not None else -1
            pos = (n == C.COUNT_POS); per[p] = {"n_below_d12": n, "positive": pos}
            any_pos = any_pos or pos
        return {"lumbarization": any_pos, "per_image": per}
