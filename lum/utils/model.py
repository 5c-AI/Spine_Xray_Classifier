import sys
from ultralytics import YOLO
from utils import config as C
from utils.preprocess import dets_from_result
sys.path.append(C.LUM_DIR)
import lumbarization_detection as L  # full original decision chain
class Model:
    def __init__(self):
        self.model = YOLO(C.MODEL_PATH)
    def predict(self, ap_images):
        per, any_pos = {}, False
        if not ap_images: return {"lumbarization": False, "per_image": per}
        results = self.model.predict(source=ap_images, conf=0.5, verbose=False)  # batched
        for p, res in zip(ap_images, results):
            dets = dets_from_result(res)
            d12 = L.find_d12_anchor(dets)
            if d12 is None:
                per[p] = {"n_below_d12": -1, "positive": False}; continue
            # EXACT original chain: below-D12 -> dedup split boxes -> reorder -> analyze (==6 or >6 => positive)
            below = L.get_vertebrae_below_d12(dets, d12)
            below = L.remove_spatial_duplicates(below)
            below = L.reorder_lumbar_vertebrae(below)
            a = L.analyze_lumbarization(d12, below)
            pos = bool(a["lumbarization_detected"]); n = int(a["vertebrae_count"])
            per[p] = {"n_below_d12": n, "positive": pos}
            any_pos = any_pos or pos
        return {"lumbarization": any_pos, "per_image": per}
