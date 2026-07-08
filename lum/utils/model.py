import sys
from ultralytics import YOLO
from utils import config as C
sys.path.append(C.LUM_DIR)
import lumbarization_detection as L  # full original decision chain (run_segmentation + analyze)
class Model:
    def __init__(self):
        self.model = YOLO(C.MODEL_PATH)
    def predict(self, ap_images):
        # PER-IMAGE by design: LUM is a count-based decision and batched YOLO rect-padding
        # shifts borderline vertebra counts (5<->6). A study has only 1-2 AP views, so per-image
        # (native imgsz, exactly reproducing lumbarization_detection.run_segmentation) costs ~nothing
        # and matches the reference predictions bit-for-bit.
        per, any_pos = {}, False
        if not ap_images: return {"lumbarization": False, "per_image": per}
        for p in ap_images:
            dets = L.run_segmentation(self.model, p)          # single-image, native imgsz
            d12 = L.find_d12_anchor(dets)
            if d12 is None:
                per[p] = {"n_below_d12": -1, "positive": False}; continue
            below = L.get_vertebrae_below_d12(dets, d12)
            below = L.remove_spatial_duplicates(below)
            below = L.reorder_lumbar_vertebrae(below)
            a = L.analyze_lumbarization(d12, below)
            pos = bool(a["lumbarization_detected"]); n = int(a["vertebrae_count"])
            per[p] = {"n_below_d12": n, "positive": pos}
            any_pos = any_pos or pos
        return {"lumbarization": any_pos, "per_image": per}
