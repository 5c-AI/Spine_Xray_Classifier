# LUM decodes inside ultralytics; parse one seg result -> vertebra detections
import sys
from utils import config as C
sys.path.append(C.LUM_DIR)
import lumbarization_detection as L  # extract_min_area_rect
def dets_from_result(result):
    out = []
    if not getattr(result, "boxes", None): return out
    for i in range(len(result.boxes)):
        b = result.boxes[i]; cid = int(b.cls.item()); cn = result.names[cid]
        d = {"class_id": cid, "class_name": cn, "confidence": float(b.conf.item()),
             "original_class_name": cn, "corrected": False}
        if getattr(result, "masks", None) is not None:
            m = result.masks[i]
            if getattr(m, "xy", None) is not None and len(m.xy) > 0:
                d["polygon_xy"] = m.xy[0].tolist()
                r = L.extract_min_area_rect(None, d["polygon_xy"])
                if r: d["min_area_rect"] = r
        if "min_area_rect" not in d:
            x1,y1,x2,y2 = b.xyxy[0].cpu().numpy()
            d["min_area_rect"] = {"cx": float((x1+x2)/2), "cy": float((y1+y2)/2),
                "width": float(x2-x1), "height": float(y2-y1), "angle_deg_opencv": 0.0,
                "corners": [{"x":float(x1),"y":float(y1)},{"x":float(x2),"y":float(y1)},
                            {"x":float(x2),"y":float(y2)},{"x":float(x1),"y":float(y2)}]}
        out.append(d)
    return out
