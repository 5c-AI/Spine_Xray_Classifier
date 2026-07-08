import torch, numpy as np
from ultralytics import YOLO
from utils import config as C
from utils.preprocess import build_tf, load
class Model:
    def __init__(self):
        yolo = YOLO(C.MODEL_PATH)
        self.ap_idx = next((i for i,n in yolo.names.items() if str(n).upper().startswith("AP")), 0)
        self.names = yolo.names
        self.nn = yolo.model.eval().to(C.DEVICE)
        if C.DEVICE.startswith("cuda"): self.nn = self.nn.half()
        self.tf = build_tf(C.IMG)
    @torch.inference_mode()
    def predict(self, paths):
        xs, ok = [], []
        for p in paths:
            t = load(p, self.tf)
            if t is not None: xs.append(t); ok.append(p)
        views, ap = {}, []
        if xs:
            x = torch.stack(xs).to(C.DEVICE)
            if C.DEVICE.startswith("cuda"): x = x.half()
            out = self.nn(x); logits = out[1] if isinstance(out,(list,tuple)) else out
            probs = torch.softmax(logits.float(), -1).cpu().numpy()
            for p, pr in zip(ok, probs):
                is_ap = int(np.argmax(pr)) == self.ap_idx
                views[p] = {"view": "AP" if is_ap else "LAT", "conf": float(pr[self.ap_idx])}
                if is_ap: ap.append(p)
        return {"views": views, "ap_images": ap}
