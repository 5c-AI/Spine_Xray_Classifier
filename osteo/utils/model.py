import sys, torch
from utils import config as C
from utils.preprocess import load
sys.path.append(C.OSTEO_DIR)
from train import EfficientDetModel, get_valid_transforms
from test import predict_batch
class Model:
    def __init__(self):
        m = EfficientDetModel.load_from_checkpoint(C.CKPT, map_location=C.DEVICE)
        self.model = m.eval().to(C.DEVICE)
        self.tf = get_valid_transforms(self.model.image_dim)
    def predict(self, images):
        xs, sizes, ok = [], [], []
        for p in images:
            try: t, sz = load(p, self.tf)
            except Exception: continue
            xs.append(t); sizes.append(sz); ok.append(p)
        per, any_pos = {}, False
        if xs:
            preds = predict_batch(self.model, torch.stack(xs), sizes, conf_threshold=0.0)
            for p, pr in zip(ok, preds):
                sc = pr["scores"]; mc = float(sc.max()) if sc.size else 0.0
                pos = mc >= C.CONF; per[p] = {"max_conf": round(mc,4), "positive": pos}
                any_pos = any_pos or pos
        return {"osteophyte": any_pos, "per_image": per}
