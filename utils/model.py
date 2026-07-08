import sys, json, torch
from utils import config as C
from utils.preprocess import load
# append (not insert) so the service's own `utils` package keeps priority over any
# module of the same name in the core repo. Core repo must be importable/vendored.
sys.path.append(C.CORE_DIR)
from model import ChestClassifier
from data_loader import build_image_processor, get_transforms
def _load_ckpt(path, model):
    sd = torch.load(path, map_location="cpu")
    state = sd.get("model_state_dict", sd.get("state_dict", sd)) if isinstance(sd, dict) else sd
    model.load_state_dict(state, strict=False)   # partial load (drops aux heads / shape mismatches)
class Model:
    def __init__(self):
        cfg = json.load(open(C.R4_CONFIG))
        m = ChestClassifier(cfg["model_name"], num_classes=1,
                            hidden_dim=cfg.get("classifier_hidden_dim",512),
                            dropout_rate=cfg.get("dropout_rate",0.1),
                            freeze_encoder=True, num_subtypes=0)
        _load_ckpt(C.CKPT, m)
        self.model = m.eval().to(C.DEVICE)
        self.proc = build_image_processor(type("Cfg",(),{"model_name":cfg["model_name"],"processor_image_size":C.IMG})())
        self.tf = get_transforms(C.IMG, is_training=False, use_clahe=False, use_letterbox=False)
    @torch.inference_mode()
    def study_score(self, images):
        xs = [t for t in (load(p, self.tf, self.proc) for p in images) if t is not None]
        if not xs: return 0.0
        out = self.model(torch.stack(xs).to(C.DEVICE))
        logits = out["logits"] if isinstance(out, dict) else out
        return float(torch.sigmoid(logits.float().view(-1)).cpu().numpy().max())
