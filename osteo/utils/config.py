import os
SERVICE_NAME = "osteo"
CKPT = os.getenv("CKPT", "models/osteophytes.ckpt")
_SVC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # service root
OSTEO_DIR = os.getenv("OSTEO_DIR", os.path.join(_SVC, "vendor"))  # vendored train.py/test.py (EfficientDetModel)
CONF = float(os.getenv("CONF", "0.30"))
DEVICE = os.getenv("DEVICE", "cuda")
PORT = int(os.getenv("PORT", "8003"))
