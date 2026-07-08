import os
SERVICE_NAME = "osteo"
CKPT = os.getenv("CKPT", "models/osteo_effdet_d5.ckpt")
OSTEO_DIR = os.getenv("OSTEO_DIR", "/root/SPINE_PATHOLOGIES/OSTEO/effdet")  # train.EfficientDetModel
CONF = float(os.getenv("CONF", "0.30"))
DEVICE = os.getenv("DEVICE", "cuda")
PORT = int(os.getenv("PORT", "8003"))
