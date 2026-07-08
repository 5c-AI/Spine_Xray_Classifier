import os
SERVICE_NAME = "lum"
MODEL_PATH = os.getenv("MODEL_PATH", "models/lum_yolo11m_seg.pt")
LUM_DIR = os.getenv("LUM_DIR", "/root/SPINE_PATHOLOGIES/LUMBARIZATION")  # helpers
COUNT_POS = int(os.getenv("COUNT_POS", "6"))
DEVICE = os.getenv("DEVICE", "cuda")
PORT = int(os.getenv("PORT", "8002"))
