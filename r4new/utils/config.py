import os
SERVICE_NAME = "r4new"
CKPT = os.getenv("CKPT", "models/Spine_classifier.pth")
R4_CONFIG = os.getenv("R4_CONFIG", "models/Spine_classifier.json")
CORE_DIR = os.getenv("CORE_DIR", "/root/SPINE/maira-binary")   # model/data_loader/utils
THRESHOLD = float(os.getenv("THRESHOLD", "0.0077"))            # recall95, orig GT (0.0102 for SB-ignored)
IMG = int(os.getenv("IMG", "1024"))
DEVICE = os.getenv("DEVICE", "cuda")
PORT = int(os.getenv("PORT", "8001"))
OSTEO_URL = os.getenv("OSTEO_URL", "http://127.0.0.1:8003")
LUM_URL   = os.getenv("LUM_URL",   "http://127.0.0.1:8002")
VIEW_URL  = os.getenv("VIEW_URL",  "http://127.0.0.1:8004")
