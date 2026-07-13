import os
SERVICE_NAME = "lum"
MODEL_PATH = os.getenv("MODEL_PATH", "models/Vertebra_seg.pt")
_SVC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # service root
LUM_DIR = os.getenv("LUM_DIR", os.path.join(_SVC, "vendor"))  # vendored lumbarization_detection.py
COUNT_POS = int(os.getenv("COUNT_POS", "6"))
DEVICE = os.getenv("DEVICE", "cuda")
PORT = int(os.getenv("PORT", "8002"))
