import os
SERVICE_NAME = "view"
MODEL_PATH = os.getenv("MODEL_PATH", "models/YOLOCLSS.pt")
DEVICE = os.getenv("DEVICE", "cuda")
IMG = int(os.getenv("IMG", "640"))
PORT = int(os.getenv("PORT", "8004"))
