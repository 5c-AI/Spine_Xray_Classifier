# spine-lum
YOLO11m-seg lumbarization (D12 count), input=AP views

## Run
```bash
pip install -r requirements.txt          # place weights in models/, set paths via env (utils/config.py)
uvicorn main:app --host 0.0.0.0 --port 8002
```
POST /predict · GET /health. Single-flight, batched, wall-clock timing.
External deps (vendor or pip-install): see utils/config.py (CORE_DIR/OSTEO_DIR/LUM_DIR) + weight in models/.
