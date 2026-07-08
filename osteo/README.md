# spine-osteo
EfficientDet-D5 osteophyte detector (batched)

## Run
```bash
pip install -r requirements.txt          # place weights in models/, set paths via env (utils/config.py)
uvicorn main:app --host 0.0.0.0 --port 8003
```
POST /predict · GET /health. Single-flight, batched, wall-clock timing.
External deps (vendor or pip-install): see utils/config.py (CORE_DIR/OSTEO_DIR/LUM_DIR) + weight in models/.
