# Spine View Classifier API

Internal helper API for lumbar-spine X-rays — **not a public entry point**. Called by the
Spine N/A API (`r4new`) to route views: only the **AP** view(s) it identifies are forwarded
to the LUM (lumbarization) detector. Model = **YOLOv8-cls** (AP vs Lateral).

Deployment model: `YOLOv8-cls AP/LAT` — per image, argmax of the 2-class head; the study's
AP views are returned as `ap_images` for downstream routing.

## Output
```
per image: {"view": "AP"|"LAT", "conf": p_ap}
ap_images = [images argmax == AP]
```
Runs on all views of the study in **one batched forward**.

## Structure
```
main.py               FastAPI app: lifespan (load + warmup), routers
api/predict.py        POST /predict  (multipart study_id + files[])
api/health.py         GET  /health   (loaded | loading | error; ?deep=1 forward check)
utils/config.py       env-driven config (MODEL_PATH, IMG)
utils/preprocess.py   classify_transforms(size=640) on RGB
utils/model.py        YOLOv8-cls batched forward, single-flight lock, infer/warmup
utils/logger.py       per-study INFO log -> logs/view_api.log
models/               weights (gitignored) — see "Download models"
logs/                 per-study logs (gitignored)
```

## Download models
```
models/YOLOCLSS.pt                 # YOLOv8-cls AP/LAT classifier (~3 MB)
```
No external code dependency (uses `ultralytics` only).

## Run
```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --host 0.0.0.0 --port 8004
```
Docker: `docker build -t spine-view-api . && docker run --gpus all -p 8004:8004 -v $PWD/models:/app/models spine-view-api`

## API
`POST /predict` — `multipart/form-data`: `study_id` (field) + `files` (repeated image files).
```json
{
  "study_id": "...",
  "views": {
    "v1.jpeg": { "view": "AP",  "conf": 0.97 },
    "v2.jpeg": { "view": "LAT", "conf": 0.06 }
  },
  "ap_images": ["v1.jpeg"],
  "timing_ms": { "wall_clock": 33.0 }
}
```
`GET /health` — `{ "status": "loaded", "model_version": "...", "device": "cuda:0" }`
(200 loaded / 503 otherwise). `?deep=1` also runs a dummy forward.

## Notes
- **Input**: already-converted 8-bit JPEG/PNG views (DICOM→JPEG upstream, training-matched).
  CPU decode/resize/normalize only; GPU does the forward.
- **Single-flight**: one study at a time (`asyncio.Lock`), held the whole request. No queue cap,
  no per-request timeout (~287 studies/day).
- **Timing** is wall-clock (`t_out - t_in`); all views go through a single batched forward.
- Lightest service in the ensemble (~3 MB model) — its only job is AP/LAT routing for LUM.
