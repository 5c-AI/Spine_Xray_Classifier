# Spine Lumbarization Detector API

Internal detector API for lumbar-spine X-rays — **not a public entry point**. Called by the
Spine N/A API (`r4new`); its positive vote is OR'd into the study verdict. Model =
**YOLO11m segmentation** of vertebrae + a deterministic **count rule**.

Deployment model: `YOLO11m-seg + count-below-D12` — an AP view is **lumbarization-positive**
if exactly `LUM_COUNT` (=6) vertebrae are found below the D12 anchor (L1–L5 + S1 instead of
L1–L5). Input is **AP views only** — the N/A entry sends the AP view(s) selected by the VIEW router.

## Output
```
segment vertebrae -> anchor on D12 -> count vertebrae below D12
image_positive = (n_below_d12 == 6)         # -1 => D12 not found (inconclusive, negative)
study_positive = any(image_positive for the study's AP views)
```

## Structure
```
main.py               FastAPI app: lifespan (load + warmup), routers
api/predict.py        POST /predict  (multipart study_id + AP files[])
api/health.py         GET  /health   (loaded | loading | error; ?deep=1 forward check)
utils/config.py       env-driven config (MODEL_PATH, COUNT_POS, LUM_DIR)
utils/preprocess.py   parse a YOLO-seg result -> vertebra detections (mirrors run_segmentation)
utils/model.py        YOLO11m-seg batched predict + D12-count rule, single-flight lock
utils/logger.py       per-study INFO log -> logs/lum_api.log
models/               weights (gitignored) — see "Download models"
logs/                 per-study logs (gitignored)
```

## Download models
```
models/Vertebra_seg.pt          # YOLO11m dorsal-lumbar vertebra segmentation (~44 MB)
```
Code dependency: the `find_d12_anchor`, `get_vertebrae_below_d12`, `extract_min_area_rect`
helpers (`LUM_DIR`, default `/root/SPINE_PATHOLOGIES/LUMBARIZATION`) must be importable —
vendor `lumbarization_detection.py` into this repo before deploy.

## Run
```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --host 0.0.0.0 --port 8002
```
Docker: `docker build -t spine-lum-api . && docker run --gpus all -p 8002:8002 -v $PWD/models:/app/models spine-lum-api`

## API
`POST /predict` — `multipart/form-data`: `study_id` (field) + `files` (repeated **AP** image files).
```json
{
  "study_id": "...",
  "lumbarization": false,
  "per_image": {
    "v_ap.jpeg": { "n_below_d12": 5, "positive": false }
  },
  "timing_ms": { "wall_clock": 90.0 }
}
```
`GET /health` — `{ "status": "loaded", "model_version": "...", "device": "cuda:0" }`
(200 loaded / 503 otherwise). `?deep=1` also runs a dummy forward.

## Notes
- **Input**: AP views only, already-converted 8-bit JPEG/PNG (DICOM→JPEG upstream). CPU decode
  (inside ultralytics); GPU does the segmentation forward.
- **Single-flight**: one study at a time (`asyncio.Lock`), held the whole request. No queue cap,
  no per-request timeout (~287 studies/day).
- **Timing** is wall-clock (`t_out - t_in`); all AP views go through a single batched seg call.
- **AP routing**: LUM does not detect view — it trusts the VIEW router; a lateral fed here would
  miscount, so the N/A entry only forwards AP views.
