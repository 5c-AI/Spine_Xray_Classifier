# Spine Osteophyte Detector API

Internal detector API for lumbar-spine X-rays — **not a public entry point**. Called in
parallel by the Spine N/A API (`r4new`); its positive vote is OR'd into the study verdict.
Model = **EfficientDet-D5** osteophyte detector.

Deployment model: `EfficientDet-D5 @ conf 0.30` — an image is **osteophyte-positive** if any
detected box survives `OSTEO_THRESHOLD`; a study is positive if any of its views is positive.

## Output
```
image_positive = (max box confidence >= 0.30)
study_positive = any(image_positive for the study's views)
```
Runs on all views of the study in **one batched forward**.

## Structure
```
main.py               FastAPI app: lifespan (load + warmup), routers
api/predict.py        POST /predict  (multipart study_id + files[])
api/health.py         GET  /health   (loaded | loading | error; ?deep=1 forward check)
utils/config.py       env-driven config (CKPT, CONF, OSTEO_DIR)
utils/preprocess.py   EXACT effdet preprocessing (get_valid_transforms @ 1280)
utils/model.py        EfficientDetModel + predict_batch, single-flight lock, infer/warmup
utils/logger.py       per-study INFO log -> logs/osteo_api.log
models/               weights (gitignored) — see "Download models"
logs/                 per-study logs (gitignored)
```

## Download models
```
models/osteophytes.ckpt        # EfficientDet-D5 fine-tuned checkpoint (~390 MB)
```
Code dependency: the `EfficientDetModel` + `get_valid_transforms` + `predict_batch` helpers
(`OSTEO_DIR`, default `/root/SPINE_PATHOLOGIES/OSTEO/effdet`) must be importable — vendor
those modules into this repo or `pip install` the effdet package before deploy.

## Run
```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --host 0.0.0.0 --port 8003
```
Docker: `docker build -t spine-osteo-api . && docker run --gpus all -p 8003:8003 -v $PWD/models:/app/models spine-osteo-api`

## API
`POST /predict` — `multipart/form-data`: `study_id` (field) + `files` (repeated image files).
Optional query: `?osteo_threshold=`.
```json
{
  "study_id": "...",
  "osteophyte": true,
  "per_image": {
    "v1.jpeg": { "max_conf": 0.42, "positive": true },
    "v2.jpeg": { "max_conf": 0.11, "positive": false }
  },
  "timing_ms": { "wall_clock": 214.0 }
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
