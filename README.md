# Spine Normal/Abnormal API

Study-level Normal/Abnormal auto-clear classifier for lumbar-spine X-rays.
Model = **R4_new RAD-DINO/MAIRA study classifier** OR'd with the osteophyte detector
(**∪osteo**) and the lumbarization detector (**∪lum**). This is the single public entry
point: every study hits this API, which runs the N/A model **and** calls the internal
OSTEO and VIEW→LUM detector APIs in parallel, then combines them.

Deployment model: `R4_new (study max-prob) ∪osteo@0.30 ∪lum(count=6)` — operating point
**S95** (`NA_THRESHOLD=0.0077`, `OSTEO_THRESHOLD=0.30`, `LUM_COUNT=6`).
Use `NA_THRESHOLD=0.0102` for the spina-bifida-ignored label definition.

## Decision
```
is_normal = (na_prob < 0.0077)
            AND (no osteophyte class >= 0.30)
            AND (lumbarization not detected)
```
`na_prob` is a single study-level probability (the classifier pools all views by **max**).
LUM runs only on the **AP** view(s) selected by the VIEW router. If any detector API is
unavailable, the study is treated conservatively as abnormal (routed to HIL) and the
corresponding block carries `error` (e.g. `osteo_results.error = "osteo_unavailable"`).

## Structure
```
main.py                   FastAPI app: lifespan (load + warmup), routers
api/predict.py            POST /predict  (multipart study_id + files[])
api/health.py             GET  /health   (loaded | loading | error; ?deep=1 forward check)
utils/config.py           env-driven config (thresholds, detector URLs)
utils/preprocess.py       EXACT R4_new preprocessing (RAD-DINO processor @1024, no letterbox/CLAHE)
utils/model.py            RAD-DINO/MAIRA encoder + binary head, single-flight lock, infer/warmup
utils/osteo_client.py     async call to the internal OSTEO API
utils/lum_client.py       async call to the internal LUM API (AP views)
utils/view_client.py      async call to the internal VIEW API (AP/LAT routing)
utils/logger.py           per-study INFO log -> logs/na_api.log
models/                   weights (gitignored) — see "Download models" below
logs/                     per-study logs (gitignored)
```

## Download models
Weights are **not** in git. Download the following into `models/`
(ask the maintainer for the storage location):
```
models/r4new_config.json       # ChestClassifier/RAD-DINO config (arch is built from this)
models/r4new_best_model.pth    # complete fine-tuned encoder + binary head (~1.0 GB)
```
Note: the base `microsoft/rad-dino-maira-2` tower is used only to **build** the architecture;
the fine-tuned `.pth` then loads the encoder + head over it, so the raw base checkpoint is not
otherwise required at serve time.

Code dependency: **vendored** — `ChestClassifier` (`model.py`) and `build_image_processor` /
`get_transforms` (`data_loader.py`) live in `r4new/core/`, committed with the repo. `CORE_DIR`
defaults to that in-repo path. No external `/root` checkout needed.

⚠ **Offline caveat (weights, not code):** building `ChestClassifier` calls
`AutoModel.from_pretrained("microsoft/rad-dino-maira-2")`, which pulls the base tower from the
HuggingFace hub on first run. For an air-gapped/offline deploy, pre-populate the HF cache
(`HF_HOME`) or snapshot that model locally and set `model_name` to the local path — otherwise
the service needs network access the first time it loads.

## Run
```bash
pip install -r requirements.txt
cp .env.example .env               # adjust if needed
# start the detector APIs first (spine-osteo-api / spine-lum-api / spine-view-api), then:
uvicorn main:app --host 0.0.0.0 --port 8001
# N/A must know where the detector APIs are:
#   OSTEO_URL=http://127.0.0.1:8003/predict
#   VIEW_URL=http://127.0.0.1:8004/predict
#   LUM_URL=http://127.0.0.1:8002/predict
```
Docker: `docker build -t spine-na-api . && docker run --gpus all -p 8001:8001 -v $PWD/models:/app/models spine-na-api`

## API
`POST /predict` — `multipart/form-data`: `study_id` (field) + `files` (repeated image files).
Optional query: `?na_threshold=` `?osteo_threshold=`.
```json
{
  "study_identifier": "...",
  "is_normal": false,
  "confidence": 0.0819,
  "osteo_results": {
    "predicted": true,
    "per_image": { "v1.jpeg": { "max_conf": 0.42, "positive": true } },
    "timing_ms": { "api": 214.0, "model": 205.1 }
  },
  "lum_results": {
    "predicted": false,
    "per_image": { "v_ap.jpeg": { "n_below_d12": 5, "positive": false } },
    "timing_ms": { "api": 90.0, "model": 61.0 }
  },
  "timings_ms": {
    "normal_abnormal_model": 180.2, "osteo_model": 205.1,
    "view_model": 33.0, "lum_model": 61.0, "overall_api": 235.9
  }
}
```
`GET /health` — `{ "status": "loaded", "model_version": "...", "device": "cuda:0" }`
(200 loaded / 503 otherwise). `?deep=1` also runs a dummy forward.

## Notes
- **Input**: engineering sends already-converted 8-bit JPEG/PNG views (DICOM→JPEG done on their
  side with the training-matched pipeline). This service only decodes/resizes/normalizes on CPU;
  the GPU does only the forward.
- **Single-flight**: one study processed at a time (`asyncio.Lock`), held for the whole request.
  No queue cap, no per-request timeout (volume ~287 studies/day).
- **Overall timing** is wall-clock (`t_out - t_in`), not the sum of the parallel model times
  (the N/A forward, the OSTEO call, and the VIEW→LUM sub-chain run concurrently).
