# spine-ensemble-serving

Serving for the finalized **R4_new + LUM + OSTEO** spine normal/abnormal ensemble.
**One branch per service** (your convention), each a standalone FastAPI app in the
`api/ · utils/ · models/ · logs/` skeleton.

| branch | service | port | role |
|--------|---------|------|------|
| `r4new`| RAD-DINO/MAIRA classifier | 8001 | **entry / orchestrator** (clients: osteo, lum, view) |
| `osteo`| EfficientDet-D5 osteophyte | 8003 | detector |
| `lum`  | YOLO11m-seg lumbarization  | 8002 | detector (input = AP views from `view`) |
| `view` | YOLOv8-cls AP/LAT          | 8004 | view router |

Fusion (in `r4new`): `Abnormal iff r4_score >= T OR osteo.osteophyte OR lum.lumbarization`.

Design: batched GPU forward per study; single-flight `asyncio.Lock` per service (held whole
request); CPU decode/resize/normalize, GPU forward-only; wall-clock timing; no queue cap /
no per-request timeout (~287 studies/day).

## Deploy each service (from its branch)
```bash
git checkout <branch>
pip install -r requirements.txt
# put weights in models/ and set paths via env (see utils/config.py)
uvicorn main:app --host 0.0.0.0 --port <port>
```
Bring-up order: view, lum, osteo, then r4new (entry).
