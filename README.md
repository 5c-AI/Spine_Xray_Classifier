# spine-ensemble-serving

Serving for the finalized **R4_new + LUM + OSTEO** spine normal/abnormal ensemble.
Each service is a self-contained FastAPI app (`api/ · utils/ · models/ · logs/`), one **folder** per service.
Deployment convention: **push each folder to its own branch** (r4new / osteo / lum / view).

| folder | service | port | role |
|--------|---------|------|------|
| `r4new/`| RAD-DINO/MAIRA classifier | 8001 | **entry / orchestrator** (clients: osteo, lum, view) |
| `osteo/`| EfficientDet-D5 osteophyte | 8003 | detector |
| `lum/`  | YOLO11m-seg lumbarization  | 8002 | detector (input = AP views from `view`) |
| `view/` | YOLOv8-cls AP/LAT          | 8004 | view router |

Fusion (in r4new): `Abnormal iff r4_score >= T OR osteo.osteophyte OR lum.lumbarization`.
Design: batched GPU forward/study · single-flight asyncio.Lock (whole request) · CPU decode + GPU forward-only · wall-clock timing · no queue cap/timeout (~287 studies/day).

Bring-up order: view, lum, osteo, then r4new.
