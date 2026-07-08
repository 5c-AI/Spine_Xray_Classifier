import time, asyncio
import httpx
from fastapi import APIRouter, Request
from pydantic import BaseModel
from utils import config as C
from utils import osteo_client, lum_client, view_client
router = APIRouter()
class Req(BaseModel):
    study_id: str
    images: list[str]
@router.post("/predict")
async def predict(req: Req, request: Request):
    st = request.app.state; t_in = time.time(); tr = {}
    async with st.lock:                       # single-flight, whole request
        async def r4_branch():
            s = time.time(); sc = await asyncio.to_thread(st.model.study_score, req.images)
            tr["r4"] = round((time.time()-s)*1000); return sc
        async def osteo_branch():
            s = time.time(); o = await osteo_client.call(st.http, req.study_id, req.images)
            tr["osteo"] = round((time.time()-s)*1000); return o
        async def viewlum_branch():
            s = time.time()
            v = await view_client.call(st.http, req.study_id, req.images)
            ap = v.get("ap_images", [])
            l = await lum_client.call(st.http, req.study_id, ap) if ap else {"lumbarization": False, "per_image": {}}
            tr["view_lum"] = round((time.time()-s)*1000); return v, l
        r4, osteo, (view, lum) = await asyncio.gather(r4_branch(), osteo_branch(), viewlum_branch())
        r4_fired = r4 >= C.THRESHOLD
        osteo_pos = bool(osteo.get("osteophyte", False)); lum_pos = bool(lum.get("lumbarization", False))
        verdict = "Abnormal" if (r4_fired or osteo_pos or lum_pos) else "Normal"
    tr["wall_clock"] = round((time.time()-t_in)*1000)
    return {"study_id": req.study_id, "verdict": verdict,
            "r4": {"score": round(r4,4), "threshold": C.THRESHOLD, "fired": r4_fired},
            "osteo": {"fired": osteo_pos, "detail": osteo.get("per_image")},
            "lum": {"fired": lum_pos, "detail": lum.get("per_image")},
            "view": view.get("views"), "timing_ms": tr}
