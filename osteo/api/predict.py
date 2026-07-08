import time, asyncio
from fastapi import APIRouter, Request
from pydantic import BaseModel
router = APIRouter()
class Req(BaseModel):
    study_id: str
    images: list[str]
@router.post("/predict")
async def predict(req: Req, request: Request):
    st = request.app.state; t0 = time.time()
    async with st.lock:
        out = await asyncio.to_thread(st.model.predict, req.images)
    out.update(study_id=req.study_id, timing_ms={"wall_clock": round((time.time()-t0)*1000)})
    return out
