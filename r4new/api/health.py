from fastapi import APIRouter, Request
router = APIRouter()
@router.get("/health")
async def health(request: Request):
    st = request.app.state
    return {"status": "ok", "service": st.service_name, "model_loaded": st.model is not None}
