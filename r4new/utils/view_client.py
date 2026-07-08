import httpx
from utils import config as C
async def call(client: httpx.AsyncClient, study_id, images):
    r = await client.post(f"{C.VIEW_URL}/predict", json={"study_id": study_id, "images": images})
    return r.json()
