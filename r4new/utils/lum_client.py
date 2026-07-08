import httpx
from utils import config as C
async def call(client: httpx.AsyncClient, study_id, ap_images):
    r = await client.post(f"{C.LUM_URL}/predict", json={"study_id": study_id, "ap_images": ap_images})
    return r.json()
