import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from utils import config as C
from utils.model import Model
from utils.logger import get_logger
from api import predict, health

log = get_logger(C.SERVICE_NAME)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info('loading model for %s ...', C.SERVICE_NAME)
    app.state.service_name = C.SERVICE_NAME
    app.state.model = Model()
    app.state.lock = asyncio.Lock()   # single-flight: one study at a time
    log.info('%s ready on :%s', C.SERVICE_NAME, C.PORT)
    yield

app = FastAPI(title=f"spine-{C.SERVICE_NAME}", lifespan=lifespan)
app.include_router(health.router)
app.include_router(predict.router)
