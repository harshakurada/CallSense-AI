"""CallSense AI REST API entrypoint.

Route handlers stay thin; real pipeline logic will live in src/inference
(see docs/ARCHITECTURE.md). Run with:
    uvicorn api.main:app --reload
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from configs.settings import get_settings
from src.utils.exceptions import CallSenseError
from src.utils.logging import get_logger
from api.routes import calls, health

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("CallSense AI API starting in %s mode", settings.app_env)
    yield


app = FastAPI(title="CallSense AI API", version="0.1.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(calls.router)


@app.exception_handler(CallSenseError)
def handle_pipeline_error(request, exc: CallSenseError):
    logger.error("Pipeline error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=422, content={"detail": str(exc)})
