"""Call analysis endpoints. Each returns 501 until the pipeline module it
depends on is implemented — see docs/PROJECT_PLAN.md for the module map."""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(tags=["calls"])

_NOT_IMPLEMENTED = {"detail": "Not implemented yet — pipeline module pending."}


@router.post("/upload-call")
def upload_call():
    return JSONResponse(status_code=501, content=_NOT_IMPLEMENTED)


@router.post("/transcribe")
def transcribe():
    return JSONResponse(status_code=501, content=_NOT_IMPLEMENTED)


@router.post("/analyze")
def analyze():
    return JSONResponse(status_code=501, content=_NOT_IMPLEMENTED)


@router.get("/conversation/{conversation_id}")
def get_conversation(conversation_id: str):
    return JSONResponse(status_code=501, content=_NOT_IMPLEMENTED)


@router.get("/analytics")
def get_analytics():
    return JSONResponse(status_code=501, content=_NOT_IMPLEMENTED)


@router.post("/search")
def search():
    return JSONResponse(status_code=501, content=_NOT_IMPLEMENTED)


@router.get("/agent/{agent_id}")
def get_agent(agent_id: str):
    return JSONResponse(status_code=501, content=_NOT_IMPLEMENTED)
