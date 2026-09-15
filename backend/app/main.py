"""CallSense AI backend entrypoint.

Route handlers stay thin; real logic lives in ml/ and future inference
pipeline modules (see docs/module_01_problem_definition_and_architecture.md,
section 9). Endpoints below are placeholders for Module 19 and return
501 until the corresponding ML module is implemented.
"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="CallSense AI API", version="0.1.0")

NOT_IMPLEMENTED = {"detail": "Not implemented yet — pipeline module pending."}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/upload-call")
def upload_call():
    return JSONResponse(status_code=501, content=NOT_IMPLEMENTED)


@app.post("/transcribe")
def transcribe():
    return JSONResponse(status_code=501, content=NOT_IMPLEMENTED)


@app.post("/analyze")
def analyze():
    return JSONResponse(status_code=501, content=NOT_IMPLEMENTED)


@app.get("/conversation/{conversation_id}")
def get_conversation(conversation_id: str):
    return JSONResponse(status_code=501, content=NOT_IMPLEMENTED)


@app.get("/analytics")
def get_analytics():
    return JSONResponse(status_code=501, content=NOT_IMPLEMENTED)


@app.post("/search")
def search():
    return JSONResponse(status_code=501, content=NOT_IMPLEMENTED)


@app.get("/agent/{agent_id}")
def get_agent(agent_id: str):
    return JSONResponse(status_code=501, content=NOT_IMPLEMENTED)
