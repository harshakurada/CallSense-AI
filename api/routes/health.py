from fastapi import APIRouter

from configs.settings import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    settings = get_settings()
    return {"status": "ok", "environment": settings.app_env}
