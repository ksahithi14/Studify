from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Backend-only Studify service for ingestion, retrieval, summary, and quiz generation.",
)
app.include_router(router)


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "name": settings.app_name,
        "message": "Studify Agent Backend is running.",
        "docs": "/docs",
    }

