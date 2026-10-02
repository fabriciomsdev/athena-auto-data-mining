from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.endpoints import data_sources, health, pipelines

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(
    data_sources.router, prefix="/data-sources", tags=["data-sources"]
)
api_router.include_router(pipelines.router, prefix="/pipelines", tags=["pipelines"])
