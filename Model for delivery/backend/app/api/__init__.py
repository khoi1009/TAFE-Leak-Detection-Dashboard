# backend/app/api/__init__.py
"""API Router initialization."""
from fastapi import APIRouter

from app.api.auth import router as auth_router
from app.api.incidents import router as incidents_router
from app.api.schools import router as schools_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(auth_router)
api_router.include_router(incidents_router)
api_router.include_router(schools_router)
