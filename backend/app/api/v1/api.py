"""
API v1 router configuration.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api.v1.endpoints import (
    auth,
    avatar,
    ecg_analysis,
    notifications,
    patients,
    users,
    validations,
)

api_router = APIRouter()

@api_router.get("/health")
async def health_check() -> JSONResponse:
    """Health check endpoint for API v1."""
    return JSONResponse(
        content={
            "status": "healthy",
            "service": "cardioai-pro-standalone",
            "version": "1.0.0"
        }
    )

api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(avatar.router, prefix="/avatar", tags=["avatar"])
api_router.include_router(ecg_analysis.router, prefix="/ecg", tags=["ecg-analysis"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(patients.router, prefix="/patients", tags=["patients"])
api_router.include_router(validations.router, prefix="/validations", tags=["validations"])
api_router.include_router(notifications.router, prefix="/notifications", tags=["notifications"])
