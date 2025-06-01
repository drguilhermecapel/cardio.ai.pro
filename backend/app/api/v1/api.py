"""
API v1 router configuration.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth,
    ecg_analysis,
    notifications,
    patients,
    users,
    validations,
)

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(ecg_analysis.router, prefix="/ecg", tags=["ecg-analysis"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(patients.router, prefix="/patients", tags=["patients"])
api_router.include_router(validations.router, prefix="/validations", tags=["validations"])
api_router.include_router(notifications.router, prefix="/notifications", tags=["notifications"])
