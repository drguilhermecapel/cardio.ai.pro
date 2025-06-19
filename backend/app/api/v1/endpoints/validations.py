from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, status, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.core.exceptions import (
    NotFoundException, 
    ValidationException, 
    UnauthorizedException
)

from app.schemas.validation import (
    ValidationCreate,
    Validation,
    ValidationSubmit
)
from app.services.validation_service import ValidationService
from app.repositories.validation_repository import ValidationRepository
from app.repositories.ecg_repository import ECGRepository
from app.repositories.notification_repository import NotificationRepository
from app.services.notification_service import NotificationService
from app.models.user import UserRoles

router = APIRouter(prefix="/validations", tags=["validations"])

@router.post("/", response_model=Validation)
async def create_validation(
    validation_data: ValidationCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new validation request"""
    validation_repo = ValidationRepository(db)
    ecg_repo = ECGRepository(db)
    notification_repo = NotificationRepository(db)
    notification_service = NotificationService(db, notification_repo)
    
    validation_service = ValidationService(
        db=db,
        validation_repository=validation_repo,
        ecg_repository=ecg_repo,
        notification_service=notification_service
    )
    
    validation = await validation_service.create_validation(
        validation_data,
        current_user.id
    )
    return validation

@router.get("/pending", response_model=List[Validation])
async def get_pending_validations(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get pending validations for current user"""
    if current_user.role not in [UserRoles.PHYSICIAN, UserRoles.CARDIOLOGIST, UserRoles.ADMIN]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    validation_repo = ValidationRepository(db)
    validations = await validation_repo.get_pending_validations_for_user(
        current_user.id,
        skip=skip,
        limit=limit
    )
    return validations

@router.get("/{validation_id}", response_model=Validation)
async def get_validation(
    validation_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get validation by ID"""
    validation_repo = ValidationRepository(db)
    validation = await validation_repo.get_validation_by_id(validation_id)
    
    if not validation:
        raise HTTPException(status_code=404, detail="Validation not found")
    
    return validation

@router.post("/{validation_id}/submit", response_model=Validation)
async def submit_validation(
    validation_id: int,
    submission: ValidationSubmit,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Submit validation results"""
    validation_repo = ValidationRepository(db)
    ecg_repo = ECGRepository(db)
    notification_repo = NotificationRepository(db)
    notification_service = NotificationService(db, notification_repo)
    
    validation_service = ValidationService(
        db=db,
        validation_repository=validation_repo,
        ecg_repository=ecg_repo,
        notification_service=notification_service
    )
    
    validation = await validation_service.submit_validation(
        validation_id,
        submission,
        current_user.id
    )
    return validation
