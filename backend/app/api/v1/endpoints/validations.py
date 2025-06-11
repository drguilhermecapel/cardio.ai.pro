"""
Validation endpoints.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import UserRoles
from app.db.session import get_db
from app.models.user import User
from app.schemas.validation import (
    Validation,
    ValidationCreate,
    ValidationList,
    ValidationSubmit,
)
from app.services.notification_service import NotificationService
from app.services.user_service import UserService
from app.services.validation_service import ValidationService

router = APIRouter()

@router.post("/", response_model=Validation)
async def create_validation(
    validation_data: ValidationCreate,
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Create validation assignment."""
    if current_user.role not in [UserRoles.ADMIN, UserRoles.CARDIOLOGIST]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to assign validations"
        )

    notification_service = NotificationService(db)
    validation_service = ValidationService(db, notification_service)

    validation = await validation_service.create_validation(
        analysis_id=validation_data.analysis_id,
        validator_id=validation_data.validator_id,
        validator_role=current_user.role,
        validator_experience_years=current_user.experience_years,
    )

    return validation

@router.get("/my-validations", response_model=ValidationList)
async def get_my_validations(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get validations assigned to current user."""
    notification_service = NotificationService(db)
    validation_service = ValidationService(db, notification_service)

    validations = await validation_service.repository.get_validations_by_validator(
        current_user.id, limit, offset
    )

    validations_schemas = [Validation.from_orm(v) for v in validations]
    return ValidationList(
        validations=validations_schemas,
        total=len(validations),  # Simplified
        page=offset // limit + 1,
        size=limit,
    )

@router.post("/{validation_id}/submit", response_model=Validation)
async def submit_validation(
    validation_id: int,
    validation_data: ValidationSubmit,
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Submit validation results."""
    notification_service = NotificationService(db)
    validation_service = ValidationService(db, notification_service)

    validation = await validation_service.submit_validation(
        validation_id=validation_id,
        validator_id=current_user.id,
        validation_data=validation_data.dict(),
    )

    return validation

@router.get("/{validation_id}", response_model=Validation)
async def get_validation(
    validation_id: int,
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get validation by ID."""
    notification_service = NotificationService(db)
    validation_service = ValidationService(db, notification_service)

    validation = await validation_service.repository.get_validation_by_id(validation_id)
    if not validation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Validation not found"
        )

    if (
        not current_user.is_superuser and
        validation.validator_id != current_user.id and
        validation.analysis.created_by != current_user.id
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this validation"
        )

    return validation

@router.get("/pending/critical", response_model=list[Validation])
async def get_pending_critical_validations(
    limit: int = 20,
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get pending critical validations."""
    if not current_user.is_physician:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    notification_service = NotificationService(db)
    validation_service = ValidationService(db, notification_service)

    from app.services.ecg_service import ECGAnalysisService
    from app.services.ml_model_service import MLModelService

    ml_service = MLModelService()
    ecg_service = ECGAnalysisService(db, ml_service, validation_service)

    critical_analyses = await ecg_service.repository.get_critical_analyses(limit)

    validations = []
    for analysis in critical_analyses:
        existing_validation = await validation_service.repository.get_validation_by_analysis(
            analysis.id
        )
        if existing_validation:
            validations.append(existing_validation)

    return validations
