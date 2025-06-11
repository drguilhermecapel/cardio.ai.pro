"""
Patient management endpoints.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.user import User
from app.schemas.patient import (
    Patient,
    PatientCreate,
    PatientList,
    PatientSearch,
    PatientUpdate,
)
from app.services.patient_service import PatientService
from app.services.user_service import UserService

router = APIRouter()

@router.post("/", response_model=Patient)
async def create_patient(
    patient_data: PatientCreate,
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Create new patient."""
    patient_service = PatientService(db)

    existing = await patient_service.get_patient_by_patient_id(patient_data.patient_id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Patient ID already exists"
        )

    patient = await patient_service.create_patient(patient_data, current_user.id)
    return patient

@router.get("/{patient_id}", response_model=Patient)
async def get_patient(
    patient_id: str,
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get patient by patient ID."""
    patient_service = PatientService(db)

    patient = await patient_service.get_patient_by_patient_id(patient_id)
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )

    return patient

@router.put("/{patient_id}", response_model=Patient)
async def update_patient(
    patient_id: str,
    patient_update: PatientUpdate,
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Update patient information."""
    patient_service = PatientService(db)

    patient = await patient_service.get_patient_by_patient_id(patient_id)
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )

    update_data = patient_update.dict(exclude_unset=True)
    updated_patient = await patient_service.update_patient(patient.id, update_data)

    return updated_patient

@router.get("/", response_model=PatientList)
async def list_patients(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """List patients."""
    patient_service = PatientService(db)

    patients, total = await patient_service.get_patients(limit, offset)

    patients_schemas = [Patient.from_orm(p) for p in patients]
    return PatientList(
        patients=patients_schemas,
        total=total,
        page=offset // limit + 1,
        size=limit,
    )

@router.post("/search", response_model=PatientList)
async def search_patients(
    search_params: PatientSearch,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Search patients."""
    patient_service = PatientService(db)

    patients, total = await patient_service.search_patients(
        search_params.query, search_params.search_fields, limit, offset
    )

    patients_schemas = [Patient.from_orm(p) for p in patients]
    return PatientList(
        patients=patients_schemas,
        total=total,
        page=offset // limit + 1,
        size=limit,
    )

@router.get("/{patient_id}/ecg-analyses")
async def get_patient_ecg_analyses(
    patient_id: str,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get ECG analyses for a specific patient."""
    from app.services.ecg_service import ECGAnalysisService
    from app.services.ml_model_service import MLModelService
    from app.services.notification_service import NotificationService
    from app.services.validation_service import ValidationService

    patient_service = PatientService(db)

    patient = await patient_service.get_patient_by_patient_id(patient_id)
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )

    ml_service = MLModelService()
    validation_service = ValidationService(db, NotificationService(db))
    ecg_service = ECGAnalysisService(db, ml_service, validation_service)

    filters = {"patient_id": patient.id}
    if not current_user.is_superuser:
        filters["created_by"] = current_user.id

    analyses, total = await ecg_service.search_analyses(filters, limit, offset)

    from app.schemas.ecg_analysis import ECGAnalysis, ECGAnalysisList
    analyses_schemas = [ECGAnalysis.from_orm(a) for a in analyses]
    return ECGAnalysisList(
        analyses=analyses_schemas,
        total=total,
        page=offset // limit + 1,
        size=limit,
    )
