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

from app.schemas.patient import PatientCreate, Patient, PatientUpdate
from app.services.patient_service import PatientService
from app.repositories.patient_repository import PatientRepository

router = APIRouter(prefix="/patients", tags=["patients"])

@router.post("/", response_model=Patient)
async def create_patient(
    patient_data: PatientCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new patient"""
    patient_repo = PatientRepository(db)
    patient_service = PatientService(db, patient_repo)
    patient = await patient_service.create_patient(patient_data)
    return patient

@router.get("/", response_model=List[Patient])
async def get_patients(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    search: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get list of patients"""
    patient_repo = PatientRepository(db)
    patient_service = PatientService(db, patient_repo)
    
    if search:
        patients = await patient_service.search_patients(search, skip, limit)
    else:
        patients = await patient_service.get_patients(skip, limit)
    
    return patients

@router.get("/{patient_id}", response_model=Patient)
async def get_patient(
    patient_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get patient by ID"""
    patient_repo = PatientRepository(db)
    patient_service = PatientService(db, patient_repo)
    patient = await patient_service.get_patient_by_patient_id(patient_id)
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    return patient

@router.put("/{patient_id}", response_model=Patient)
async def update_patient(
    patient_id: str,
    patient_update: PatientUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update patient information"""
    patient_repo = PatientRepository(db)
    patient_service = PatientService(db, patient_repo)
    patient = await patient_service.update_patient(patient_id, patient_update)
    return patient

@router.delete("/{patient_id}")
async def delete_patient(
    patient_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete patient"""
    patient_repo = PatientRepository(db)
    await patient_repo.delete_patient(patient_id)
    return {"message": "Patient deleted successfully"}
