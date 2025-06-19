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

from app.schemas.ecg_analysis import (
    ECGAnalysisResponse, 
    ECGAnalysisUpdate
)
from app.services.ecg_service import ECGAnalysisService
from app.repositories.ecg_repository import ECGRepository
from app.services.patient_service import PatientService
from app.repositories.patient_repository import PatientRepository
from app.services.ml_model_service import MLModelService
from app.services.notification_service import NotificationService
from app.repositories.notification_repository import NotificationRepository
from app.services.interpretability_service import InterpretabilityService
from app.services.multi_pathology_service import MultiPathologyService
from app.core.constants import FileType

router = APIRouter(prefix="/ecg", tags=["ecg_analysis"])

@router.post("/analyses", response_model=ECGAnalysisResponse)
async def create_analysis(
    analysis_data: dict,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new ECG analysis"""
    # Initialize all services
    ecg_repo = ECGRepository(db)
    patient_repo = PatientRepository(db)
    patient_service = PatientService(db, patient_repo)
    notification_repo = NotificationRepository(db)
    notification_service = NotificationService(db, notification_repo)
    ml_service = MLModelService()
    interpretability_service = InterpretabilityService()
    multi_pathology_service = MultiPathologyService(ml_service)
    
    ecg_service = ECGAnalysisService(
        db=db,
        ecg_repository=ecg_repo,
        patient_service=patient_service,
        ml_service=ml_service,
        notification_service=notification_service,
        interpretability_service=interpretability_service,
        multi_pathology_service=multi_pathology_service
    )
    
    analysis = await ecg_service.create_analysis(analysis_data, current_user.id)
    return analysis

@router.post("/upload", response_model=ECGAnalysisResponse)
async def upload_ecg_file(
    patient_id: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Upload and analyze ECG file"""
    # Save uploaded file
    file_path = f"/tmp/{file.filename}"
    content = await file.read()
    
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Determine file type
    file_extension = file.filename.split('.')[-1].lower()
    file_type_map = {
        'csv': FileType.CSV,
        'edf': FileType.EDF,
        'dat': FileType.MIT,
        'dcm': FileType.DICOM,
        'json': FileType.JSON
    }
    file_type = file_type_map.get(file_extension, FileType.OTHER)
    
    # Create analysis
    analysis_data = dict(
        patient_id=patient_id,
        recording_date=datetime.utcnow(),
        file_path=file_path,
        file_type=file_type
    )
    
    return await create_analysis(analysis_data, current_user, db)

@router.get("/analyses", response_model=List[ECGAnalysisResponse])
async def search_analyses(
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Search ECG analyses"""
    ecg_repo = ECGRepository(db)
    analyses = await ecg_repo.get_all_analyses(limit=100)
    return analyses

@router.get("/analyses/{analysis_id}", response_model=ECGAnalysisResponse)
async def get_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get ECG analysis by ID"""
    ecg_repo = ECGRepository(db)
    analysis = await ecg_repo.get_analysis_by_analysis_id(analysis_id)
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis

@router.get("/analyses/{analysis_id}/report")
async def get_analysis_report(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Generate analysis report"""
    # Initialize services
    ecg_repo = ECGRepository(db)
    patient_repo = PatientRepository(db)
    patient_service = PatientService(db, patient_repo)
    notification_repo = NotificationRepository(db)
    notification_service = NotificationService(db, notification_repo)
    ml_service = MLModelService()
    interpretability_service = InterpretabilityService()
    multi_pathology_service = MultiPathologyService(ml_service)
    
    ecg_service = ECGAnalysisService(
        db=db,
        ecg_repository=ecg_repo,
        patient_service=patient_service,
        ml_service=ml_service,
        notification_service=notification_service,
        interpretability_service=interpretability_service,
        multi_pathology_service=multi_pathology_service
    )
    
    report = await ecg_service.generate_report(analysis_id)
    return report

@router.delete("/analyses/{analysis_id}")
async def delete_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete ECG analysis"""
    ecg_repo = ECGRepository(db)
    await ecg_repo.delete_analysis(analysis_id)
    return {"message": "Analysis deleted successfully"}
