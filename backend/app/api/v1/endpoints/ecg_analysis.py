"""
ECG Analysis endpoints.
"""

import os
import uuid
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_db
from app.models.user import User
from app.schemas.ecg_analysis import (
    ECGAnalysis,
    ECGAnalysisList,
    ECGAnalysisSearch,
    ECGAnnotation,
    ECGMeasurement,
    ECGUploadResponse,
)
from app.services.ecg_service import ECGAnalysisService
from app.services.ml_model_service import MLModelService
from app.services.notification_service import NotificationService
from app.services.user_service import UserService
from app.services.validation_service import ValidationService
from app.tasks.ecg_tasks import process_ecg_analysis_sync

router = APIRouter()


@router.post("/upload", response_model=ECGUploadResponse)
async def upload_ecg(
    patient_id: int = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Upload ECG file for analysis."""
    allowed_extensions = {".csv", ".txt", ".xml", ".dat", ".png", ".jpg", ".jpeg"}
    file_extension = os.path.splitext(file.filename or "")[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}",
        )

    if file_extension in {".png", ".jpg", ".jpeg"}:
        try:
            from PIL import Image

            with Image.open(file.file) as img:
                img.verify()
            await file.seek(0)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image file: {str(e)}",
            ) from e

    if file.size and file.size > settings.MAX_ECG_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.MAX_ECG_FILE_SIZE} bytes",
        )

    file_id = str(uuid.uuid4())
    file_path = os.path.join(settings.ECG_UPLOAD_DIR, f"{file_id}{file_extension}")

    os.makedirs(settings.ECG_UPLOAD_DIR, exist_ok=True)

    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    ml_service = MLModelService()
    validation_service = ValidationService(db, NotificationService(db))
    ecg_service = ECGAnalysisService(db, ml_service, validation_service)

    analysis = await ecg_service.create_analysis(
        patient_id=patient_id,
        file_path=file_path,
        original_filename=file.filename or "unknown",
        created_by=current_user.id,
    )

    try:
        await process_ecg_analysis_sync(analysis.id)
        message = "ECG uploaded and analyzed successfully."
    except Exception as e:
        message = f"ECG uploaded but analysis failed: {str(e)}"

    return ECGUploadResponse(
        analysis_id=analysis.analysis_id,
        message=message,
        status=analysis.status,
        estimated_processing_time_seconds=0,
    )


@router.get("/{analysis_id}", response_model=ECGAnalysis)
async def get_analysis(
    analysis_id: str,
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get ECG analysis by ID."""
    ml_service = MLModelService()
    validation_service = ValidationService(db, NotificationService(db))
    ecg_service = ECGAnalysisService(db, ml_service, validation_service)

    analysis = await ecg_service.repository.get_analysis_by_analysis_id(analysis_id)
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found"
        )

    if not current_user.is_superuser and analysis.created_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this analysis",
        )

    return analysis


@router.get("/", response_model=ECGAnalysisList)
async def list_analyses(
    patient_id: int | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """List ECG analyses."""
    ml_service = MLModelService()
    validation_service = ValidationService(db, NotificationService(db))
    ecg_service = ECGAnalysisService(db, ml_service, validation_service)

    filters: dict[str, Any] = {}
    if patient_id:
        filters["patient_id"] = patient_id
    if status:
        filters["status"] = status

    if not current_user.is_superuser:
        filters["created_by"] = current_user.id

    analyses, total = await ecg_service.search_analyses(filters, limit, offset)

    analyses_schemas = [ECGAnalysis.from_orm(a) for a in analyses]
    return ECGAnalysisList(
        analyses=analyses_schemas,
        total=total,
        page=offset // limit + 1,
        size=limit,
    )


@router.post("/search", response_model=ECGAnalysisList)
async def search_analyses(
    search_params: ECGAnalysisSearch,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Search ECG analyses."""
    ml_service = MLModelService()
    validation_service = ValidationService(db, NotificationService(db))
    ecg_service = ECGAnalysisService(db, ml_service, validation_service)

    filters: dict[str, Any] = {}
    if search_params.patient_id:
        filters["patient_id"] = search_params.patient_id
    if search_params.status:
        filters["status"] = search_params.status.value
    if search_params.clinical_urgency:
        filters["clinical_urgency"] = search_params.clinical_urgency.value
    if search_params.diagnosis_category:
        filters["diagnosis_category"] = search_params.diagnosis_category.value
    if search_params.date_from:
        filters["date_from"] = search_params.date_from.isoformat()
    if search_params.date_to:
        filters["date_to"] = search_params.date_to.isoformat()
    if search_params.is_validated is not None:
        filters["is_validated"] = search_params.is_validated
    if search_params.requires_validation is not None:
        filters["requires_validation"] = search_params.requires_validation

    if not current_user.is_superuser:
        filters["created_by"] = current_user.id

    analyses, total = await ecg_service.search_analyses(filters, limit, offset)

    analyses_schemas = [ECGAnalysis.from_orm(a) for a in analyses]
    return ECGAnalysisList(
        analyses=analyses_schemas,
        total=total,
        page=offset // limit + 1,
        size=limit,
    )


@router.get("/{analysis_id}/measurements", response_model=list[ECGMeasurement])
async def get_measurements(
    analysis_id: str,
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get ECG measurements for analysis."""
    ml_service = MLModelService()
    validation_service = ValidationService(db, NotificationService(db))
    ecg_service = ECGAnalysisService(db, ml_service, validation_service)

    analysis = await ecg_service.repository.get_analysis_by_analysis_id(analysis_id)
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found"
        )

    if not current_user.is_superuser and analysis.created_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this analysis",
        )

    measurements = await ecg_service.repository.get_measurements_by_analysis(
        analysis.id
    )
    return measurements


@router.get("/{analysis_id}/annotations", response_model=list[ECGAnnotation])
async def get_annotations(
    analysis_id: str,
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get ECG annotations for analysis."""
    ml_service = MLModelService()
    validation_service = ValidationService(db, NotificationService(db))
    ecg_service = ECGAnalysisService(db, ml_service, validation_service)

    analysis = await ecg_service.repository.get_analysis_by_analysis_id(analysis_id)
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found"
        )

    if not current_user.is_superuser and analysis.created_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this analysis",
        )

    annotations = await ecg_service.repository.get_annotations_by_analysis(analysis.id)
    return annotations


@router.delete("/{analysis_id}")
async def delete_analysis(
    analysis_id: str,
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Delete ECG analysis."""
    ml_service = MLModelService()
    validation_service = ValidationService(db, NotificationService(db))
    ecg_service = ECGAnalysisService(db, ml_service, validation_service)

    analysis = await ecg_service.repository.get_analysis_by_analysis_id(analysis_id)
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found"
        )

    if not current_user.is_superuser and analysis.created_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this analysis",
        )

    success = await ecg_service.delete_analysis(analysis.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete analysis",
        )

    return {"message": "Analysis deleted successfully"}


@router.post("/{analysis_id}/generate-report")
async def generate_report(
    analysis_id: str,
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Generate medical report for ECG analysis."""
    ml_service = MLModelService()
    validation_service = ValidationService(db, NotificationService(db))
    ecg_service = ECGAnalysisService(db, ml_service, validation_service)

    analysis = await ecg_service.repository.get_analysis_by_analysis_id(analysis_id)
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found"
        )

    if not current_user.is_superuser and analysis.created_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to generate report for this analysis",
        )

    try:
        report = await ecg_service.generate_report(analysis.id)
        return {
            "message": "Report generated successfully",
            "report_id": report.get("report_id"),
            "analysis_id": analysis_id,
            "status": "completed",
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate report: {str(e)}",
        ) from e


@router.get("/critical/pending", response_model=list[ECGAnalysis])
async def get_critical_pending(
    limit: int = 20,
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get critical analyses pending validation."""
    if not current_user.is_physician:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions"
        )

    ml_service = MLModelService()
    validation_service = ValidationService(db, NotificationService(db))
    ecg_service = ECGAnalysisService(db, ml_service, validation_service)

    critical_analyses = await ecg_service.repository.get_critical_analyses(limit)
    return critical_analyses
