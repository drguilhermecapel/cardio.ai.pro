"""
ECG Analysis endpoints.
"""

import os
import uuid
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import NonECGImageException
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
from app.services.adaptive_feedback_service import adaptive_feedback_service

router = APIRouter()


@router.post("/upload", response_model=ECGUploadResponse)
async def upload_ecg(
    patient_id: int = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Upload ECG file for analysis."""
    allowed_extensions = {'.csv', '.txt', '.xml', '.dat', '.jpg', '.jpeg', '.png'}
    file_extension = os.path.splitext(file.filename or "")[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )

    if file.size and file.size > settings.MAX_ECG_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.MAX_ECG_FILE_SIZE} bytes"
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

    document_scanning_metadata = None
    estimated_time = 30
    
    if file_extension in {'.jpg', '.jpeg', '.png'}:
        try:
            from app.services.ecg_document_scanner import ECGDocumentScanner
            scanner = ECGDocumentScanner()
            
            user_session = None
            try:
                from app.models.user import UserSession
                from sqlalchemy import select
                stmt = select(UserSession).where(UserSession.user_id == current_user.id).order_by(UserSession.created_at.desc())
                result = await db.execute(stmt)
                user_session = result.scalar_one_or_none()
            except Exception:
                pass  # Continue without user session if not available
            
            scan_result = await scanner.process_ecg_image(
                file_path, 
                user_session=user_session,
                raise_non_ecg_exception=True  # This will raise NonECGImageException for non-ECG images
            )
            
            document_scanning_metadata = {
                "scanner_confidence": scan_result.get("confidence", 0.0),
                "document_detected": scan_result.get("validation", {}).get("is_valid_ecg", False),
                "processing_method": scan_result.get("metadata", {}).get("processing_method", "unknown"),
                "grid_detected": scan_result.get("validation", {}).get("grid_detected", False),
                "leads_detected": scan_result.get("validation", {}).get("leads_detected", 0),
                "original_size": scan_result.get("metadata", {}).get("original_size", [0, 0]),
                "processed_size": scan_result.get("metadata", {}).get("processed_size", [0, 0])
            }
            
            estimated_time = 60 if scan_result.get("confidence", 0.0) > 0.5 else 45
            
        except NonECGImageException as e:
            if user_session:
                await adaptive_feedback_service.track_user_attempt(
                    user_session=user_session,
                    category=e.details["category"],
                    success=False,
                    confidence=e.details["confidence"]
                )
            
            try:
                os.unlink(file_path)
            except Exception:
                pass  # File might already be deleted
            
            raise HTTPException(
                status_code=e.status_code,
                detail={
                    "error_code": e.error_code,
                    "message": e.message,
                    "category": e.details["category"],
                    "confidence": e.details["confidence"],
                    "contextual_response": e.details["contextual_response"]
                }
            )
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Image preprocessing failed for {file_path}: {str(e)}")
            
            document_scanning_metadata = {
                "scanner_confidence": 0.0,
                "document_detected": False,
                "processing_method": "fallback",
                "error": str(e)
            }

    analysis = await ecg_service.create_analysis(
        patient_id=patient_id,
        file_path=file_path,
        original_filename=file.filename or "unknown",
        created_by=current_user.id,
    )

    if file_extension in {'.jpg', '.jpeg', '.png'} and user_session:
        await adaptive_feedback_service.track_user_attempt(
            user_session=user_session,
            category="ecg_success",
            success=True,
            confidence=document_scanning_metadata.get("scanner_confidence", 0.0) if document_scanning_metadata else 0.0
        )

    if file_extension in {'.jpg', '.jpeg', '.png'}:
        if document_scanning_metadata and document_scanning_metadata.get("document_detected", False):
            message = "ECG image uploaded and document detected successfully. Analysis started."
        else:
            message = "Image uploaded. Document detection had low confidence. Analysis started with fallback processing."
    else:
        message = "ECG uploaded successfully. Analysis started."

    return ECGUploadResponse(
        analysis_id=analysis.analysis_id,
        message=message,
        status=analysis.status,
        estimated_processing_time_seconds=estimated_time,
        file_type=file_extension,
        document_scanning_metadata=document_scanning_metadata,
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
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )

    if not current_user.is_superuser and analysis.created_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this analysis"
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
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )

    if not current_user.is_superuser and analysis.created_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this analysis"
        )

    measurements = await ecg_service.repository.get_measurements_by_analysis(analysis.id)
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
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )

    if not current_user.is_superuser and analysis.created_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this analysis"
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
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )

    if not current_user.is_superuser and analysis.created_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this analysis"
        )

    success = await ecg_service.delete_analysis(analysis.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete analysis"
        )

    return {"message": "Analysis deleted successfully"}


@router.get("/critical/pending", response_model=list[ECGAnalysis])
async def get_critical_pending(
    limit: int = 20,
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get critical analyses pending validation."""
    if not current_user.is_physician:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    ml_service = MLModelService()
    validation_service = ValidationService(db, NotificationService(db))
    ecg_service = ECGAnalysisService(db, ml_service, validation_service)

    critical_analyses = await ecg_service.repository.get_critical_analyses(limit)
    return critical_analyses
