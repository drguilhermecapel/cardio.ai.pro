from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ....models.user import User
from ....services.i18n_service import i18n_service
from ....services.user_service import UserService

logger = structlog.get_logger(__name__)

router = APIRouter()


class MedicalValidationRequest(BaseModel):
    term: str
    translation: str
    language: str
    validator_credentials: str


class BatchValidationRequest(BaseModel):
    translations: dict[str, str]
    language: str


class ValidationStatusResponse(BaseModel):
    term: str
    language: str
    is_critical: bool
    has_validated_translation: bool
    severity: str
    validated_translation: str | None
    validation_info: dict[str, Any] | None


@router.get("/status/{term}/{language}", response_model=ValidationStatusResponse)
async def get_validation_status(
    term: str,
    language: str,
    current_user: User = Depends(UserService.get_current_user)
):
    """Get validation status for a specific medical term and language."""
    try:
        status_info = i18n_service.get_medical_validation_status(term, language)
        return ValidationStatusResponse(**status_info)
    except Exception as e:
        logger.error(f"Error getting validation status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get validation status"
        ) from e


@router.get("/pending/{language}")
async def get_pending_validations(
    language: str,
    current_user: User = Depends(UserService.get_current_user)
) -> list[dict[str, Any]]:
    """Get list of medical terms pending validation for a language."""
    try:
        return i18n_service.get_pending_medical_validations(language)
    except Exception as e:
        logger.error(f"Error getting pending validations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get pending validations"
        ) from e


@router.post("/validate")
async def register_medical_validation(
    request: MedicalValidationRequest,
    current_user: User = Depends(UserService.get_current_user)
):
    """Register professional validation for a medical term."""
    try:
        if not hasattr(current_user, 'is_medical_validator') or not current_user.is_medical_validator:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User does not have medical validation privileges"
            )

        i18n_service.register_medical_validation(
            request.term,
            request.translation,
            request.language,
            str(current_user.id),
            request.validator_credentials
        )

        logger.info(f"Medical validation registered for term '{request.term}' by user {current_user.id}")

        return {"message": "Medical validation registered successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering medical validation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register medical validation"
        ) from e


@router.post("/validate-batch")
async def validate_translation_batch(
    request: BatchValidationRequest,
    current_user: User = Depends(UserService.get_current_user)
) -> dict[str, dict[str, Any]]:
    """Validate a batch of medical translations."""
    try:
        results = {}

        for term, translation in request.translations.items():
            is_valid, message, severity = i18n_service.validate_medical_translation(
                term, translation, request.language
            )
            results[term] = {
                "is_valid": is_valid,
                "message": message,
                "severity": severity,
                "translation": translation
            }

        return results

    except Exception as e:
        logger.error(f"Error validating translation batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate translation batch"
        ) from e


@router.get("/report/{language}")
async def get_validation_report(
    language: str,
    current_user: User = Depends(UserService.get_current_user)
) -> dict[str, Any]:
    """Get comprehensive medical validation report for a language."""
    try:
        return i18n_service.export_medical_validation_report(language)
    except Exception as e:
        logger.error(f"Error generating validation report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate validation report"
        ) from e


@router.get("/languages")
async def get_supported_languages(
    current_user: User = Depends(UserService.get_current_user)
) -> list[str]:
    """Get list of supported languages for medical validation."""
    try:
        return i18n_service.get_available_languages()
    except Exception as e:
        logger.error(f"Error getting supported languages: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get supported languages"
        ) from e
