import asyncio
import logging
from typing import Any

from app.core.config import settings
from app.db.session import get_session_factory

# from app.repositories.ecg_repository import ECGRepository  # Reserved for future use
from app.services.ecg_service import ECGAnalysisService
from app.services.ml_model_service import MLModelService
from app.services.validation_service import ValidationService

logger = logging.getLogger(__name__)

if not settings.STANDALONE_MODE:
    try:
        from celery import current_task
        from app.core.celery import celery_app
        CELERY_AVAILABLE = True
    except ImportError:
        CELERY_AVAILABLE = False
else:
    CELERY_AVAILABLE = False

def process_ecg_analysis_sync(analysis_id: int) -> dict[str, Any]:
    """Process ECG analysis synchronously for standalone mode"""
    try:
        async def _process() -> dict[str, Any]:
            session_factory = get_session_factory()
            async with session_factory() as db:
                # ecg_repo = ECGRepository(db)  # Reserved for future use
                ml_service = MLModelService()
                from app.services.notification_service import NotificationService
                notification_service = NotificationService(db)
                validation_service = ValidationService(db, notification_service)
                service = ECGAnalysisService(db, ml_service, validation_service)
                await service._process_analysis_async(analysis_id)
                return {"status": "completed", "analysis_id": analysis_id}

        result = asyncio.run(_process())
        logger.info(f"ECG analysis completed synchronously for analysis_id={analysis_id}")
        return result

    except Exception as exc:
        logger.error("ECG analysis failed: %s", exc)
        raise

if CELERY_AVAILABLE:
    @celery_app.task(bind=True)
    def process_ecg_analysis(self: Any, analysis_id: int) -> dict[str, Any]:
        """Process ECG analysis in background"""
        try:
            current_task.update_state(
                state="PROGRESS",
                meta={"current": 0, "total": 100, "status": "Starting analysis..."}
            )

            result = process_ecg_analysis_sync(analysis_id)

            current_task.update_state(
                state="SUCCESS",
                meta={"current": 100, "total": 100, "status": "Analysis complete", "result": result}
            )

            return result

        except Exception as exc:
            logger.error("ECG analysis task failed: %s", exc)
            current_task.update_state(
                state="FAILURE",
                meta={"current": 0, "total": 100, "status": str(exc)}
            )
            raise
