import asyncio
import logging
from typing import Any

from app.db.session import get_session_factory
from app.services.ecg_service import ECGAnalysisService
from app.services.ml_model_service import MLModelService
from app.services.validation_service import ValidationService

logger = logging.getLogger(__name__)

async def process_ecg_analysis_sync(analysis_id: int) -> dict[str, Any]:
    """Process ECG analysis synchronously (converted from Celery task)"""
    try:
        logger.info(f"Starting ECG analysis {analysis_id}")
        
        session_factory = get_session_factory()
        async with session_factory() as db:
            ml_service = MLModelService()
            from app.services.notification_service import NotificationService
            notification_service = NotificationService(db)
            validation_service = ValidationService(db, notification_service)
            service = ECGAnalysisService(db, ml_service, validation_service)
            
            await service._process_analysis_async(analysis_id)
            
            logger.info(f"ECG analysis {analysis_id} completed successfully")
            return {"status": "completed", "analysis_id": analysis_id}

    except Exception as exc:
        logger.error("ECG analysis failed: %s", exc)
        raise
