import asyncio
import logging
from typing import Any

from celery import current_task

from app.core.celery import celery_app
from app.db.session import async_sessionmaker
from app.repositories.ecg_repository import ECGAnalysisRepository
from app.services.ecg_service import ECGAnalysisService
from app.services.ml_service import MLService
from app.services.validation_service import ValidationService

logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
def process_ecg_analysis(self, analysis_id: int) -> dict[str, Any]:
    """Process ECG analysis in background"""
    try:
        current_task.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 100, "status": "Starting analysis..."}
        )

        async def _process() -> dict[str, Any]:
            async with async_sessionmaker() as db:
                ecg_repo = ECGAnalysisRepository(db)
                ml_service = MLService()
                validation_service = ValidationService(db)
                service = ECGAnalysisService(db, ml_service, validation_service)
                return await service._process_analysis_async(analysis_id)

        result = asyncio.run(_process())

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
