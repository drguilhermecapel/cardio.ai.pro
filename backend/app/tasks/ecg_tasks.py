import logging

from celery import current_task

from app.core.celery import celery_app
from app.db.session import get_db
from app.services.ecg_service import ECGAnalysisService

logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
def process_ecg_analysis(self, analysis_id: int):
    """Process ECG analysis in background"""
    try:
        db = next(get_db())
        service = ECGAnalysisService(db)

        current_task.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 100, "status": "Starting analysis..."}
        )

        result = service.process_analysis(analysis_id)

        current_task.update_state(
            state="SUCCESS",
            meta={"current": 100, "total": 100, "status": "Analysis complete", "result": result}
        )

        return result

    except Exception as exc:
        logger.error(f"ECG analysis task failed: {exc}")
        current_task.update_state(
            state="FAILURE",
            meta={"current": 0, "total": 100, "status": str(exc)}
        )
        raise
