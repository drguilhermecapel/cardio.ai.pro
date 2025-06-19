"""
Logging configuration for CardioAI Pro.
"""

import logging
import sys
import json
from datetime import datetime
from typing import Any

import structlog
from structlog.types import Processor

from app.core.config import settings


def configure_logging() -> None:
    """Configure structured logging."""

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL.upper()),
    )

    processors: list[Processor] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.ENVIRONMENT == "development":
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    """Get a structured logger."""
    return structlog.get_logger(name)


class AuditLogger:
    """Audit logger for regulatory compliance."""

    def __init__(self) -> None:
        self.logger = get_logger("audit")

    def log_user_action(
        self,
        user_id: int,
        action: str,
        resource_type: str,
        resource_id: str,
        details: dict[str, Any],
        ip_address: str,
        user_agent: str,
    ) -> None:
        """Log user action for audit trail."""
        self.logger.info(
            "User action",
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
            audit=True,
        )

    def log_system_event(
        self,
        event_type: str,
        description: str,
        details: dict[str, Any],
    ) -> None:
        """Log system event for audit trail."""
        self.logger.info(
            "System event",
            event_type=event_type,
            description=description,
            details=details,
            audit=True,
        )

    def log_data_access(self, access_type, resource_type=None, resource_id=None, user_id=None, **kwargs) -> None:
        """Log data access event for audit trail and HIPAA compliance."""
        self._log_audit_event(
            event_type="DATA_ACCESS",
            user_id=user_id,
            details={
                "resource_type": resource_type,
                "resource_id": resource_id,
                "action": action,
                "ip_address": ip_address,
                "additional_context": additional_context or {},
            },
        )

    def log_medical_action(
        self,
        user_id: int,
        action_type: str,
        patient_id: int | None = None,
        analysis_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log medical action for compliance."""
        self._log_audit_event(
            event_type="MEDICAL_ACTION",
            user_id=user_id,
            details={
                "action_type": action_type,
                "patient_id": patient_id,
                "analysis_id": analysis_id,
                "details": details or {},
            },
        )

    def _log_audit_event(
        self,
        event_type: str,
        user_id: int,
        details: dict[str, Any],
    ) -> None:
        """Internal method to log audit events."""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "details": details,
        }
        
        # Log to structured logger
        self.logger.info(
            f"AUDIT: {event_type}",
            extra={"audit_data": json.dumps(audit_entry)},
            audit=True,
            user_id=user_id,
            event_type=event_type,
            **details,
        )

    def log_authentication(
        self,
        user_id: int | None,
        action: str,
        success: bool,
        ip_address: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log authentication events."""
        self.logger.info(
            "Authentication event",
            user_id=user_id,
            action=action,
            success=success,
            ip_address=ip_address,
            details=details or {},
            audit=True,
        )

    def log_ecg_analysis(
        self,
        analysis_id: str,
        user_id: int,
        patient_id: int,
        processing_time: float,
        ai_results: dict[str, Any],
        clinical_urgency: str,
    ) -> None:
        """Log ECG analysis for audit trail."""
        self.logger.info(
            "ECG analysis",
            analysis_id=analysis_id,
            user_id=user_id,
            patient_id=patient_id,
            processing_time=processing_time,
            ai_confidence=ai_results.get("confidence", 0.0),
            clinical_urgency=clinical_urgency,
            audit=True,
        )

    def log_validation(
        self,
        validation_id: int,
        analysis_id: str,
        validator_id: int,
        status: str,
        agrees_with_ai: bool | None,
    ) -> None:
        """Log validation events."""
        self.logger.info(
            "Validation event",
            validation_id=validation_id,
            analysis_id=analysis_id,
            validator_id=validator_id,
            status=status,
            agrees_with_ai=agrees_with_ai,
            audit=True,
        )

    def log_error(
        self,
        error_type: str,
        error_message: str,
        user_id: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Log error events."""
        self.logger.error(
            "Error event",
            error_type=error_type,
            error_message=error_message,
            user_id=user_id,
            context=context or {},
            audit=True,
        )


# Create global audit logger instance
audit_logger = AuditLogger()
