"""
Logging configuration for CardioAI Pro.
"""

import logging
import sys
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


def get_logger(name: str) -> structlog.BoundLogger:
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

    def log_data_access(
        self,
        user_id: int,
        data_type: str,
        data_id: str,
        access_type: str,
        ip_address: str,
    ) -> None:
        """Log data access for compliance."""
        self.logger.info(
            "Data access",
            user_id=user_id,
            data_type=data_type,
            data_id=data_id,
            access_type=access_type,
            ip_address=ip_address,
            audit=True,
        )


audit_logger = AuditLogger()
