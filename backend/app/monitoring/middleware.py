"""
Custom middleware for ECG analysis monitoring
"""

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.monitoring.structured_logging import get_ecg_logger

logger = get_ecg_logger(__name__)


class ECGMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect ECG-specific metrics"""

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()
        path = request.url.path
        method = request.method

        try:
            response = await call_next(request)
            duration = time.time() - start_time

            logger.logger.info(
                "http_request_completed",
                method=method,
                path=path,
                status_code=response.status_code,
                duration_seconds=duration
            )

            return response

        except Exception as e:
            duration = time.time() - start_time

            logger.logger.error(
                "http_request_failed",
                method=method,
                path=path,
                error_type=type(e).__name__,
                error_message=str(e),
                duration_seconds=duration
            )

            raise
