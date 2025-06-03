"""
CardioAI Pro - Main FastAPI Application
Enterprise ECG Analysis System with AI
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.v1.api import api_router
from app.core.config import settings
from app.core.exceptions import (
    AuthenticationException,
    CardioAIException,
    PermissionDeniedException,
    ValidationException,
)
from app.core.logging import configure_logging
from app.db.session import get_engine
from app.monitoring.health_checks import router as health_router
from app.websockets.ecg_streaming import handle_ecg_websocket, streaming_manager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    configure_logging()
    logger = structlog.get_logger()
    logger.info("Starting CardioAI Pro Backend", version="1.0.0")

    if settings.ENVIRONMENT != "test":
        try:
            engine = get_engine()
            async with engine.begin() as conn:
                from sqlalchemy import text
                await conn.execute(text("SELECT 1"))
            logger.info("Database connection established")
        except Exception as e:
            logger.error("Failed to connect to database", error=str(e))
            raise
    else:
        logger.info("Skipping database connection in test environment")

    yield

    logger.info("Shutting down CardioAI Pro Backend")
    if settings.ENVIRONMENT != "test":
        try:
            engine = get_engine()
            await engine.dispose()
        except Exception:
            pass  # Ignore disposal errors in shutdown


app = FastAPI(
    title="CardioAI Pro API",
    description="Enterprise ECG Analysis System with AI",
    version="1.0.0",
    docs_url="/api/v1/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/api/v1/redoc" if settings.ENVIRONMENT != "production" else None,
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS,
)

if settings.ENABLE_METRICS:
    instrumentator = Instrumentator()
    instrumentator.instrument(app).expose(app)


@app.exception_handler(CardioAIException)
async def cardioai_exception_handler(request: Request, exc: CardioAIException) -> JSONResponse:
    """Handle CardioAI custom exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(ValidationException)
async def validation_exception_handler(request: Request, exc: ValidationException) -> JSONResponse:
    """Handle validation exceptions."""
    return JSONResponse(
        status_code=422,
        content={
            "error": "VALIDATION_ERROR",
            "message": "Validation failed",
            "details": exc.errors,
        },
    )


@app.exception_handler(AuthenticationException)
async def auth_exception_handler(request: Request, exc: AuthenticationException) -> JSONResponse:
    """Handle authentication exceptions."""
    return JSONResponse(
        status_code=401,
        content={
            "error": "AUTHENTICATION_ERROR",
            "message": str(exc),
        },
    )


@app.exception_handler(PermissionDeniedException)
async def permission_exception_handler(request: Request, exc: PermissionDeniedException) -> JSONResponse:
    """Handle permission exceptions."""
    return JSONResponse(
        status_code=403,
        content={
            "error": "PERMISSION_DENIED",
            "message": str(exc),
        },
    )


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "cardioai-pro-api"}


app.include_router(api_router, prefix="/api/v1")
app.include_router(health_router, prefix="/monitoring", tags=["monitoring"])


@app.websocket("/ws/ecg/{session_id}")
async def websocket_ecg_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time ECG streaming"""
    await handle_ecg_websocket(websocket, session_id)


@app.get("/streaming/stats")
async def get_streaming_stats():
    """Get real-time streaming statistics"""
    return await streaming_manager.get_all_sessions_stats()


@app.post("/streaming/cleanup")
async def cleanup_inactive_sessions(timeout_seconds: int = 300):
    """Clean up inactive streaming sessions"""
    cleaned_count = await streaming_manager.cleanup_inactive_sessions(timeout_seconds)
    return {"cleaned_sessions": cleaned_count}


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "message": "CardioAI Pro API",
        "version": "1.0.0",
        "docs": "/api/v1/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower(),
    )
