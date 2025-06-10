"""
CardioAI Pro - Standalone FastAPI Application
Simplified ECG Analysis System for Desktop
"""

import sys
from pathlib import Path

if getattr(sys, 'frozen', False):
    bundle_dir = Path(getattr(sys, '_MEIPASS', ''))
    app_dir = bundle_dir / 'app'
    if app_dir.exists():
        sys.path.insert(0, str(app_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.api import api_router


# Simplified startup for standalone version
def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="CardioAI Pro API",
        description="Standalone ECG Analysis System",
        version="1.0.0",
        docs_url="/api/v1/docs",
        redoc_url="/api/v1/redoc",
        openapi_url="/api/v1/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "*",  # Allow all origins for standalone
            "https://emr-installer-app-vxljcvay.devinapps.com",  # Deployed frontend
            "http://localhost:5173",  # Local development
            "http://localhost:3000",  # Alternative local port
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api/v1")

    return app


app = create_app()


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "cardioai-pro-standalone"}


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "message": "CardioAI Pro Standalone API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,  # Pass the app object directly instead of string reference
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disabled for standalone
        log_level="info",
    )
