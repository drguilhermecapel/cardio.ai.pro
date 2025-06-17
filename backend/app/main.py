from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.api import api_router
from app.core.config import settings
from app.core.logging import configure_logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    configure_logging()
    print(f"🚀 {settings.PROJECT_NAME} v{settings.VERSION} starting up...")
    print(f"📍 Environment: {settings.ENVIRONMENT}")
    print(f"🌐 API docs available at: http://localhost:8000/docs")
    yield
    # Shutdown
    print(f"👋 {settings.PROJECT_NAME} shutting down...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "version": settings.VERSION,
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": settings.PROJECT_NAME}
