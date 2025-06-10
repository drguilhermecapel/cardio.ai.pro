"""
Avatar generation API endpoints.
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.avatar_generator import generate_avatar

logger = logging.getLogger(__name__)

router = APIRouter()


class AvatarRequest(BaseModel):
    """Request model for avatar generation."""

    prompt: str | None = Field(
        default=None,
        description="Custom prompt for avatar generation. Uses default if not provided."
    )
    seed: int | None = Field(
        default=42,
        description="Random seed for reproducible generation",
        ge=0
    )
    steps: int = Field(
        default=30,
        description="Number of inference steps",
        ge=1,
        le=100
    )
    height: int = Field(
        default=768,
        description="Image height in pixels",
        ge=256,
        le=1024
    )
    width: int = Field(
        default=512,
        description="Image width in pixels",
        ge=256,
        le=1024
    )
    model_name: str = Field(
        default="runwayml/stable-diffusion-v1-5",
        description="Hugging Face model identifier"
    )


class AvatarResponse(BaseModel):
    """Response model for avatar generation."""

    file_path: str = Field(description="Absolute path to the generated avatar image")
    filename: str = Field(description="Name of the generated file")
    size_bytes: int = Field(description="File size in bytes")
    generation_time_seconds: float = Field(description="Time taken to generate the avatar")


@router.post("/", response_model=AvatarResponse)
async def create_avatar(request: AvatarRequest) -> AvatarResponse:
    """
    Generate a hyper-realistic avatar using Stable Diffusion.

    This endpoint creates a photorealistic portrait of a 50-year-old caucasian woman
    with glasses using the specified parameters. The generation process is optimized
    for minimal VRAM usage with GPU acceleration when available.

    Args:
        request: Avatar generation parameters

    Returns:
        Information about the generated avatar file

    Raises:
        HTTPException: If avatar generation fails
    """
    import time

    start_time = time.time()

    try:
        logger.info(f"Starting avatar generation with seed={request.seed}")

        file_path = generate_avatar(
            prompt_override=request.prompt,
            seed=request.seed or 42,
            steps=request.steps,
            height=request.height,
            width=request.width,
            model_name=request.model_name,
        )

        generation_time = time.time() - start_time
        file_size = file_path.stat().st_size

        logger.info(
            f"Avatar generated successfully: {file_path.name} "
            f"({file_size} bytes, {generation_time:.2f}s)"
        )

        return AvatarResponse(
            file_path=str(file_path.absolute()),
            filename=file_path.name,
            size_bytes=file_size,
            generation_time_seconds=round(generation_time, 2)
        )

    except Exception as e:
        logger.error(f"Avatar generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate avatar: {str(e)}"
        ) from e


@router.get("/health")
async def avatar_health_check() -> dict[str, str]:
    """
    Health check endpoint for avatar generation service.

    Returns:
        Service status information
    """
    return {
        "status": "healthy",
        "service": "avatar-generation",
        "version": "1.0.0"
    }
