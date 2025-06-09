"""
Avatar generation service using Stable Diffusion.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from diffusers import StableDiffusionPipeline

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "ultra-photorealistic portrait of a 50-year-old caucasian woman, "
    "short grayish-blonde hair, wearing modern rectangular eyeglasses, "
    "natural soft studio lighting, 85mm lens, f/1.8, "
    "high-resolution skin texture, subtle makeup, neutral background"
)

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, cartoon, anime, drawing, painting, sketch, watermark, "
    "text, logo, signature, low quality, low resolution, pixelated, "
    "distorted, deformed, ugly, bad anatomy, extra limbs, missing limbs, "
    "duplicate, multiple people, crowd"
)


class AvatarGeneratorService:
    """Service for generating avatar images using Stable Diffusion."""

    def __init__(self) -> None:
        """Initialize the avatar generator service."""
        self.pipeline: StableDiffusionPipeline | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Avatar generator initialized with device: {self.device}")

    def _load_pipeline(self, model_name: str) -> Any:
        """Load the Stable Diffusion pipeline with optimizations."""
        if self.pipeline is not None:
            return self.pipeline

        logger.info(f"Loading Stable Diffusion model: {model_name}")

        try:
            if self.device == "cuda":
                pipeline: Any = StableDiffusionPipeline.from_pretrained(  # type: ignore[no-untyped-call]
                    model_name,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
                pipeline = pipeline.to(self.device)
                pipeline.enable_attention_slicing()
                pipeline.enable_memory_efficient_attention()
                self.pipeline = pipeline
                logger.info("Enabled CUDA optimizations: fp16, attention slicing, memory efficient attention")
            else:
                cpu_pipeline: Any = StableDiffusionPipeline.from_pretrained(  # type: ignore[no-untyped-call]
                    model_name,
                    torch_dtype=torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
                cpu_pipeline = cpu_pipeline.to(self.device)
                self.pipeline = cpu_pipeline
                logger.info("Using CPU mode with fp32")

        except Exception as e:
            logger.error(f"Failed to load Stable Diffusion model: {e}")
            raise RuntimeError(f"Could not load model {model_name}: {e}") from e

        return self.pipeline

    def generate_avatar(
        self,
        prompt_override: str | None = None,
        seed: int = 42,
        steps: int = 30,
        height: int = 768,
        width: int = 512,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        negative: str | None = DEFAULT_NEGATIVE_PROMPT,
        out_dir: Path | str = Path("media/avatars"),
    ) -> Path:
        """
        Generate an avatar image using Stable Diffusion.

        Args:
            prompt_override: Custom prompt to use instead of default
            seed: Random seed for reproducible generation
            steps: Number of inference steps
            height: Image height in pixels
            width: Image width in pixels
            model_name: Hugging Face model identifier
            negative: Negative prompt to avoid unwanted features
            out_dir: Output directory for generated images

        Returns:
            Path to the generated PNG file

        Raises:
            RuntimeError: If generation fails
            OSError: If output directory cannot be created
        """
        prompt = prompt_override or DEFAULT_PROMPT
        negative_prompt = negative or DEFAULT_NEGATIVE_PROMPT
        output_dir = Path(out_dir)

        logger.info(f"Generating avatar with prompt: {prompt[:50]}...")
        logger.info(f"Parameters: seed={seed}, steps={steps}, size={width}x{height}")

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}")
            raise OSError(f"Cannot create output directory: {e}") from e

        try:
            pipeline = self._load_pipeline(model_name)

            generator = torch.Generator(device=self.device).manual_seed(seed)

            with torch.inference_mode():
                result = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    generator=generator,
                    guidance_scale=7.5,
                )

            if not result.images:
                raise RuntimeError("No images generated")

            image = result.images[0]

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}.png"
            output_path = output_dir / filename

            image.save(output_path, "PNG", optimize=True)

            file_size = output_path.stat().st_size
            logger.info(f"Avatar generated successfully: {output_path} ({file_size} bytes)")

            return output_path

        except Exception as e:
            logger.error(f"Avatar generation failed: {e}")
            raise RuntimeError(f"Failed to generate avatar: {e}") from e

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Avatar generation model unloaded")


def generate_avatar(
    prompt_override: str | None = None,
    seed: int = 42,
    steps: int = 30,
    height: int = 768,
    width: int = 512,
    model_name: str = "runwayml/stable-diffusion-v1-5",
    negative: str | None = DEFAULT_NEGATIVE_PROMPT,
    out_dir: Path | str = Path("media/avatars"),
) -> Path:
    """
    Standalone function to generate an avatar image.

    This is a convenience function that creates a service instance
    and generates a single avatar.

    Returns the file path of the generated PNG.
    """
    service = AvatarGeneratorService()
    try:
        return service.generate_avatar(
            prompt_override=prompt_override,
            seed=seed,
            steps=steps,
            height=height,
            width=width,
            model_name=model_name,
            negative=negative,
            out_dir=out_dir,
        )
    finally:
        service.unload_model()
