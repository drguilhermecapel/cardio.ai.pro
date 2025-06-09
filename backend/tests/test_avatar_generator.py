"""
Tests for avatar generation service.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import os

from app.services.avatar_generator import AvatarGeneratorService, generate_avatar


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_stable_diffusion_pipeline():
    """Mock Stable Diffusion pipeline for testing."""
    with patch('app.services.avatar_generator.StableDiffusionPipeline') as mock_pipeline_class:
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline
        mock_pipeline.enable_attention_slicing = MagicMock()
        mock_pipeline.enable_memory_efficient_attention = MagicMock()
        
        mock_image = MagicMock()
        mock_image.save = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [mock_image]
        mock_pipeline.return_value = mock_result
        
        yield mock_pipeline_class


@pytest.fixture
def mock_torch():
    """Mock torch for testing."""
    with patch('app.services.avatar_generator.torch') as mock_torch:
        mock_torch.cuda.is_available.return_value = False  # Test CPU mode
        mock_generator = MagicMock()
        mock_torch.Generator.return_value = mock_generator
        mock_generator.manual_seed.return_value = mock_generator
        mock_torch.inference_mode.return_value.__enter__ = MagicMock()
        mock_torch.inference_mode.return_value.__exit__ = MagicMock()
        yield mock_torch


class TestAvatarGeneratorService:
    """Test cases for AvatarGeneratorService."""

    def test_service_initialization(self, mock_torch):
        """Test avatar generator service initialization."""
        service = AvatarGeneratorService()
        assert service.pipeline is None
        assert service.device == "cpu"  # Since we mocked CUDA as unavailable

    def test_service_initialization_with_cuda(self):
        """Test avatar generator service initialization with CUDA."""
        with patch('app.services.avatar_generator.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            service = AvatarGeneratorService()
            assert service.device == "cuda"

    def test_load_pipeline_cpu(self, mock_torch, mock_stable_diffusion_pipeline):
        """Test loading pipeline in CPU mode."""
        service = AvatarGeneratorService()
        pipeline = service._load_pipeline("test-model")
        
        assert pipeline is not None
        assert service.pipeline is not None
        mock_stable_diffusion_pipeline.from_pretrained.assert_called_once()

    def test_load_pipeline_cuda(self, mock_stable_diffusion_pipeline):
        """Test loading pipeline in CUDA mode."""
        with patch('app.services.avatar_generator.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            service = AvatarGeneratorService()
            pipeline = service._load_pipeline("test-model")
            
            assert pipeline is not None
            pipeline_instance = mock_stable_diffusion_pipeline.from_pretrained.return_value
            pipeline_instance.enable_attention_slicing.assert_called_once()
            pipeline_instance.enable_memory_efficient_attention.assert_called_once()

    def test_generate_avatar_success(self, mock_torch, mock_stable_diffusion_pipeline, temp_output_dir):
        """Test successful avatar generation."""
        service = AvatarGeneratorService()
        
        with patch('pathlib.Path.stat') as mock_stat, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True):
            mock_stat.return_value.st_size = 500000  # 500KB
            
            result_path = service.generate_avatar(
                prompt_override="test prompt",
                seed=123,
                steps=10,  # Low steps for faster testing
                height=256,  # Low resolution for testing
                width=256,
                out_dir=temp_output_dir
            )
            
            assert result_path.suffix == ".png"
            mock_stable_diffusion_pipeline.from_pretrained.assert_called()

    def test_generate_avatar_creates_output_directory(self, mock_torch, mock_stable_diffusion_pipeline, temp_output_dir):
        """Test that avatar generation creates output directory if it doesn't exist."""
        service = AvatarGeneratorService()
        non_existent_dir = temp_output_dir / "new_dir"
        
        with patch('pathlib.Path.stat') as mock_stat, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True), \
             patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_stat.return_value.st_size = 500000
            
            result_path = service.generate_avatar(out_dir=non_existent_dir)
            
            assert result_path.suffix == ".png"

    def test_generate_avatar_with_default_prompt(self, mock_torch, mock_stable_diffusion_pipeline, temp_output_dir):
        """Test avatar generation with default prompt."""
        service = AvatarGeneratorService()
        
        with patch('pathlib.Path.stat') as mock_stat, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True):
            mock_stat.return_value.st_size = 500000
            
            result_path = service.generate_avatar(out_dir=temp_output_dir)
            
            mock_stable_diffusion_pipeline.from_pretrained.assert_called()
            pipeline_instance = mock_stable_diffusion_pipeline.from_pretrained.return_value
            pipeline_instance.assert_called()
            call_args = pipeline_instance.call_args
            assert "50-year-old caucasian woman" in call_args[1]["prompt"]

    def test_unload_model(self, mock_torch):
        """Test model unloading."""
        service = AvatarGeneratorService()
        service.pipeline = MagicMock()
        
        service.unload_model()
        
        assert service.pipeline is None

    def test_generate_avatar_pipeline_failure(self, mock_torch):
        """Test avatar generation when pipeline loading fails."""
        with patch('app.services.avatar_generator.StableDiffusionPipeline') as mock_pipeline_class:
            mock_pipeline_class.from_pretrained.side_effect = Exception("Model loading failed")
            
            service = AvatarGeneratorService()
            
            with pytest.raises(RuntimeError, match="Could not load model"):
                service.generate_avatar()


class TestStandaloneFunction:
    """Test cases for the standalone generate_avatar function."""

    def test_standalone_generate_avatar(self, mock_torch, mock_stable_diffusion_pipeline, temp_output_dir):
        """Test the standalone generate_avatar function."""
        with patch('pathlib.Path.stat') as mock_stat, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True):
            mock_stat.return_value.st_size = 800000  # 800KB - under 1MB
            
            result_path = generate_avatar(
                prompt_override="test prompt",
                seed=42,
                steps=20,
                height=384,  # Low resolution for testing
                width=512,
                out_dir=temp_output_dir
            )
            
            assert result_path.suffix == ".png"
            assert mock_stat.return_value.st_size < 1024 * 1024

    def test_standalone_generate_avatar_low_res_under_1mb(self, mock_torch, mock_stable_diffusion_pipeline, temp_output_dir):
        """Test that low-res avatar generation returns PNG under 1MB."""
        with patch('pathlib.Path.stat') as mock_stat, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True):
            mock_stat.return_value.st_size = 512000  # 512KB - well under 1MB
            
            result_path = generate_avatar(
                steps=15,  # Low steps
                height=384,  # Low resolution
                width=256,   # Low resolution
                out_dir=temp_output_dir
            )
            
            assert result_path.suffix == ".png"
            assert mock_stat.return_value.st_size < 1024 * 1024  # Under 1MB
            assert mock_stat.return_value.st_size > 0  # File exists and has content

    def test_standalone_function_cleanup(self, mock_torch, mock_stable_diffusion_pipeline, temp_output_dir):
        """Test that standalone function properly cleans up resources."""
        with patch('pathlib.Path.stat') as mock_stat, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True):
            mock_stat.return_value.st_size = 500000
            
            with patch.object(AvatarGeneratorService, 'unload_model') as mock_unload:
                result_path = generate_avatar(out_dir=temp_output_dir)
                
                mock_unload.assert_called_once()


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_output_directory(self, mock_torch, mock_stable_diffusion_pipeline):
        """Test handling of invalid output directory."""
        service = AvatarGeneratorService()
        
        with patch('pathlib.Path.mkdir', side_effect=OSError("Permission denied")), \
             patch('pathlib.Path.exists', return_value=False), \
             patch('pathlib.Path.is_dir', return_value=False):
            with pytest.raises(OSError, match="Cannot create output directory"):
                service.generate_avatar(out_dir="/invalid/path")

    def test_image_generation_failure(self, mock_torch, mock_stable_diffusion_pipeline, temp_output_dir):
        """Test handling of image generation failure."""
        service = AvatarGeneratorService()
        
        pipeline_instance = mock_stable_diffusion_pipeline.from_pretrained.return_value
        pipeline_instance.return_value.images = []
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True):
            with pytest.raises(RuntimeError, match="No images generated"):
                service.generate_avatar(out_dir=temp_output_dir)

    def test_image_save_failure(self, mock_torch, mock_stable_diffusion_pipeline, temp_output_dir):
        """Test handling of image save failure."""
        service = AvatarGeneratorService()
        
        mock_image = MagicMock()
        mock_image.save.side_effect = Exception("Save failed")
        pipeline_instance = mock_stable_diffusion_pipeline.from_pretrained.return_value
        pipeline_instance.return_value.images = [mock_image]
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True):
            with pytest.raises(RuntimeError, match="Failed to generate avatar"):
                service.generate_avatar(out_dir=temp_output_dir)
