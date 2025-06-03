"""
Comprehensive tests for ECG Document Scanner Service

This module provides extensive testing for the ECG document scanning functionality
including edge detection, perspective correction, image enhancement, and validation.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest
from PIL import Image

from app.core.exceptions import ECGProcessingException
from app.services.ecg_document_scanner import BatchECGScanner, ECGDocumentScanner


class TestECGDocumentScanner:
    """Test suite for ECG Document Scanner functionality."""

    @pytest.fixture
    def scanner(self) -> ECGDocumentScanner:
        """Create ECG document scanner instance."""
        return ECGDocumentScanner()

    @pytest.fixture
    def sample_image_path(self) -> str:
        """Create a sample test image."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            image = np.ones((600, 800, 3), dtype=np.uint8) * 255
            
            for i in range(0, 800, 50):
                cv2.line(image, (i, 0), (i, 600), (200, 200, 200), 1)
            for i in range(0, 600, 25):
                cv2.line(image, (0, i), (800, i), (200, 200, 200), 1)
            
            for lead in range(3):
                y_offset = 150 + lead * 150
                for x in range(0, 800, 2):
                    y = y_offset + int(20 * np.sin(x * 0.1) + 10 * np.sin(x * 0.3))
                    if 0 <= y < 600:
                        cv2.circle(image, (x, y), 1, (0, 0, 0), -1)
            
            cv2.imwrite(tmp_file.name, image)
            return tmp_file.name

    @pytest.fixture
    def invalid_image_path(self) -> str:
        """Create an invalid image path."""
        return "/nonexistent/path/image.jpg"

    @pytest.fixture
    def non_ecg_image_path(self) -> str:
        """Create a non-ECG image."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
            cv2.imwrite(tmp_file.name, image)
            return tmp_file.name

    def teardown_method(self):
        """Clean up temporary files."""
        for file_path in ['/tmp/test_ecg.jpg', '/tmp/processed_ecg.jpg']:
            if os.path.exists(file_path):
                os.unlink(file_path)

    @pytest.mark.asyncio
    async def test_process_ecg_image_success(self, scanner: ECGDocumentScanner, sample_image_path: str):
        """Test successful ECG image processing."""
        result = await scanner.process_ecg_image(sample_image_path)
        
        assert result is not None
        assert "processed_image" in result
        assert "original_image" in result
        assert "confidence" in result
        assert "validation" in result
        assert "metadata" in result
        
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0
        
        validation = result["validation"]
        assert "is_valid_ecg" in validation
        assert "confidence" in validation
        assert "detected_features" in validation
        
        metadata = result["metadata"]
        assert "original_size" in metadata
        assert "processed_size" in metadata
        assert "processing_method" in metadata

    @pytest.mark.asyncio
    async def test_process_ecg_image_with_output_path(self, scanner: ECGDocumentScanner, sample_image_path: str):
        """Test ECG image processing with output path."""
        output_path = "/tmp/processed_ecg.jpg"
        
        result = await scanner.process_ecg_image(sample_image_path, output_path)
        
        assert result is not None
        assert os.path.exists(output_path)
        
        processed_image = cv2.imread(output_path)
        assert processed_image is not None

    @pytest.mark.asyncio
    async def test_process_ecg_image_invalid_path(self, scanner: ECGDocumentScanner, invalid_image_path: str):
        """Test ECG image processing with invalid path."""
        with pytest.raises(ECGProcessingException):
            await scanner.process_ecg_image(invalid_image_path)

    @pytest.mark.asyncio
    async def test_detect_document_edges_success(self, scanner: ECGDocumentScanner, sample_image_path: str):
        """Test document edge detection."""
        image = await scanner._load_image(sample_image_path)
        assert image is not None
        
        corners = await scanner.detect_document_edges(image)
        
        if corners is not None:
            assert corners.shape == (4, 2)
            assert corners.dtype == np.float32
            
            height, width = image.shape[:2]
            assert np.all(corners[:, 0] >= 0)
            assert np.all(corners[:, 0] <= width)
            assert np.all(corners[:, 1] >= 0)
            assert np.all(corners[:, 1] <= height)

    @pytest.mark.asyncio
    async def test_validate_ecg_document_valid(self, scanner: ECGDocumentScanner, sample_image_path: str):
        """Test ECG document validation with valid ECG."""
        image = await scanner._load_image(sample_image_path)
        assert image is not None
        
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        
        validation_result = await scanner.validate_ecg_document(gray_image)
        
        assert "is_valid_ecg" in validation_result
        assert "confidence" in validation_result
        assert "detected_features" in validation_result
        assert "grid_detected" in validation_result
        assert "leads_detected" in validation_result
        assert "waveform_quality" in validation_result
        
        assert isinstance(validation_result["confidence"], float)
        assert 0.0 <= validation_result["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_validate_ecg_document_invalid(self, scanner: ECGDocumentScanner, non_ecg_image_path: str):
        """Test ECG document validation with non-ECG image."""
        image = await scanner._load_image(non_ecg_image_path)
        assert image is not None
        
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        
        validation_result = await scanner.validate_ecg_document(gray_image)
        
        assert validation_result["confidence"] < 0.5
        assert not validation_result["is_valid_ecg"]

    def test_order_corner_points(self, scanner: ECGDocumentScanner):
        """Test corner point ordering."""
        corners = np.array([
            [100, 100],  # top-left
            [300, 100],  # top-right
            [300, 200],  # bottom-right
            [100, 200]   # bottom-left
        ], dtype=np.float32)
        
        np.random.shuffle(corners)
        
        ordered = scanner._order_corner_points(corners)
        
        assert ordered.shape == (4, 2)
        assert np.sum(ordered[0]) <= np.sum(ordered[1])


class TestBatchECGScanner:
    """Test suite for Batch ECG Scanner functionality."""

    @pytest.fixture
    def batch_scanner(self) -> BatchECGScanner:
        """Create batch ECG scanner instance."""
        return BatchECGScanner(max_workers=2)

    @pytest.fixture
    def sample_image_paths(self) -> list[str]:
        """Create multiple sample test images."""
        paths = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix=f'_test_{i}.jpg', delete=False) as tmp_file:
                image = np.ones((400, 600, 3), dtype=np.uint8) * 255
                
                for j in range(0, 600, 50):
                    cv2.line(image, (0, j), (600, j), (200, 200, 200), 1)
                for j in range(0, 400, 25):
                    cv2.line(image, (j, 0), (j, 400), (200, 200, 200), 1)
                
                cv2.imwrite(tmp_file.name, image)
                paths.append(tmp_file.name)
        
        return paths

    def teardown_method(self):
        """Clean up temporary files."""
        import glob
        for file_path in glob.glob('/tmp/*_test_*.jpg'):
            if os.path.exists(file_path):
                os.unlink(file_path)

    @pytest.mark.asyncio
    async def test_process_batch_success(self, batch_scanner: BatchECGScanner, sample_image_paths: list[str]):
        """Test successful batch processing."""
        results = await batch_scanner.process_batch(sample_image_paths)
        
        assert len(results) == len(sample_image_paths)
        
        for result in results:
            assert "input_path" in result
            assert "success" in result
            assert result["input_path"] in sample_image_paths

    @pytest.mark.asyncio
    async def test_get_processing_stats_success(self, batch_scanner: BatchECGScanner, sample_image_paths: list[str]):
        """Test processing statistics generation."""
        results = await batch_scanner.process_batch(sample_image_paths)
        stats = await batch_scanner.get_processing_stats(results)
        
        assert "total_processed" in stats
        assert "successful" in stats
        assert "failed" in stats
        assert "success_rate" in stats
        assert "average_confidence" in stats
        assert "valid_ecgs_detected" in stats
        assert "ecg_detection_rate" in stats
        
        assert stats["total_processed"] == len(sample_image_paths)
        assert 0.0 <= stats["success_rate"] <= 1.0
        assert 0.0 <= stats["average_confidence"] <= 1.0
        assert 0.0 <= stats["ecg_detection_rate"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
