"""
Tests for the ECG Image Extractor functionality.
"""

import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from app.services.ecg_image_extractor import ECGImageExtractor
from app.core.exceptions import ECGProcessingException


class TestECGImageExtractor:
    """Test suite for ECG Image Extractor."""

    def test_init(self):
        """Test initialization of ECGImageExtractor."""
        extractor = ECGImageExtractor()
        assert extractor is not None
        assert hasattr(extractor, "supported_formats")
        assert ".png" in extractor.supported_formats
        assert ".jpg" in extractor.supported_formats
        assert ".jpeg" in extractor.supported_formats
        assert ".pdf" in extractor.supported_formats

    @patch("cv2.imread")
    def test_process_image_file_not_found(self, mock_imread):
        """Test handling of file not found."""
        mock_imread.return_value = None
        extractor = ECGImageExtractor()
        
        with pytest.raises(ECGProcessingException) as excinfo:
            extractor._process_image("nonexistent_file.jpg")
        
        assert "Could not load image" in str(excinfo.value)

    @patch("cv2.imread")
    @patch("cv2.cvtColor")
    @patch("cv2.GaussianBlur")
    @patch("cv2.adaptiveThreshold")
    @patch("cv2.getStructuringElement")
    @patch("cv2.morphologyEx")
    @patch("cv2.add")
    @patch("cv2.HoughLinesP")
    @patch("cv2.findContours")
    def test_process_image_basic_workflow(
        self, mock_findcontours, mock_hough, mock_add, mock_morphology, 
        mock_structuring, mock_threshold, mock_blur, mock_cvtcolor, mock_imread
    ):
        """Test the basic workflow of image processing."""
        # Mock image data
        mock_image = np.zeros((300, 400, 3), dtype=np.uint8)
        mock_gray = np.zeros((300, 400), dtype=np.uint8)
        mock_processed = np.zeros((300, 400), dtype=np.uint8)
        
        # Set up mocks
        mock_imread.return_value = mock_image
        mock_cvtcolor.return_value = mock_gray
        mock_blur.return_value = mock_gray
        mock_threshold.return_value = mock_processed
        mock_structuring.return_value = np.ones((3, 3), dtype=np.uint8)
        mock_morphology.return_value = mock_processed
        mock_add.return_value = mock_processed
        mock_hough.return_value = None
        mock_findcontours.return_value = ([], None)
        
        # Create extractor
        extractor = ECGImageExtractor()
        
        # Test with mocked functions
        with patch.object(extractor, '_detect_lead_regions', return_value=[mock_processed]):
            with patch.object(extractor, '_extract_signal_from_region', 
                             return_value=(np.zeros(5000), 0.5)):
                result = extractor._process_image("test_image.jpg")
        
        # Verify result structure
        assert "signal" in result
        assert "sampling_rate" in result
        assert "labels" in result
        assert "metadata" in result
        assert result["metadata"]["source"] == "digitized_image"
        assert result["metadata"]["source_type"] == "image"

    def test_extract_from_file_unsupported_format(self):
        """Test handling of unsupported file formats."""
        extractor = ECGImageExtractor()
        
        with pytest.raises(ECGProcessingException) as excinfo:
            extractor.extract_from_file("test.unsupported")
        
        assert "Unsupported file format" in str(excinfo.value)

    @patch("pdf2image.convert_from_path")
    def test_process_pdf_no_images(self, mock_convert):
        """Test handling of PDF with no images."""
        mock_convert.return_value = []
        extractor = ECGImageExtractor()
        
        with pytest.raises(ECGProcessingException) as excinfo:
            extractor._process_pdf("test.pdf")
        
        assert "No images extracted from PDF" in str(excinfo.value)

    @patch("os.path.join")
    @patch("pdf2image.convert_from_path")
    def test_process_pdf_with_image(self, mock_convert, mock_join):
        """Test processing PDF with an image."""
        # Mock PDF conversion
        mock_image = MagicMock()
        mock_convert.return_value = [mock_image]
        mock_join.return_value = "temp_dir/temp_image.png"
        
        extractor = ECGImageExtractor()
        
        # Mock the image processing
        with patch.object(extractor, '_process_image', 
                         return_value={"signal": np.zeros((5000, 12)), 
                                      "sampling_rate": 500, 
                                      "labels": extractor.default_leads,
                                      "metadata": {"source": "digitized_image"}}):
            result = extractor._process_pdf("test.pdf")
        
        # Verify result
        assert "signal" in result
        assert "metadata" in result
        assert result["metadata"]["source_type"] == "pdf"
        assert result["metadata"]["page_count"] == 1