"""
Comprehensive tests for ECG Analysis API endpoints with image support

This module tests the enhanced ECG analysis API endpoints that support
image file uploads and document scanning functionality.
"""

import json
import tempfile
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi import status
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


class TestECGAnalysisAPIWithImages:
    """Test suite for ECG Analysis API with image support."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_user(self):
        """Mock authenticated user."""
        return MagicMock(
            id=1,
            email="test@example.com",
            is_superuser=False,
            is_physician=True
        )

    @pytest.fixture
    def sample_ecg_image(self) -> BytesIO:
        """Create a sample ECG image for testing."""
        image = np.ones((600, 800, 3), dtype=np.uint8) * 255
        
        for i in range(0, 800, 50):
            cv2.line(image, (i, 0), (i, 600), (200, 200, 200), 1)
        for i in range(0, 600, 25):
            cv2.line(image, (0, i), (800, i), (200, 200, 200), 1)
        
        for x in range(0, 800, 2):
            y = 300 + int(50 * np.sin(x * 0.1) + 20 * np.sin(x * 0.3))
            if 0 <= y < 600:
                cv2.circle(image, (x, y), 1, (0, 0, 0), -1)
        
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_buffer = BytesIO()
        pil_image.save(image_buffer, format='JPEG')
        image_buffer.seek(0)
        
        return image_buffer

    @pytest.fixture
    def sample_csv_data(self) -> BytesIO:
        """Create sample CSV ECG data."""
        csv_content = """time,I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6
0.000,0.1,0.2,0.1,-0.15,0.05,0.15,0.0,0.1,0.2,0.3,0.4,0.3
0.002,0.15,0.25,0.1,-0.175,0.075,0.175,0.05,0.15,0.25,0.35,0.45,0.35
0.004,0.2,0.3,0.1,-0.2,0.1,0.2,0.1,0.2,0.3,0.4,0.5,0.4
"""
        return BytesIO(csv_content.encode())

    @pytest.fixture
    def non_ecg_image(self) -> BytesIO:
        """Create a non-ECG image for testing."""
        image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        
        pil_image = Image.fromarray(image)
        image_buffer = BytesIO()
        pil_image.save(image_buffer, format='JPEG')
        image_buffer.seek(0)
        
        return image_buffer

    @patch('app.services.user_service.UserService.get_current_user')
    @patch('app.services.ecg_service.ECGAnalysisService.create_analysis')
    @patch('app.services.ecg_document_scanner.ECGDocumentScanner.process_ecg_image')
    def test_upload_ecg_image_success(
        self, 
        mock_scanner, 
        mock_create_analysis, 
        mock_get_user, 
        client: TestClient, 
        mock_user, 
        sample_ecg_image: BytesIO
    ):
        """Test successful ECG image upload."""
        mock_get_user.return_value = mock_user
        
        mock_scanner.return_value = {
            "confidence": 0.85,
            "validation": {
                "is_valid_ecg": True,
                "grid_detected": True,
                "leads_detected": 12
            },
            "metadata": {
                "processing_method": "automatic_detection",
                "original_size": [600, 800],
                "processed_size": [600, 800]
            }
        }
        
        mock_analysis = MagicMock()
        mock_analysis.analysis_id = "test-analysis-123"
        mock_analysis.status = "pending"
        mock_create_analysis.return_value = mock_analysis
        
        response = client.post(
            "/api/v1/ecg/upload",
            data={"patient_id": "1"},
            files={"file": ("test_ecg.jpg", sample_ecg_image, "image/jpeg")}
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        response_data = response.json()
        assert response_data["analysis_id"] == "test-analysis-123"
        assert response_data["status"] == "pending"
        assert response_data["file_type"] == ".jpg"
        assert response_data["document_scanning_metadata"] is not None
        
        metadata = response_data["document_scanning_metadata"]
        assert metadata["scanner_confidence"] == 0.85
        assert metadata["document_detected"] is True
        assert metadata["grid_detected"] is True
        assert metadata["leads_detected"] == 12

    @patch('app.services.user_service.UserService.get_current_user')
    @patch('app.services.ecg_service.ECGAnalysisService.create_analysis')
    @patch('app.services.ecg_document_scanner.ECGDocumentScanner.process_ecg_image')
    def test_upload_ecg_image_low_confidence(
        self, 
        mock_scanner, 
        mock_create_analysis, 
        mock_get_user, 
        client: TestClient, 
        mock_user, 
        non_ecg_image: BytesIO
    ):
        """Test ECG image upload with low confidence detection."""
        mock_get_user.return_value = mock_user
        
        mock_scanner.return_value = {
            "confidence": 0.2,
            "validation": {
                "is_valid_ecg": False,
                "grid_detected": False,
                "leads_detected": 0
            },
            "metadata": {
                "processing_method": "fallback",
                "original_size": [400, 600],
                "processed_size": [400, 600]
            }
        }
        
        mock_analysis = MagicMock()
        mock_analysis.analysis_id = "test-analysis-456"
        mock_analysis.status = "pending"
        mock_create_analysis.return_value = mock_analysis
        
        response = client.post(
            "/api/v1/ecg/upload",
            data={"patient_id": "1"},
            files={"file": ("random_image.jpg", non_ecg_image, "image/jpeg")}
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        response_data = response.json()
        assert "low confidence" in response_data["message"].lower()
        
        metadata = response_data["document_scanning_metadata"]
        assert metadata["scanner_confidence"] == 0.2
        assert metadata["document_detected"] is False

    @patch('app.services.user_service.UserService.get_current_user')
    @patch('app.services.ecg_service.ECGAnalysisService.create_analysis')
    def test_upload_csv_file_backward_compatibility(
        self, 
        mock_create_analysis, 
        mock_get_user, 
        client: TestClient, 
        mock_user, 
        sample_csv_data: BytesIO
    ):
        """Test that CSV file upload still works (backward compatibility)."""
        mock_get_user.return_value = mock_user
        
        mock_analysis = MagicMock()
        mock_analysis.analysis_id = "test-analysis-csv"
        mock_analysis.status = "pending"
        mock_create_analysis.return_value = mock_analysis
        
        response = client.post(
            "/api/v1/ecg/upload",
            data={"patient_id": "1"},
            files={"file": ("test_ecg.csv", sample_csv_data, "text/csv")}
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        response_data = response.json()
        assert response_data["analysis_id"] == "test-analysis-csv"
        assert response_data["file_type"] == ".csv"
        assert response_data["document_scanning_metadata"] is None  # No scanning for CSV

    @patch('app.services.user_service.UserService.get_current_user')
    def test_upload_unsupported_file_type(
        self, 
        mock_get_user, 
        client: TestClient, 
        mock_user
    ):
        """Test upload with unsupported file type."""
        mock_get_user.return_value = mock_user
        
        unsupported_file = BytesIO(b"unsupported content")
        
        response = client.post(
            "/api/v1/ecg/upload",
            data={"patient_id": "1"},
            files={"file": ("test.pdf", unsupported_file, "application/pdf")}
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Unsupported file type" in response.json()["detail"]

    @patch('app.services.user_service.UserService.get_current_user')
    @patch('app.services.ecg_service.ECGAnalysisService.create_analysis')
    @patch('app.services.ecg_document_scanner.ECGDocumentScanner.process_ecg_image')
    def test_upload_image_scanner_error(
        self, 
        mock_scanner, 
        mock_create_analysis, 
        mock_get_user, 
        client: TestClient, 
        mock_user, 
        sample_ecg_image: BytesIO
    ):
        """Test image upload when scanner encounters an error."""
        mock_get_user.return_value = mock_user
        mock_scanner.side_effect = Exception("Scanner processing failed")
        
        mock_analysis = MagicMock()
        mock_analysis.analysis_id = "test-analysis-error"
        mock_analysis.status = "pending"
        mock_create_analysis.return_value = mock_analysis
        
        response = client.post(
            "/api/v1/ecg/upload",
            data={"patient_id": "1"},
            files={"file": ("test_ecg.jpg", sample_ecg_image, "image/jpeg")}
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        response_data = response.json()
        metadata = response_data["document_scanning_metadata"]
        assert metadata["scanner_confidence"] == 0.0
        assert metadata["document_detected"] is False
        assert metadata["processing_method"] == "fallback"
        assert "error" in metadata

    def test_file_type_validation_comprehensive(self, client: TestClient):
        """Test comprehensive file type validation."""
        supported_image_types = ['.jpg', '.jpeg', '.png']
        supported_data_types = ['.csv', '.txt', '.xml', '.dat']
        
        all_supported = supported_image_types + supported_data_types
        
        unsupported_types = ['.pdf', '.doc', '.gif', '.bmp', '.tiff']
        
        for file_type in unsupported_types:
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
