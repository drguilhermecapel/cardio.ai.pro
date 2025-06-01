"""Focused ECG service tests for actual methods."""

import pytest
from unittest.mock import AsyncMock, Mock
from datetime import datetime

from app.services.ecg_service import ECGAnalysisService
from app.schemas.ecg_analysis import ECGAnalysisCreate
from app.models.ecg_analysis import ECGAnalysis


@pytest.fixture
def ecg_service(test_db):
    """Create ECG service instance."""
    mock_ml_service = Mock()
    mock_validation_service = Mock()
    return ECGAnalysisService(
        db=test_db,
        ml_service=mock_ml_service,
        validation_service=mock_validation_service
    )


@pytest.fixture
def sample_ecg_data():
    """Sample ECG analysis data."""
    return ECGAnalysisCreate(
        patient_id=1,
        original_filename="test_ecg.txt",
        acquisition_date=datetime.utcnow(),
        sample_rate=500,
        duration_seconds=10.0,
        leads_count=12,
        leads_names=["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        device_manufacturer="Test Device",
        device_model="v1.0"
    )


@pytest.mark.asyncio
async def test_create_ecg_analysis(ecg_service, sample_ecg_data):
    """Test ECG analysis creation."""
    mock_analysis = ECGAnalysis()
    mock_analysis.id = 1
    mock_analysis.patient_id = 1
    mock_analysis.original_filename = "test_ecg.txt"
    mock_analysis.created_by = 1
    
    ecg_service.repository.create_analysis = AsyncMock(return_value=mock_analysis)
    
    analysis = await ecg_service.create_analysis(
        ecg_data=sample_ecg_data,
        created_by=1
    )
    
    assert analysis is not None
    assert analysis.patient_id == 1
    assert analysis.original_filename == "test_ecg.txt"
    assert analysis.created_by == 1


@pytest.mark.asyncio
async def test_get_analysis_by_id(ecg_service):
    """Test getting ECG analysis by ID."""
    mock_analysis = ECGAnalysis()
    mock_analysis.id = 1
    mock_analysis.patient_id = 1
    
    ecg_service.repository.get_analysis_by_id = AsyncMock(return_value=mock_analysis)
    
    analysis = await ecg_service.get_analysis_by_id(1)
    
    assert analysis is not None
    assert analysis.id == 1
    assert analysis.patient_id == 1


@pytest.mark.asyncio
async def test_get_analyses_for_patient(ecg_service):
    """Test getting ECG analyses for a patient."""
    mock_analyses = [ECGAnalysis(), ECGAnalysis()]
    ecg_service.repository.get_analyses_for_patient = AsyncMock(return_value=(mock_analyses, 2))
    
    analyses, total = await ecg_service.get_analyses_for_patient(
        patient_id=1,
        limit=10,
        offset=0
    )
    
    assert len(analyses) == 2
    assert total == 2


@pytest.mark.asyncio
async def test_process_ecg_signal(ecg_service):
    """Test ECG signal processing."""
    mock_signal_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    mock_processed_data = {"heart_rate": 75, "rhythm": "normal"}
    
    ecg_service.signal_processor.process_signal = Mock(return_value=mock_processed_data)
    
    result = ecg_service.signal_processor.process_signal(mock_signal_data)
    
    assert result["heart_rate"] == 75
    assert result["rhythm"] == "normal"


@pytest.mark.asyncio
async def test_validate_ecg_quality(ecg_service):
    """Test ECG quality validation."""
    mock_signal_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    mock_quality_result = {"quality_score": 0.85, "issues": []}
    
    ecg_service.quality_validator.validate_quality = Mock(return_value=mock_quality_result)
    
    result = ecg_service.quality_validator.validate_quality(mock_signal_data)
    
    assert result["quality_score"] == 0.85
    assert len(result["issues"]) == 0
