"""Test ECG Analysis Service."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.ecg_service import ECGAnalysisService
from app.models.ecg_analysis import ECGAnalysis
from app.models.patient import Patient
from app.schemas.ecg_analysis import ECGAnalysisCreate


@pytest.fixture
def mock_ml_service():
    """Mock ML model service."""
    service = Mock()
    service.analyze_ecg = AsyncMock(return_value={
        "classification": "normal",
        "confidence": 0.95,
        "rhythm": "sinus",
        "quality_score": 0.88
    })
    service.get_interpretability_map = AsyncMock(return_value={
        "attention_weights": [0.1, 0.2, 0.3],
        "feature_importance": {"hr": 0.8, "qrs": 0.6}
    })
    return service


@pytest.fixture
def mock_validation_service():
    """Mock validation service."""
    service = Mock()
    service.validate_ecg_analysis = AsyncMock(return_value=True)
    return service


@pytest.fixture
def ecg_service(test_db, mock_ml_service, mock_validation_service):
    """Create ECG service instance."""
    return ECGAnalysisService(
        db=test_db,
        ml_service=mock_ml_service,
        validation_service=mock_validation_service
    )


@pytest.fixture
def sample_patient_data():
    """Sample patient data."""
    return {
        "name": "Test Patient",
        "birth_date": "1990-01-01",
        "gender": "M",
        "medical_record_number": "MRN123456"
    }


@pytest.fixture
def sample_ecg_data():
    """Sample ECG analysis data."""
    return ECGAnalysisCreate(
        patient_id=1,
        original_filename="test_ecg.txt",
        acquisition_date="2025-06-01T14:00:00Z",
        sample_rate=500,
        duration_seconds=10.0,
        leads_count=12,
        leads_names=["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        device_manufacturer="Test Device",
        device_model="v1.0"
    )


@pytest.mark.asyncio
async def test_create_ecg_analysis_success(ecg_service, sample_ecg_data, mock_ml_service):
    """Test successful ECG analysis creation."""
    with patch('app.utils.ecg_processor.ECGProcessor.process_file') as mock_processor:
        mock_processor.return_value = {
            "signal_data": [[1, 2, 3], [4, 5, 6]],
            "quality_metrics": {"snr": 15.5, "baseline_drift": 0.02}
        }
        
        result = await ecg_service.create_ecg_analysis(sample_ecg_data)
        
        assert result is not None
        assert result.classification == "normal"
        assert result.confidence == 0.95
        assert result.rhythm == "sinus"
        mock_ml_service.analyze_ecg.assert_called_once()


@pytest.mark.asyncio
async def test_create_ecg_analysis_with_patient_creation(ecg_service, sample_patient_data):
    """Test ECG analysis creation with new patient."""
    ecg_data = ECGAnalysisCreate(
        patient_id=1,
        original_filename="test_ecg.txt",
        acquisition_date="2025-06-01T14:00:00Z",
        sample_rate=500,
        duration_seconds=10.0,
        leads_count=12,
        leads_names=["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        device_manufacturer="Test Device",
        device_model="v1.0"
    )
    
    with patch('app.utils.ecg_processor.ECGProcessor.process_file') as mock_processor:
        mock_processor.return_value = {
            "signal_data": [[1, 2, 3], [4, 5, 6]],
            "quality_metrics": {"snr": 15.5, "baseline_drift": 0.02}
        }
        
        result = await ecg_service.create_ecg_analysis(ecg_data)
        
        assert result is not None
        assert result.patient_id is not None


@pytest.mark.asyncio
async def test_process_ecg_file_invalid_format(ecg_service):
    """Test processing invalid ECG file format."""
    with pytest.raises(ValueError, match="Unsupported file format"):
        await ecg_service.process_ecg_file("/tmp/invalid.xyz")


@pytest.mark.asyncio
async def test_process_ecg_file_missing_file(ecg_service):
    """Test processing missing ECG file."""
    with pytest.raises(FileNotFoundError):
        await ecg_service.process_ecg_file("/tmp/nonexistent.txt")


@pytest.mark.asyncio
async def test_get_analysis_by_id(ecg_service, test_db):
    """Test retrieving ECG analysis by ID."""
    analysis = ECGAnalysis(
        patient_id=1,
        file_path="/tmp/test.txt",
        classification="normal",
        confidence=0.95,
        rhythm="sinus",
        heart_rate=72,
        signal_quality=0.88
    )
    test_db.add(analysis)
    await test_db.commit()
    await test_db.refresh(analysis)
    
    result = await ecg_service.get_analysis_by_id(analysis.id)
    
    assert result is not None
    assert result.id == analysis.id
    assert result.classification == "normal"


@pytest.mark.asyncio
async def test_get_analysis_by_id_not_found(ecg_service):
    """Test retrieving non-existent ECG analysis."""
    result = await ecg_service.get_analysis_by_id(99999)
    assert result is None


@pytest.mark.asyncio
async def test_get_analyses_by_patient(ecg_service, test_db):
    """Test retrieving ECG analyses by patient ID."""
    patient_id = 1
    
    for i in range(3):
        analysis = ECGAnalysis(
            patient_id=patient_id,
            file_path=f"/tmp/test_{i}.txt",
            classification="normal",
            confidence=0.95,
            rhythm="sinus",
            heart_rate=72 + i,
            signal_quality=0.88
        )
        test_db.add(analysis)
    
    await test_db.commit()
    
    results = await ecg_service.get_analyses_by_patient(patient_id)
    
    assert len(results) == 3
    assert all(r.patient_id == patient_id for r in results)


@pytest.mark.asyncio
async def test_update_analysis_status(ecg_service, test_db):
    """Test updating ECG analysis status."""
    analysis = ECGAnalysis(
        patient_id=1,
        file_path="/tmp/test.txt",
        classification="normal",
        confidence=0.95,
        rhythm="sinus",
        heart_rate=72,
        signal_quality=0.88,
        status="pending"
    )
    test_db.add(analysis)
    await test_db.commit()
    await test_db.refresh(analysis)
    
    updated = await ecg_service.update_analysis_status(analysis.id, "completed")
    
    assert updated is not None
    assert updated.status == "completed"


@pytest.mark.asyncio
async def test_delete_analysis(ecg_service, test_db):
    """Test deleting ECG analysis."""
    analysis = ECGAnalysis(
        patient_id=1,
        file_path="/tmp/test.txt",
        classification="normal",
        confidence=0.95,
        rhythm="sinus",
        heart_rate=72,
        signal_quality=0.88
    )
    test_db.add(analysis)
    await test_db.commit()
    await test_db.refresh(analysis)
    
    success = await ecg_service.delete_analysis(analysis.id)
    assert success is True
    
    deleted = await ecg_service.get_analysis_by_id(analysis.id)
    assert deleted is None


@pytest.mark.asyncio
async def test_validate_signal_quality_good(ecg_service):
    """Test signal quality validation for good quality signal."""
    quality_metrics = {
        "snr": 20.0,
        "baseline_drift": 0.01,
        "artifacts": 0.05,
        "completeness": 0.98
    }
    
    is_valid, score = await ecg_service.validate_signal_quality(quality_metrics)
    
    assert is_valid is True
    assert score > 0.8


@pytest.mark.asyncio
async def test_validate_signal_quality_poor(ecg_service):
    """Test signal quality validation for poor quality signal."""
    quality_metrics = {
        "snr": 5.0,
        "baseline_drift": 0.1,
        "artifacts": 0.3,
        "completeness": 0.6
    }
    
    is_valid, score = await ecg_service.validate_signal_quality(quality_metrics)
    
    assert is_valid is False
    assert score < 0.5


@pytest.mark.asyncio
async def test_generate_report(ecg_service, test_db):
    """Test generating ECG analysis report."""
    analysis = ECGAnalysis(
        patient_id=1,
        file_path="/tmp/test.txt",
        classification="abnormal",
        confidence=0.85,
        rhythm="atrial_fibrillation",
        heart_rate=95,
        signal_quality=0.88,
        findings=["Irregular rhythm", "Fast heart rate"]
    )
    test_db.add(analysis)
    await test_db.commit()
    await test_db.refresh(analysis)
    
    report = await ecg_service.generate_report(analysis.id)
    
    assert report is not None
    assert "abnormal" in report.lower()
    assert "atrial_fibrillation" in report.lower()
    assert "95" in report


@pytest.mark.asyncio
async def test_ml_service_error_handling(ecg_service, mock_ml_service, sample_ecg_data):
    """Test handling ML service errors."""
    mock_ml_service.analyze_ecg.side_effect = Exception("ML service error")
    
    with patch('app.utils.ecg_processor.ECGProcessor.process_file') as mock_processor:
        mock_processor.return_value = {
            "signal_data": [[1, 2, 3], [4, 5, 6]],
            "quality_metrics": {"snr": 15.5, "baseline_drift": 0.02}
        }
        
        with pytest.raises(Exception, match="ML service error"):
            await ecg_service.create_ecg_analysis(sample_ecg_data)


@pytest.mark.asyncio
async def test_concurrent_analysis_processing(ecg_service, sample_ecg_data):
    """Test concurrent ECG analysis processing."""
    import asyncio
    
    with patch('app.utils.ecg_processor.ECGProcessor.process_file') as mock_processor:
        mock_processor.return_value = {
            "signal_data": [[1, 2, 3], [4, 5, 6]],
            "quality_metrics": {"snr": 15.5, "baseline_drift": 0.02}
        }
        
        tasks = []
        for i in range(3):
            ecg_data = ECGAnalysisCreate(
                patient_id=i + 1,
                original_filename=f"test_{i}.txt",
                acquisition_date="2025-06-01T14:00:00Z",
                sample_rate=500,
                duration_seconds=10.0,
                leads_count=12,
                leads_names=["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
                device_manufacturer="Test Device",
                device_model="v1.0"
            )
            tasks.append(ecg_service.create_ecg_analysis(ecg_data))
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert all(r is not None for r in results)
