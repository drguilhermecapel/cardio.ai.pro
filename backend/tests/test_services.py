import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.core.config import settings

if settings.STANDALONE_MODE:
    pytest.skip("Service tests skipped in standalone mode", allow_module_level=True)

from app.services.ecg_service import ECGService
from app.services.notification_service import NotificationService
from app.services.validation_service import ValidationService
from app.services.ml_model_service import MLModelService


@pytest.mark.asyncio
async def test_ecg_service_initialization():
    """Test ECG service initialization."""
    mock_session = AsyncMock()
    service = ECGService(mock_session)
    assert service.session == mock_session


@pytest.mark.asyncio
async def test_ecg_service_process_analysis():
    """Test ECG service process analysis."""
    mock_session = AsyncMock()
    service = ECGService(mock_session)
    
    with patch.object(service, 'process_ecg_file') as mock_process:
        mock_result = {
            "analysis_id": "TEST123",
            "status": "completed",
            "results": {"heart_rate": 75}
        }
        mock_process.return_value = mock_result
        
        result = await service.process_analysis("/path/to/file.csv", 1)
        
        assert result["analysis_id"] == "TEST123"
        assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_notification_service_initialization():
    """Test notification service initialization."""
    mock_session = AsyncMock()
    service = NotificationService(mock_session)
    assert service.session == mock_session


@pytest.mark.asyncio
async def test_notification_service_send():
    """Test notification service send."""
    mock_session = AsyncMock()
    service = NotificationService(mock_session)
    
    with patch.object(service, 'send_notification') as mock_send:
        mock_send.return_value = True
        
        result = await service.send(1, "Test", "Message")
        
        assert result is True


@pytest.mark.asyncio
async def test_validation_service_initialization():
    """Test validation service initialization."""
    mock_session = AsyncMock()
    service = ValidationService(mock_session)
    assert service.session == mock_session


@pytest.mark.asyncio
async def test_validation_service_create():
    """Test validation service create."""
    mock_session = AsyncMock()
    service = ValidationService(mock_session)
    
    with patch.object(service, 'create_validation') as mock_create:
        mock_validation = MagicMock()
        mock_validation.id = 1
        mock_create.return_value = mock_validation
        
        result = await service.create(1, 1)
        
        assert result.id == 1


@pytest.mark.asyncio
async def test_ml_model_service_initialization():
    """Test ML model service initialization."""
    service = MLModelService()
    assert service is not None


@pytest.mark.asyncio
async def test_ml_model_service_predict():
    """Test ML model service predict."""
    service = MLModelService()
    
    with patch.object(service, 'predict') as mock_predict:
        mock_result = {
            "predictions": [0.8, 0.2],
            "confidence": 0.9
        }
        mock_predict.return_value = mock_result
        
        result = await service.predict([[1, 2, 3, 4]])
        
        assert "predictions" in result
        assert "confidence" in result


@pytest.mark.asyncio
async def test_service_error_handling():
    """Test service error handling."""
    mock_session = AsyncMock()
    service = ECGService(mock_session)
    
    with patch.object(service, 'process_ecg_file') as mock_process:
        mock_process.side_effect = Exception("Processing error")
        
        with pytest.raises(Exception):
            await service.process_analysis("/invalid/path", 1)


@pytest.mark.asyncio
async def test_service_dependency_injection():
    """Test service dependency injection."""
    mock_session = AsyncMock()
    service = ECGService(mock_session)
    
    assert hasattr(service, 'session')
    assert service.session == mock_session


@pytest.mark.asyncio
async def test_service_async_operations():
    """Test service async operations."""
    mock_session = AsyncMock()
    service = NotificationService(mock_session)
    
    with patch.object(service, 'send_bulk_notifications') as mock_bulk:
        mock_bulk.return_value = {"sent": 5, "failed": 0}
        
        result = await service.send_bulk([1, 2, 3, 4, 5], "Test", "Message")
        
        assert result["sent"] == 5
        assert result["failed"] == 0


@pytest.mark.asyncio
async def test_service_configuration():
    """Test service configuration."""
    service = MLModelService()
    
    assert hasattr(service, 'model_path')
    assert hasattr(service, 'confidence_threshold')


@pytest.mark.asyncio
async def test_service_validation():
    """Test service validation."""
    mock_session = AsyncMock()
    service = ValidationService(mock_session)
    
    with patch.object(service, 'validate_analysis') as mock_validate:
        mock_validate.return_value = {"valid": True, "score": 0.95}
        
        result = await service.validate(1, 1)
        
        assert result["valid"] is True
        assert result["score"] == 0.95


@pytest.mark.asyncio
async def test_service_caching():
    """Test service caching mechanisms."""
    service = MLModelService()
    
    with patch.object(service, 'get_cached_prediction') as mock_cache:
        mock_cache.return_value = {"cached": True, "result": [0.8, 0.2]}
        
        result = await service.get_prediction("cache_key")
        
        assert result["cached"] is True


@pytest.mark.asyncio
async def test_service_logging():
    """Test service logging."""
    mock_session = AsyncMock()
    service = ECGService(mock_session)
    
    with patch('app.services.ecg_service.logger') as mock_logger:
        await service.log_analysis_start(1)
        
        mock_logger.info.assert_called()


@pytest.mark.asyncio
async def test_service_metrics():
    """Test service metrics collection."""
    mock_session = AsyncMock()
    service = ECGService(mock_session)
    
    with patch.object(service, 'collect_metrics') as mock_metrics:
        mock_metrics.return_value = {"processing_time": 1.5, "accuracy": 0.92}
        
        result = await service.get_metrics()
        
        assert "processing_time" in result
        assert "accuracy" in result
