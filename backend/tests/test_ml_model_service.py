"""Test ML Model Service."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from app.services.ml_model_service import MLModelService


@pytest.fixture
def mock_onnx_session():
    """Mock ONNX Runtime session."""
    session = Mock()
    session.run.return_value = [
        np.array([[0.1, 0.9]]),  # Classification probabilities
        np.array([[0.2, 0.8]])   # Rhythm probabilities
    ]
    return session


@pytest.fixture
def ml_service():
    """Create ML model service instance."""
    with patch('onnxruntime.InferenceSession'):
        service = MLModelService()
        service.classification_model = Mock()
        service.rhythm_model = Mock()
        service.is_loaded = True
        return service


@pytest.fixture
def sample_ecg_data():
    """Sample ECG signal data."""
    return {
        "signal_data": np.random.randn(12, 5000).tolist(),  # 12 leads, 5000 samples
        "sampling_rate": 500,
        "duration": 10.0
    }


@pytest.mark.asyncio
async def test_load_models_success(ml_service):
    """Test successful model loading."""
    with patch('onnxruntime.InferenceSession') as mock_session:
        mock_session.return_value = Mock()
        
        await ml_service.load_models()
        
        assert ml_service.is_loaded is True
        assert ml_service.classification_model is not None
        assert ml_service.rhythm_model is not None


@pytest.mark.asyncio
async def test_load_models_file_not_found():
    """Test model loading with missing files."""
    service = MLModelService()
    
    with patch('onnxruntime.InferenceSession', side_effect=FileNotFoundError("Model not found")):
        with pytest.raises(FileNotFoundError):
            await service.load_models()


@pytest.mark.asyncio
async def test_analyze_ecg_success(ml_service, sample_ecg_data, mock_onnx_session):
    """Test successful ECG analysis."""
    ml_service.classification_model = mock_onnx_session
    ml_service.rhythm_model = mock_onnx_session
    
    with patch.object(ml_service, '_preprocess_signal') as mock_preprocess:
        mock_preprocess.return_value = np.random.randn(1, 12, 5000)
        
        result = await ml_service.analyze_ecg(sample_ecg_data)
        
        assert result is not None
        assert "classification" in result
        assert "confidence" in result
        assert "rhythm" in result
        assert "quality_score" in result
        assert 0 <= result["confidence"] <= 1
        assert 0 <= result["quality_score"] <= 1


@pytest.mark.asyncio
async def test_analyze_ecg_not_loaded():
    """Test ECG analysis when models not loaded."""
    service = MLModelService()
    service.is_loaded = False
    
    with pytest.raises(RuntimeError, match="Models not loaded"):
        await service.analyze_ecg({"signal_data": [[1, 2, 3]]})


@pytest.mark.asyncio
async def test_analyze_ecg_invalid_data(ml_service):
    """Test ECG analysis with invalid data."""
    invalid_data = {"signal_data": "invalid"}
    
    with pytest.raises(ValueError):
        await ml_service.analyze_ecg(invalid_data)


@pytest.mark.asyncio
async def test_get_interpretability_map(ml_service, sample_ecg_data):
    """Test generating interpretability map."""
    with patch.object(ml_service, '_generate_attention_weights') as mock_attention:
        mock_attention.return_value = np.random.randn(12, 100)
        
        with patch.object(ml_service, '_calculate_feature_importance') as mock_features:
            mock_features.return_value = {
                "heart_rate": 0.8,
                "qrs_width": 0.6,
                "pr_interval": 0.4
            }
            
            result = await ml_service.get_interpretability_map(sample_ecg_data)
            
            assert result is not None
            assert "attention_weights" in result
            assert "feature_importance" in result
            assert len(result["attention_weights"]) == 12


@pytest.mark.asyncio
async def test_assess_signal_quality(ml_service, sample_ecg_data):
    """Test signal quality assessment."""
    with patch.object(ml_service, '_calculate_snr') as mock_snr:
        mock_snr.return_value = 15.5
        
        with patch.object(ml_service, '_detect_artifacts') as mock_artifacts:
            mock_artifacts.return_value = 0.05
            
            quality_score = await ml_service.assess_signal_quality(sample_ecg_data)
            
            assert 0 <= quality_score <= 1


@pytest.mark.asyncio
async def test_preprocess_signal(ml_service):
    """Test signal preprocessing."""
    raw_signal = np.random.randn(12, 5000)
    
    processed = ml_service._preprocess_signal(raw_signal, sampling_rate=500)
    
    assert processed.shape[0] == 1  # Batch dimension
    assert processed.shape[1] == 12  # Number of leads
    assert processed.dtype == np.float32


@pytest.mark.asyncio
async def test_postprocess_predictions(ml_service):
    """Test prediction postprocessing."""
    class_probs = np.array([[0.1, 0.9]])
    rhythm_probs = np.array([[0.2, 0.8]])
    
    result = ml_service._postprocess_predictions(class_probs, rhythm_probs)
    
    assert result["classification"] in ["normal", "abnormal"]
    assert result["rhythm"] in ["sinus", "atrial_fibrillation", "other"]
    assert 0 <= result["confidence"] <= 1


@pytest.mark.asyncio
async def test_calculate_feature_importance(ml_service):
    """Test feature importance calculation."""
    signal_data = np.random.randn(12, 5000)
    
    with patch('app.utils.signal_quality.calculate_heart_rate') as mock_hr:
        mock_hr.return_value = 72
        
        with patch('app.utils.signal_quality.calculate_qrs_width') as mock_qrs:
            mock_qrs.return_value = 0.08
            
            importance = ml_service._calculate_feature_importance(signal_data)
            
            assert isinstance(importance, dict)
            assert "heart_rate" in importance
            assert all(0 <= v <= 1 for v in importance.values())


@pytest.mark.asyncio
async def test_generate_attention_weights(ml_service):
    """Test attention weights generation."""
    signal_data = np.random.randn(12, 5000)
    
    weights = ml_service._generate_attention_weights(signal_data)
    
    assert weights.shape[0] == 12  # Number of leads
    assert weights.shape[1] > 0    # Time dimension
    assert np.all(weights >= 0)    # Non-negative weights


@pytest.mark.asyncio
async def test_memory_monitoring(ml_service):
    """Test memory usage monitoring."""
    with patch('psutil.Process') as mock_process:
        mock_process.return_value.memory_info.return_value.rss = 1024 * 1024 * 100  # 100MB
        
        memory_usage = ml_service.get_memory_usage()
        
        assert memory_usage > 0
        assert isinstance(memory_usage, (int, float))


@pytest.mark.asyncio
async def test_model_metadata(ml_service):
    """Test model metadata retrieval."""
    with patch.object(ml_service.classification_model, 'get_inputs') as mock_inputs:
        mock_inputs.return_value = [Mock(name="input", shape=[1, 12, 5000])]
        
        with patch.object(ml_service.classification_model, 'get_outputs') as mock_outputs:
            mock_outputs.return_value = [Mock(name="output", shape=[1, 2])]
            
            metadata = ml_service.get_model_metadata()
            
            assert "classification_model" in metadata
            assert "input_shape" in metadata["classification_model"]
            assert "output_shape" in metadata["classification_model"]


@pytest.mark.asyncio
async def test_batch_analysis(ml_service, mock_onnx_session):
    """Test batch ECG analysis."""
    ml_service.classification_model = mock_onnx_session
    ml_service.rhythm_model = mock_onnx_session
    
    batch_data = [
        {"signal_data": np.random.randn(12, 5000).tolist(), "sampling_rate": 500},
        {"signal_data": np.random.randn(12, 5000).tolist(), "sampling_rate": 500},
        {"signal_data": np.random.randn(12, 5000).tolist(), "sampling_rate": 500}
    ]
    
    with patch.object(ml_service, '_preprocess_signal') as mock_preprocess:
        mock_preprocess.return_value = np.random.randn(1, 12, 5000)
        
        results = await ml_service.analyze_batch(batch_data)
        
        assert len(results) == 3
        assert all("classification" in r for r in results)
        assert all("confidence" in r for r in results)


@pytest.mark.asyncio
async def test_model_performance_metrics(ml_service):
    """Test model performance metrics calculation."""
    predictions = [
        {"classification": "normal", "confidence": 0.95},
        {"classification": "abnormal", "confidence": 0.85},
        {"classification": "normal", "confidence": 0.92}
    ]
    
    ground_truth = ["normal", "abnormal", "normal"]
    
    metrics = ml_service.calculate_performance_metrics(predictions, ground_truth)
    
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert all(0 <= v <= 1 for v in metrics.values())


@pytest.mark.asyncio
async def test_model_calibration(ml_service):
    """Test model confidence calibration."""
    confidences = [0.9, 0.8, 0.7, 0.6, 0.5]
    accuracies = [0.95, 0.85, 0.75, 0.65, 0.55]
    
    calibration_error = ml_service.calculate_calibration_error(confidences, accuracies)
    
    assert isinstance(calibration_error, float)
    assert calibration_error >= 0


@pytest.mark.asyncio
async def test_error_handling_corrupted_model(ml_service, sample_ecg_data):
    """Test handling of corrupted model files."""
    ml_service.classification_model.run.side_effect = Exception("Model corrupted")
    
    with pytest.raises(Exception, match="Model corrupted"):
        await ml_service.analyze_ecg(sample_ecg_data)


@pytest.mark.asyncio
async def test_concurrent_inference(ml_service, mock_onnx_session):
    """Test concurrent model inference."""
    import asyncio
    
    ml_service.classification_model = mock_onnx_session
    ml_service.rhythm_model = mock_onnx_session
    
    with patch.object(ml_service, '_preprocess_signal') as mock_preprocess:
        mock_preprocess.return_value = np.random.randn(1, 12, 5000)
        
        tasks = []
        for i in range(5):
            ecg_data = {
                "signal_data": np.random.randn(12, 5000).tolist(),
                "sampling_rate": 500
            }
            tasks.append(ml_service.analyze_ecg(ecg_data))
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all(r is not None for r in results)
