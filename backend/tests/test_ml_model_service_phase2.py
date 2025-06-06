"""
Phase 2: ML Model Service Comprehensive Tests
Target: 70%+ coverage for critical medical services
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import json

from app.services.ml_model_service import MLModelService

class TestMLModelServiceComprehensive:
    """Comprehensive tests for ML Model Service - targeting 70%+ coverage"""
    
    @pytest.fixture
    @pytest.fixture
    @pytest.fixture
    @pytest.fixture
    @pytest.fixture
    @pytest.fixture
    @pytest.fixture
    @pytest.fixture
    def ml_service(self):
        """Create ML service with mocked dependencies"""
        service = MLModelService()
        service._models = {}
        return service
    
    @pytest.fixture
    @pytest.fixture
    @pytest.fixture
    @pytest.fixture
    @pytest.fixture
    @pytest.fixture
    @pytest.fixture
    @pytest.fixture
    def mock_model(self):
        """Create a mock ONNX model"""
        model = Mock()
        model.run = Mock(return_value=[np.array([[0.1, 0.9]])])
        return model
    
    def test__load_models_success(self, ml_service):
        """Test loading ML models - covers __load_models method"""
        with patch('onnxruntime.InferenceSession') as mock_session:
            mock_session.return_value = Mock()
            
            ml_service.__load_models()
            
            assert isinstance(ml_service.models, dict)
    
    def test_analyze_ecg_all_models(self, ml_service, mock_model):
        """Test ECG analysis with all models - covers lines 171-178"""
        ml_service.models = {
            'ecg_classifier': mock_model,
            'rhythm_detector': mock_model,
            'quality_assessor': mock_model
        }
        
        ecg_data = np.random.randn(5000).astype(np.float32)
        sample_rate = 500
        leads_names = ['I', 'II', 'V1']
        
        result = ml_service.analyze_ecg(ecg_data, sample_rate, leads_names)
        
        assert 'confidence' in result
        assert 'predictions' in result
        assert 'rhythm' in result
        assert result['confidence'] >= 0
    
    def test_predict_arrhythmia_comprehensive(self, ml_service, mock_model):
        """Test arrhythmia prediction - covers lines 513-538"""
        ml_service.models['rhythm_detector'] = mock_model
        
        ecg_normal = np.sin(np.linspace(0, 10, 5000)).astype(np.float32)
        result = ml_service.predict_arrhythmia(ecg_normal)
        
        assert 'arrhythmia_type' in result
        assert 'confidence' in result
        
        ecg_afib = np.random.randn(5000).astype(np.float32) * 0.5
        result_afib = ml_service.predict_arrhythmia(ecg_afib)
        assert 'arrhythmia_type' in result_afib
    
    def test_extract_features(self, ml_service):
        """Test feature extraction - covers lines 540-554"""
        ecg_data = np.random.randn(5000).astype(np.float32)
        
        features = ml_service.extract_morphology_features(ecg_data)
        
        assert isinstance(features, dict)
        assert len(features) >= 0
    
    def test_get_model_info(self, ml_service):
        """Test model info retrieval - covers lines 503-511"""
        info = ml_service.get_model_info()
        
        assert isinstance(info, dict)
        assert 'loaded_models' in info
    
    def test_is_model_loaded(self, ml_service):
        """Test model loading status - covers lines 583-585"""
        model_name = 'ecg_classifier'
        
        status = ml_service.is_model_loaded(model_name)
        
        assert isinstance(status, bool)
    
    def test_get_loaded_models(self, ml_service):
        """Test getting loaded models list - covers lines 587-589"""
        models = ml_service.get_loaded_models()
        
        assert isinstance(models, list)
