#!/usr/bin/env python3
"""
Create targeted tests for HybridECGAnalysisService to achieve 80% coverage
Focus on the 29 methods identified in the file outline
"""

import os

def create_targeted_hybrid_ecg_tests():
    """Create focused tests for HybridECGAnalysisService methods"""
    
    test_content = '''"""
Targeted tests for HybridECGAnalysisService - focusing on 80% coverage
Based on actual method analysis from hybrid_ecg_service.py
"""
import pytest
import numpy as np
import numpy.typing as npt
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
import asyncio

from app.services.hybrid_ecg_service import (
    HybridECGAnalysisService, 
    UniversalECGReader, 
    AdvancedPreprocessor, 
    FeatureExtractor,
    ClinicalUrgency
)


@pytest.fixture
def mock_db():
    """Mock database session"""
    return Mock()


@pytest.fixture
def mock_validation_service():
    """Mock validation service"""
    return Mock()


@pytest.fixture
def sample_ecg_signal():
    """Sample ECG signal for testing"""
    return np.random.randn(5000, 12).astype(np.float64)


@pytest.fixture
def sample_1d_signal():
    """Sample 1D ECG signal"""
    return np.random.randn(5000).astype(np.float64)


class TestHybridECGAnalysisServiceTargeted:
    """Targeted tests for HybridECGAnalysisService - 80% coverage focus"""
    
    def test_init_with_all_parameters(self, mock_db, mock_validation_service):
        """Test initialization with all parameters"""
        service = HybridECGAnalysisService(
            db=mock_db, 
            validation_service=mock_validation_service, 
            sampling_rate=500
        )
        
        assert service.db is mock_db
        assert service.validation_service is mock_validation_service
        assert service.fs == 500
        assert hasattr(service, 'reader')
        assert hasattr(service, 'preprocessor')
        assert hasattr(service, 'feature_extractor')
        assert hasattr(service, 'ml_service')
        assert len(service.pathology_classes) == 14
    
    def test_init_default_parameters(self):
        """Test initialization with default parameters"""
        service = HybridECGAnalysisService()
        
        assert service.db is None
        assert service.validation_service is None
        assert service.fs == 250  # Default sampling rate
        assert hasattr(service, 'reader')
    
    @patch('app.services.hybrid_ecg_service.logger')
    def test_analyze_ecg_file_success(self, mock_logger, mock_db, mock_validation_service):
        """Test successful ECG file analysis"""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        
        mock_ecg_data = {
            'signal': np.random.randn(5000, 12).astype(np.float64),
            'metadata': {'patient_id': 'P001'},
            'sampling_rate': 500,
            'labels': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        }
        service.reader.read_ecg = Mock(return_value=mock_ecg_data)
        
        result = service.analyze_ecg_file('/fake/path/test.csv')
        
        assert isinstance(result, dict)
        assert 'features' in result
        assert 'predictions' in result
        assert 'metadata' in result
        assert 'processing_info' in result
        assert result['processing_info']['sampling_rate'] == 500
        assert len(result['processing_info']['leads']) == 12
    
    @patch('app.services.hybrid_ecg_service.logger')
    def test_analyze_ecg_file_exception(self, mock_logger, mock_db, mock_validation_service):
        """Test ECG file analysis with exception"""
        from app.core.exceptions import ECGProcessingException
        
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        service.reader.read_ecg = Mock(side_effect=Exception("File not found"))
        
        with pytest.raises(ECGProcessingException):
            service.analyze_ecg_file('/fake/path/nonexistent.csv')
        
        mock_logger.error.assert_called()
    
    def test_analyze_ecg_signal_with_sampling_rate(self, mock_db, mock_validation_service, sample_ecg_signal):
        """Test ECG signal analysis with custom sampling rate"""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        
        result = service.analyze_ecg_signal(sample_ecg_signal, sampling_rate=1000)
        
        assert isinstance(result, dict)
        assert 'features' in result
        assert 'predictions' in result
        assert 'processing_info' in result
        assert result['processing_info']['sampling_rate'] == 1000
        assert service.fs == 1000
        assert service.preprocessor.fs == 1000
        assert service.feature_extractor.fs == 1000
    
    def test_analyze_ecg_signal_default_sampling_rate(self, mock_db, mock_validation_service, sample_ecg_signal):
        """Test ECG signal analysis with default sampling rate"""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        
        result = service.analyze_ecg_signal(sample_ecg_signal)
        
        assert isinstance(result, dict)
        assert result['processing_info']['sampling_rate'] == 250  # Default
    
    @patch('app.services.hybrid_ecg_service.logger')
    def test_analyze_ecg_signal_exception(self, mock_logger, mock_db, mock_validation_service):
        """Test ECG signal analysis with exception"""
        from app.core.exceptions import ECGProcessingException
        
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        service.preprocessor.preprocess_signal = Mock(side_effect=Exception("Processing failed"))
        
        with pytest.raises(ECGProcessingException):
            service.analyze_ecg_signal(np.array([[1, 2, 3]]))
        
        mock_logger.error.assert_called()
    
    def test_simulate_predictions_high_heart_rate(self, mock_db, mock_validation_service):
        """Test prediction simulation with high heart rate"""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        features = {'heart_rate': 120}
        
        result = service._simulate_predictions(features)
        
        assert isinstance(result, dict)
        assert 'class_probabilities' in result
        assert 'predicted_class' in result
        assert 'confidence' in result
        assert 'pathology_detected' in result
        assert result['class_probabilities']['Atrial Fibrillation'] == 0.8
        assert result['predicted_class'] == 'Atrial Fibrillation'
        assert result['pathology_detected'] is True
    
    def test_simulate_predictions_low_heart_rate(self, mock_db, mock_validation_service):
        """Test prediction simulation with low heart rate"""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        features = {'heart_rate': 45}
        
        result = service._simulate_predictions(features)
        
        assert result['class_probabilities']['AV Block 1st Degree'] == 0.7
        assert result['predicted_class'] == 'AV Block 1st Degree'
        assert result['pathology_detected'] is True
    
    def test_simulate_predictions_normal_heart_rate(self, mock_db, mock_validation_service):
        """Test prediction simulation with normal heart rate"""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        features = {'heart_rate': 75}
        
        result = service._simulate_predictions(features)
        
        assert result['class_probabilities']['Normal'] == 0.9
        assert result['predicted_class'] == 'Normal'
        assert result['pathology_detected'] is False
    
    def test_get_supported_pathologies(self, mock_db, mock_validation_service):
        """Test getting supported pathologies"""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        
        result = service.get_supported_pathologies()
        
        assert isinstance(result, list)
        assert len(result) == 14
        assert 'normal' in result
        assert 'atrial_fibrillation' in result
        assert 'ventricular_tachycardia' in result
        assert 'stemi' in result
    
    def test_validate_signal_valid_signal(self, mock_db, mock_validation_service):
        """Test signal validation with valid signal"""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        signal = np.random.randn(1000).astype(np.float64)
        
        result = service.validate_signal(signal, sampling_rate=500)
        
        assert isinstance(result, dict)
        assert 'is_valid' in result
        assert 'quality_score' in result
        assert 'issues' in result
        assert 'sampling_rate' in result
        assert result['is_valid'] is True
        assert result['quality_score'] == 0.8
        assert result['sampling_rate'] == 500
    
    def test_validate_signal_empty_signal(self, mock_db, mock_validation_service):
        """Test signal validation with empty signal"""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        signal = np.array([])
        
        result = service.validate_signal(signal)
        
        assert result['is_valid'] is False
        assert result['quality_score'] == 0.0
        assert 'Empty signal' in result['issues']
    
    def test_validate_signal_none_signal(self, mock_db, mock_validation_service):
        """Test signal validation with None signal"""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        
        result = service.validate_signal(None)
        
        assert result['is_valid'] is False
        assert result['quality_score'] == 0.0
        assert 'Empty signal' in result['issues']
    
    def test_validate_signal_nan_values(self, mock_db, mock_validation_service):
        """Test signal validation with NaN values"""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        signal = np.array([1.0, 2.0, np.nan, 4.0])
        
        result = service.validate_signal(signal)
        
        assert result['is_valid'] is False
        assert result['quality_score'] == 0.0
        assert 'Invalid values detected' in result['issues']
    
    def test_validate_signal_inf_values(self, mock_db, mock_validation_service):
        """Test signal validation with infinite values"""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        signal = np.array([1.0, 2.0, np.inf, 4.0])
        
        result = service.validate_signal(signal)
        
        assert result['is_valid'] is False
        assert result['quality_score'] == 0.0
        assert 'Invalid values detected' in result['issues']
    
    def test_validate_signal_too_flat(self, mock_db, mock_validation_service):
        """Test signal validation with too flat signal"""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        signal = np.array([1.0, 1.01, 1.02, 1.01])  # Range < 0.1
        
        result = service.validate_signal(signal)
        
        assert result['is_valid'] is False
        assert result['quality_score'] == 0.2
        assert 'Signal too flat' in result['issues']
    
    def test_validate_signal_too_noisy(self, mock_db, mock_validation_service):
        """Test signal validation with too noisy signal"""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        signal = np.array([-5.0, 15.0, -8.0, 12.0])  # Range > 10
        
        result = service.validate_signal(signal)
        
        assert result['is_valid'] is True  # Still valid but with warning
        assert result['quality_score'] == 0.5
        assert 'High amplitude detected' in result['issues']
    
    @patch('app.services.hybrid_ecg_service.logger')
    def test_validate_signal_exception(self, mock_logger, mock_db, mock_validation_service):
        """Test signal validation with exception"""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        
        with patch('numpy.max', side_effect=Exception("Calculation failed")):
            result = service.validate_signal(np.array([1, 2, 3]))
        
        assert result['is_valid'] is False
        assert result['quality_score'] == 0.0
        assert 'Validation failed' in result['issues']
        mock_logger.error.assert_called()
    
    def test_analyze_ecg_comprehensive_with_file_path(self, mock_db, mock_validation_service):
        """Test comprehensive ECG analysis with file path"""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        
        result = service.analyze_ecg_comprehensive(
            file_path='/fake/path/test.csv',
            patient_id=123,
            analysis_id='TEST_001'
        )
        
        assert isinstance(result, dict)
        assert result['analysis_id'] == 'TEST_001'
        assert result['patient_id'] == 123
        assert 'timestamp' in result
        assert 'heart_rate' in result
        assert 'predictions' in result
        assert 'features' in result
        assert result['clinical_significance'] == 'normal'
        assert result['urgency_level'] == 'routine'
    
    def test_analyze_ecg_comprehensive_with_ecg_data(self, mock_db, mock_validation_service):
        """Test comprehensive ECG analysis with ECG data"""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        
        ecg_data = {
            'signal': np.random.randn(5000, 12),
            'sampling_rate': 500
        }
        
        result = service.analyze_ecg_comprehensive(
            ecg_data=ecg_data,
            patient_id=456
        )
        
        assert result['patient_id'] == 456
        assert 'analysis_id' in result
    
    def test_analyze_ecg_comprehensive_no_input(self, mock_db, mock_validation_service):
        """Test comprehensive ECG analysis with no input"""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        
        result = service.analyze_ecg_comprehensive()
        
        assert 'error' in result
    
    @patch('app.services.hybrid_ecg_service.logger')
    def test_analyze_ecg_comprehensive_exception(self, mock_logger, mock_db, mock_validation_service):
        """Test comprehensive ECG analysis with exception"""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        
        with patch('app.services.hybrid_ecg_service.datetime') as mock_datetime:
            mock_datetime.now.side_effect = Exception("Time error")
            
            result = service.analyze_ecg_comprehensive(file_path='/fake/path')
        
        assert 'error' in result
        mock_logger.error.assert_called()
    
    def test_run_simplified_analysis(self, mock_db, mock_validation_service, sample_1d_signal):
        """Test simplified analysis"""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        
        result = service._run_simplified_analysis(sample_1d_signal)
        
        assert isinstance(result, dict)
        assert 'features' in result
        assert 'predictions' in result
        assert 'heart_rate' in result
        assert 'rhythm' in result
        assert result['rhythm'] == 'normal_sinus_rhythm'
    
    @patch('app.services.hybrid_ecg_service.logger')
    def test_run_simplified_analysis_exception(self, mock_logger, mock_db, mock_validation_service):
        """Test simplified analysis with exception"""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        service.feature_extractor.extract_all_features = Mock(side_effect=Exception("Feature extraction failed"))
        
        result = service._run_simplified_analysis(np.array([1, 2, 3]))
        
        assert 'error' in result
        mock_logger.error.assert_called()
    
    def test_detect_pathologies(self, mock_db, mock_validation_service):
        """Test pathology detection"""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        features = {'heart_rate': 75, 'qt_interval': 450}
        
        result = service._detect_pathologies(features)
        
        assert isinstance(result, dict)
        assert 'atrial_fibrillation' in result
        assert 'long_qt' in result
    
    @patch('app.services.hybrid_ecg_service.logger')
    def test_detect_pathologies_exception(self, mock_logger, mock_db, mock_validation_service):
        """Test pathology detection with exception"""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        service._detect_atrial_fibrillation = Mock(side_effect=Exception("AF detection failed"))
        
        result = service._detect_pathologies({'heart_rate': 75})
        
        assert result == {}
        mock_logger.error.assert_called()
    
    def test_get_supported_formats(self, mock_db, mock_validation_service):
        """Test getting supported formats"""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        
        result = service.get_supported_formats()
        
        assert result is not None
    
    def test_read_ecg_file_fallback(self, mock_db, mock_validation_service):
        """Test ECG file fallback reading"""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        
        result = service._read_ecg_file_fallback('/fake/path')
        
        assert result is None


class TestClinicalUrgency:
    """Test ClinicalUrgency enum"""
    
    def test_clinical_urgency_values(self):
        """Test clinical urgency enum values"""
        assert ClinicalUrgency.LOW == "low"
        assert ClinicalUrgency.MEDIUM == "medium"
        assert ClinicalUrgency.HIGH == "high"
        assert ClinicalUrgency.CRITICAL == "critical"


@pytest.mark.asyncio
class TestHybridECGAnalysisServiceAsync:
    """Test async methods of HybridECGAnalysisService"""
    
    async def test_analyze_ecg_comprehensive_async_with_ecg_data(self, mock_db, mock_validation_service):
        """Test async comprehensive ECG analysis with ECG data"""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        
        ecg_data = {
            'signal': np.random.randn(5000, 12),
            'sampling_rate': 500
        }
        
        result = await service.analyze_ecg_comprehensive_async(
            ecg_data=ecg_data,
            patient_id=789,
            analysis_id='ASYNC_001'
        )
        
        assert isinstance(result, dict)
        assert result['patient_id'] == 789
        assert result['analysis_id'] == 'ASYNC_001'
        assert 'abnormalities' in result
        assert 'pathologies' in result
        assert 'clinical_assessment' in result
    
    async def test_analyze_ecg_comprehensive_async_with_file_path(self, mock_db, mock_validation_service):
        """Test async comprehensive ECG analysis with file path"""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        
        mock_signal_data = {
            'signal': np.random.randn(5000, 12),
            'sampling_rate': 500
        }
        service.ecg_reader.read_ecg = Mock(return_value=mock_signal_data)
        
        result = await service.analyze_ecg_comprehensive_async(
            file_path='/fake/path/test.csv',
            patient_id=999
        )
        
        assert result['patient_id'] == 999
        assert 'metadata' in result
    
    async def test_analyze_ecg_comprehensive_async_no_input(self, mock_db, mock_validation_service):
        """Test async comprehensive ECG analysis with no input"""
        from app.core.exceptions import ECGProcessingException
        
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        
        with pytest.raises(ECGProcessingException):
            await service.analyze_ecg_comprehensive_async()
    
    async def test_analyze_ecg_comprehensive_async_empty_signal_data(self, mock_db, mock_validation_service):
        """Test async comprehensive ECG analysis with empty signal data"""
        from app.core.exceptions import ECGProcessingException
        
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        service.ecg_reader.read_ecg = Mock(return_value=None)
        
        with pytest.raises(ECGProcessingException):
            await service.analyze_ecg_comprehensive_async(file_path='/fake/path')
    
    async def test_assess_signal_quality(self, mock_db, mock_validation_service, sample_1d_signal):
        """Test signal quality assessment"""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        
        result = await service._assess_signal_quality(sample_1d_signal)
        
        assert isinstance(result, dict)
        assert 'snr' in result
        assert 'snr_classification' in result
        assert isinstance(result['snr'], float)
        assert result['snr_classification'] in ['good', 'acceptable', 'poor']
'''
    
    test_file_path = "tests/test_hybrid_ecg_service_targeted_80_coverage.py"
    with open(test_file_path, 'w') as f:
        f.write(test_content)
    
    print(f"✅ Created targeted test file: {test_file_path}")
    print(f"📊 Test methods created: 35+ targeted methods")
    print(f"🎯 Focus: HybridECGAnalysisService 29 methods for 80% coverage")

if __name__ == "__main__":
    create_targeted_hybrid_ecg_tests()
