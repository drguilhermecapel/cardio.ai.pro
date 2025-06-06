"""
Focused tests for HybridECGAnalysisService - targeting critical paths for 80% coverage.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from app.services.hybrid_ecg_service import HybridECGAnalysisService, UniversalECGReader, AdvancedPreprocessor, FeatureExtractor


@pytest.fixture
def ecg_sample_data():
    """Simple ECG sample data fixture."""
    return {
        'signal': np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64),
        'sampling_rate': 500,
        'labels': ['I'],
        'metadata': {}
    }


@pytest.fixture
def mock_db():
    """Mock database session."""
    return Mock()


@pytest.fixture
def mock_validation_service():
    """Mock validation service."""
    return Mock()


class TestUniversalECGReader:
    """Test UniversalECGReader critical paths."""
    
    def test_init_supported_formats(self):
        """Test initialization includes all supported formats."""
        reader = UniversalECGReader()
        
        assert '.csv' in reader.supported_formats
        assert '.edf' in reader.supported_formats
        assert '.dat' in reader.supported_formats
        assert '.png' in reader.supported_formats
    
    def test_read_ecg_none_input(self):
        """Test read_ecg with None input."""
        reader = UniversalECGReader()
        result = reader.read_ecg(None)
        assert result is None
    
    def test_read_ecg_empty_string(self):
        """Test read_ecg with empty string."""
        reader = UniversalECGReader()
        result = reader.read_ecg('')
        assert result is None
    
    def test_read_ecg_unsupported_format(self):
        """Test read_ecg with unsupported format."""
        reader = UniversalECGReader()
        
        with pytest.raises(ValueError, match="Unsupported format"):
            reader.read_ecg('/fake/test.xyz')
    
    def test_read_image_placeholder(self):
        """Test PNG image reading placeholder."""
        reader = UniversalECGReader()
        
        result = reader._read_image('/fake/test.png')
        assert isinstance(result, dict)
        assert 'signal' in result
        assert 'sampling_rate' in result
        assert 'labels' in result


class TestAdvancedPreprocessor:
    """Test AdvancedPreprocessor critical paths."""
    
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        preprocessor = AdvancedPreprocessor()
        
        assert preprocessor.fs == 250  # Default sampling rate
    
    def test_preprocess_signal_basic(self):
        """Test basic signal preprocessing."""
        preprocessor = AdvancedPreprocessor()
        signal = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        
        result = await preprocessor.preprocess_signal(signal)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
    
    def test_remove_baseline_wandering(self):
        """Test baseline wandering removal."""
        preprocessor = AdvancedPreprocessor()
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        
        result = preprocessor._remove_baseline_wandering(signal)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(signal)


class TestFeatureExtractor:
    """Test FeatureExtractor critical paths."""
    
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        extractor = FeatureExtractor()
        
        assert extractor.fs == 250  # Default sampling rate
    
    def test_extract_all_features_basic(self):
        """Test basic feature extraction."""
        extractor = FeatureExtractor()
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        
        result = extractor.extract_all_features(signal)
        
        assert isinstance(result, dict)
        assert 'signal_mean' in result
        assert 'signal_std' in result
    
    def test_detect_r_peaks(self):
        """Test R peak detection."""
        extractor = FeatureExtractor()
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        
        result = extractor._detect_r_peaks(signal)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int64


class TestHybridECGAnalysisService:
    """Test HybridECGAnalysisService critical paths."""
    
    def test_init_service(self, mock_db, mock_validation_service):
        """Test service initialization."""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        
        assert service.db is mock_db
        assert service.validation_service is mock_validation_service
        assert hasattr(service, 'ecg_reader')
        assert hasattr(service, 'preprocessor')
        assert hasattr(service, 'feature_extractor')
    
    def test_validate_signal_valid(self, mock_db, mock_validation_service):
        """Test signal validation with valid signal."""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        signal = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        
        result = await service.validate_signal(valid_signal)
        assert result is True
    
    def test_validate_signal_invalid_empty(self, mock_db, mock_validation_service):
        """Test signal validation with empty signal."""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        signal = np.array([], dtype=np.float64)
        
        result = await service.validate_signal(valid_signal)
        assert result is False
    
    def test_get_supported_pathologies(self, mock_db, mock_validation_service):
        """Test getting supported pathologies."""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        
        result = await service.get_supported_pathologies()
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert 'Atrial Fibrillation' in result
    
    def test_simulate_predictions(self, mock_db, mock_validation_service):
        """Test prediction simulation."""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        features = {'heart_rate': 75.0}
        
        result = await service._simulate_predictions(features)
        
        assert isinstance(result, dict)
        assert 'class_probabilities' in result
        assert 'confidence' in result
    
    @pytest.mark.asyncio
    async def test_analyze_ecg_comprehensive_with_data(self, mock_db, mock_validation_service, ecg_sample_data):
        """Test comprehensive ECG analysis with direct data."""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        
        result = await service.analyze_ecg_comprehensive(
            ecg_data=ecg_sample_data,
            patient_id=1,
            analysis_id="test_123"
        )
        
        assert isinstance(result, dict)
        assert 'abnormalities' in result
        assert 'patient_id' in result
        assert result['patient_id'] == 1
    
    @pytest.mark.asyncio
    async def test_analyze_ecg_comprehensive_no_input(self, mock_db, mock_validation_service):
        """Test comprehensive ECG analysis with no input."""
        from app.core.exceptions import ECGProcessingException
        
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        
        with pytest.raises(ECGProcessingException):
            await service.analyze_ecg_comprehensive()
    
    @pytest.mark.asyncio
    async def test_assess_signal_quality(self, mock_db, mock_validation_service):
        """Test signal quality assessment."""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        signal = np.random.randn(1000).astype(np.float64)
        
        result = await service._assess_signal_quality(signal)
        
        assert isinstance(result, dict)
        assert 'snr' in result
        assert 'snr_classification' in result
        assert isinstance(result['snr'], float)
        assert result['snr_classification'] in ['good', 'acceptable', 'poor']
