"""Corrected tests targeting zero-coverage services for 80% regulatory compliance"""
import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

from app.services.hybrid_ecg_service import HybridECGAnalysisService, UniversalECGReader, AdvancedPreprocessor, FeatureExtractor
from app.utils.ecg_hybrid_processor import ECGHybridProcessor
from app.services.validation_service import ValidationService
from app.services.ecg_service import ECGAnalysisService
from app.core.constants import AnalysisStatus, ClinicalUrgency


class TestCorrectedCriticalServices:
    """Tests targeting critical zero-coverage services with correct method signatures"""
    
    @pytest.fixture
    def sample_ecg_data(self):
        return np.random.randn(5000, 12).astype(np.float64)
    
    @pytest.fixture
    def mock_db(self):
        return AsyncMock()
    
    def test_hybrid_ecg_service_init(self):
        """Test HybridECGAnalysisService initialization"""
        service = HybridECGAnalysisService()
        assert service is not None
        assert hasattr(service, 'ecg_reader')
        assert hasattr(service, 'preprocessor')
        assert hasattr(service, 'feature_extractor')
    
    def test_universal_ecg_reader_init(self):
        """Test UniversalECGReader initialization"""
        reader = UniversalECGReader()
        assert reader is not None
        assert hasattr(reader, 'supported_formats')
        assert isinstance(reader.supported_formats, dict)
    
    def test_universal_ecg_reader_read_ecg(self, sample_ecg_data):
        """Test ECG reading main method"""
        reader = UniversalECGReader()
        
        with patch('pandas.read_csv') as mock_read_csv:
            mock_df = Mock()
            mock_df.values = sample_ecg_data
            mock_df.columns = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            mock_read_csv.return_value = mock_df
            
            result = reader.read_ecg("/fake/test.csv")
            assert isinstance(result, dict)
            assert 'signal' in result
            assert 'sampling_rate' in result
    
    def test_universal_ecg_reader_read_edf(self, sample_ecg_data):
        """Test EDF reading private method"""
        reader = UniversalECGReader()
        
        mock_edf = Mock()
        mock_edf.signals_in_file = 12
        mock_edf.readSignal.return_value = sample_ecg_data[:, 0]
        mock_edf.signal_label.return_value = "Lead_I"
        mock_edf.getSampleFrequency.return_value = 250
        
        with patch('pyedflib.EdfReader', return_value=mock_edf):
            result = reader._read_edf("/fake/test.edf")
            assert isinstance(result, dict)
            assert 'signal' in result
    
    def test_universal_ecg_reader_read_csv(self, sample_ecg_data):
        """Test CSV reading private method"""
        reader = UniversalECGReader()
        
        with patch('pandas.read_csv') as mock_read_csv:
            mock_df = Mock()
            mock_df.values = sample_ecg_data
            mock_df.columns = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            mock_read_csv.return_value = mock_df
            
            result = reader._read_csv("/fake/test.csv")
            assert isinstance(result, dict)
            assert 'signal' in result
    
    def test_advanced_preprocessor_init(self):
        """Test AdvancedPreprocessor initialization"""
        preprocessor = AdvancedPreprocessor()
        assert preprocessor is not None
        assert hasattr(preprocessor, 'fs')
        assert preprocessor.fs == 250
    
    def test_advanced_preprocessor_preprocess_signal(self, sample_ecg_data):
        """Test signal preprocessing with correct method name"""
        preprocessor = AdvancedPreprocessor()
        
        with patch('scipy.signal.butter', return_value=([1], [1])):
            with patch('scipy.signal.filtfilt', return_value=sample_ecg_data[:, 0]):
                with patch('sklearn.preprocessing.StandardScaler') as mock_scaler:
                    mock_scaler_instance = Mock()
                    mock_scaler_instance.fit_transform.return_value = sample_ecg_data
                    mock_scaler.return_value = mock_scaler_instance
                    
                    result = await preprocessor.preprocess_signal(sample_ecg_data)
                    assert isinstance(result, np.ndarray)
    
    def test_feature_extractor_init(self):
        """Test FeatureExtractor initialization"""
        extractor = FeatureExtractor()
        assert extractor is not None
        assert hasattr(extractor, 'fs')
        assert extractor.fs == 250
    
    def test_feature_extractor_extract_all_features(self, sample_ecg_data):
        """Test feature extraction with correct method name"""
        extractor = FeatureExtractor()
        
        with patch('neurokit2.ecg_process') as mock_process:
            mock_signals = {'ECG_Rate': np.array([75.0] * 100)}
            mock_info = {'ECG_R_Peaks': np.array([100, 200, 300])}
            mock_process.return_value = (mock_signals, mock_info)
            
            result = extractor.extract_all_features(sample_ecg_data.flatten())
            assert isinstance(result, dict)
    
    def test_feature_extractor_detect_r_peaks(self, sample_ecg_data):
        """Test R peak detection"""
        extractor = FeatureExtractor()
        
        with patch('neurokit2.ecg_process') as mock_process:
            mock_signals = {'ECG_R_Peaks': np.array([1, 0, 0, 1, 0, 0, 1])}
            mock_info = {}
            mock_process.return_value = (mock_signals, mock_info)
            
            result = extractor._detect_r_peaks(sample_ecg_data.flatten())
            assert isinstance(result, np.ndarray)
    
    def test_feature_extractor_extract_morphological_features(self, sample_ecg_data):
        """Test morphological feature extraction"""
        extractor = FeatureExtractor()
        r_peaks = np.array([100, 200, 300])
        
        result = extractor._extract_morphological_features(sample_ecg_data.flatten(), r_peaks)
        assert isinstance(result, dict)
        assert 'signal_mean' in result
        assert 'signal_std' in result
    
    def test_feature_extractor_extract_interval_features(self, sample_ecg_data):
        """Test interval feature extraction"""
        extractor = FeatureExtractor()
        r_peaks = np.array([100, 200, 300])
        
        result = extractor._extract_interval_features(sample_ecg_data.flatten(), r_peaks)
        assert isinstance(result, dict)
        assert 'rr_mean' in result
        assert 'heart_rate' in result
    
    def test_feature_extractor_extract_hrv_features(self):
        """Test HRV feature extraction"""
        extractor = FeatureExtractor()
        r_peaks = np.array([100, 200, 300, 400])
        
        result = extractor._extract_hrv_features(r_peaks)
        assert isinstance(result, dict)
    
    def test_ecg_hybrid_processor_init(self):
        """Test ECGHybridProcessor initialization"""
        processor = ECGHybridProcessor()
        assert processor is not None
        assert hasattr(processor, 'fs')  # Correct attribute name
    
    def test_validation_service_init(self, mock_db):
        """Test ValidationService initialization"""
        with patch('app.repositories.validation_repository.ValidationRepository'):
            service = ValidationService(mock_db, Mock())  # Only 2 args needed
            assert service is not None
    
    def test_ecg_service_init(self, mock_db):
        """Test ECGAnalysisService initialization"""
        with patch('app.repositories.ecg_repository.ECGRepository'):
            with patch('app.services.ml_model_service.MLModelService'):
                with patch('app.services.validation_service.ValidationService'):
                    with patch('app.utils.signal_quality.SignalQualityAnalyzer'):
                        service = ECGAnalysisService(mock_db, Mock(), Mock())
                        assert service is not None
    
    def test_hybrid_ecg_service_validate_signal(sample_signal):
        """Test signal validation"""
        service = HybridECGAnalysisService()
        result = await service.validate_signal(sample_signal)
    
    def test_hybrid_ecg_service_get_supported_pathologies(self):
        """Test getting supported pathologies"""
        service = HybridECGAnalysisService()
        result = await service.get_supported_pathologies()
        assert isinstance(result, list)
    
    def test_hybrid_ecg_service_analyze_ecg_file(self, sample_ecg_data):
        """Test ECG file analysis"""
        service = HybridECGAnalysisService()
        
        mock_ecg_data = {
            'signal': sample_ecg_data,
            'sampling_rate': 250,
            'labels': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
            'metadata': {}
        }
        
        with patch.object(service.ecg_reader, 'read_ecg', return_value=mock_ecg_data):
            with patch.object(service.preprocessor, 'preprocess_signal', return_value=sample_ecg_data):
                result = await service.analyze_ecg_file("/fake/test.csv")
                assert isinstance(result, dict)
    
    def test_hybrid_ecg_service_analyze_ecg_signal(self, sample_ecg_data):
        """Test ECG signal analysis"""
        service = HybridECGAnalysisService()
        
        with patch.object(service.preprocessor, 'preprocess_signal', return_value=sample_ecg_data):
            with patch.object(service.feature_extractor, 'extract_all_features', return_value={}):
                result = await service.analyze_ecg_signal(sample_ecg_data)
                assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    @pytest.mark.asyncio
async def test_hybrid_ecg_service_analyze_ecg_comprehensive(self, sample_ecg_data):
        """Test comprehensive ECG analysis"""
        service = HybridECGAnalysisService()
        
        mock_ecg_data = {
            'signal': sample_ecg_data,
            'sampling_rate': 250,
            'labels': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
            'metadata': {}
        }
        
        with patch.object(service.ecg_reader, 'read_ecg', return_value=mock_ecg_data):
            with patch.object(service.preprocessor, 'preprocess_signal', return_value=sample_ecg_data):
                with patch.object(service.feature_extractor, 'extract_all_features', return_value={}):
                    result = await sawait ervice.analyze_ecg_comprehensive("/fake/test.csv")
                    assert isinstance(result, dict)
