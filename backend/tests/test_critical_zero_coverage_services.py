"""Critical tests targeting zero-coverage services for 80% regulatory compliance"""
import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

from app.services.hybrid_ecg_service import HybridECGAnalysisService, UniversalECGReader, AdvancedPreprocessor, FeatureExtractor
from app.utils.ecg_hybrid_processor import ECGHybridProcessor
from app.services.validation_service import ValidationService
from app.services.ecg_service import ECGAnalysisService
from app.core.constants import AnalysisStatus, ClinicalUrgency


class TestCriticalZeroCoverageServices:
    """Tests targeting critical zero-coverage services for regulatory compliance"""
    
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
    
    @pytest.mark.asyncio
    async def test_hybrid_ecg_service_analyze_ecg_comprehensive(self, sample_ecg_data):
        """Test comprehensive ECG analysis"""
        service = HybridECGAnalysisService()
        
        with patch.object(service.ecg_reader, 'read_ecg', return_value=sample_ecg_data):
            with patch.object(service.preprocessor, 'preprocess', return_value=sample_ecg_data):
                with patch.object(service.feature_extractor, 'extract_features', return_value={}):
                    result = await service.analyze_ecg_comprehensive("/fake/test.csv")
                    assert isinstance(result, dict)
    
    def test_universal_ecg_reader_init(self):
        """Test UniversalECGReader initialization"""
        reader = UniversalECGReader()
        assert reader is not None
        assert hasattr(reader, 'supported_formats')
    
    @pytest.mark.asyncio
    async def test_universal_ecg_reader_read_csv(self, sample_ecg_data):
        """Test CSV reading"""
        reader = UniversalECGReader()
        
        with patch('numpy.loadtxt', return_value=sample_ecg_data):
            result = await reader.read_csv("/fake/test.csv")
            assert isinstance(result, np.ndarray)
    
    @pytest.mark.asyncio
    async def test_universal_ecg_reader_read_edf(self, sample_ecg_data):
        """Test EDF reading"""
        reader = UniversalECGReader()
        
        mock_edf = Mock()
        mock_edf.getNSamples.return_value = [5000] * 12
        mock_edf.readSignal.return_value = sample_ecg_data[:, 0]
        
        with patch('pyedflib.EdfReader', return_value=mock_edf):
            result = await reader.read_edf("/fake/test.edf")
            assert isinstance(result, np.ndarray)
    
    @pytest.mark.asyncio
    async def test_universal_ecg_reader_read_wfdb(self, sample_ecg_data):
        """Test WFDB reading"""
        reader = UniversalECGReader()
        
        mock_record = Mock()
        mock_record.p_signal = sample_ecg_data
        
        with patch('wfdb.rdrecord', return_value=mock_record):
            result = await reader.read_wfdb("/fake/test")
            assert isinstance(result, np.ndarray)
    
    def test_advanced_preprocessor_init(self):
        """Test AdvancedPreprocessor initialization"""
        preprocessor = AdvancedPreprocessor()
        assert preprocessor is not None
        assert hasattr(preprocessor, 'sampling_rate')
    
    @pytest.mark.asyncio
    async def test_advanced_preprocessor_preprocess(self, sample_ecg_data):
        """Test signal preprocessing"""
        preprocessor = AdvancedPreprocessor()
        
        with patch('scipy.signal.butter', return_value=([1], [1])):
            with patch('scipy.signal.filtfilt', return_value=sample_ecg_data[:, 0]):
                result = await preprocessor.preprocess(sample_ecg_data)
                assert isinstance(result, np.ndarray)
    
    @pytest.mark.asyncio
    async def test_advanced_preprocessor_remove_baseline_wander(self, sample_ecg_data):
        """Test baseline wander removal"""
        preprocessor = AdvancedPreprocessor()
        
        with patch('scipy.signal.medfilt', return_value=sample_ecg_data[:, 0]):
            result = await preprocessor.remove_baseline_wander(sample_ecg_data[:, 0])
            assert isinstance(result, np.ndarray)
    
    @pytest.mark.asyncio
    async def test_advanced_preprocessor_remove_powerline_interference(self, sample_ecg_data):
        """Test powerline interference removal"""
        preprocessor = AdvancedPreprocessor()
        
        with patch('scipy.signal.iirnotch', return_value=([1], [1])):
            with patch('scipy.signal.filtfilt', return_value=sample_ecg_data[:, 0]):
                result = await preprocessor.remove_powerline_interference(sample_ecg_data[:, 0])
                assert isinstance(result, np.ndarray)
    
    def test_feature_extractor_init(self):
        """Test FeatureExtractor initialization"""
        extractor = FeatureExtractor()
        assert extractor is not None
        assert hasattr(extractor, 'sampling_rate')
    
    @pytest.mark.asyncio
    async def test_feature_extractor_extract_features(self, sample_ecg_data):
        """Test feature extraction"""
        extractor = FeatureExtractor()
        
        with patch('neurokit2.ecg_process') as mock_process:
            mock_signals = {'ECG_Rate': np.array([75.0] * 100)}
            mock_info = {'ECG_R_Peaks': np.array([100, 200, 300])}
            mock_process.return_value = (mock_signals, mock_info)
            
            result = await extractor.extract_morphology_features(sample_ecg_data)
            assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_feature_extractor_extract_time_domain_features(self, sample_ecg_data):
        """Test time domain feature extraction"""
        extractor = FeatureExtractor()
        
        rr_intervals = np.array([0.8, 0.85, 0.82, 0.88, 0.79])
        result = await extractor.extract_time_domain_features(rr_intervals)
        assert isinstance(result, dict)
        assert "mean_rr" in result
        assert "sdnn" in result
    
    @pytest.mark.asyncio
    async def test_feature_extractor_extract_frequency_domain_features(self, sample_ecg_data):
        """Test frequency domain feature extraction"""
        extractor = FeatureExtractor()
        
        rr_intervals = np.array([0.8, 0.85, 0.82, 0.88, 0.79])
        with patch('scipy.signal.welch', return_value=(np.array([0.1, 0.2, 0.3]), np.array([1, 2, 3]))):
            result = await extractor.extract_frequency_domain_features(rr_intervals)
            assert isinstance(result, dict)
    
    def test_ecg_hybrid_processor_init(self):
        """Test ECGHybridProcessor initialization"""
        processor = ECGHybridProcessor()
        assert processor is not None
        assert hasattr(processor, 'sampling_rate')
    
    @pytest.mark.asyncio
    async def test_ecg_hybrid_processor_process_signal(self, sample_ecg_data):
        """Test signal processing"""
        processor = ECGHybridProcessor()
        
        with patch('neurokit2.ecg_clean', return_value=sample_ecg_data[:, 0]):
            result = await processor.process_signal(sample_ecg_data)
            assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_ecg_hybrid_processor_detect_arrhythmias(self, sample_ecg_data):
        """Test arrhythmia detection"""
        processor = ECGHybridProcessor()
        
        with patch('neurokit2.ecg_process') as mock_process:
            mock_signals = {'ECG_Rate': np.array([75.0] * 100)}
            mock_info = {'ECG_R_Peaks': np.array([100, 200, 300])}
            mock_process.return_value = (mock_signals, mock_info)
            
            result = await processor.detect_arrhythmias(sample_ecg_data)
            assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_ecg_hybrid_processor_calculate_hrv_metrics(self):
        """Test HRV metrics calculation"""
        processor = ECGHybridProcessor()
        
        rr_intervals = np.array([0.8, 0.85, 0.82, 0.88, 0.79])
        result = await processor.calculate_hrv_metrics(rr_intervals)
        assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_ecg_hybrid_processor_extract_morphological_features(self, sample_ecg_data):
        """Test morphological feature extraction"""
        processor = ECGHybridProcessor()
        
        with patch('neurokit2.ecg_process') as mock_process:
            mock_signals = {'ECG_Clean': sample_ecg_data[:, 0]}
            mock_info = {'ECG_R_Peaks': np.array([100, 200, 300])}
            mock_process.return_value = (mock_signals, mock_info)
            
            result = await processor.extract_morphological_features(sample_ecg_data[:, 0])
            assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_validation_service_create_validation(self, mock_db):
        """Test validation creation"""
        with patch('app.repositories.validation_repository.ValidationRepository'):
            service = ValidationService(mock_db)
            service.repository = Mock()
            service.repository.create_validation = AsyncMock(return_value=Mock(id=1))
            
            result = await service.create_validation(1, 1, "Test validation")
            assert result.id == 1
    
    @pytest.mark.asyncio
    async def test_validation_service_get_validation_by_id(self, mock_db):
        """Test validation retrieval by ID"""
        with patch('app.repositories.validation_repository.ValidationRepository'):
            service = ValidationService(mock_db)
            service.repository = Mock()
            mock_validation = Mock(id=1)
            service.repository.get_validation_by_id = AsyncMock(return_value=mock_validation)
            
            result = await service.get_validation_by_id(1)
            assert result == mock_validation
    
    @pytest.mark.asyncio
    async def test_validation_service_update_validation_status(self, mock_db):
        """Test validation status update"""
        with patch('app.repositories.validation_repository.ValidationRepository'):
            service = ValidationService(mock_db)
            service.repository = Mock()
            service.repository.update_validation_status = AsyncMock(return_value=True)
            
            result = await service.update_validation_status(1, "approved")
            assert result is True
    
    @pytest.mark.asyncio
    async def test_validation_service_get_pending_validations(self, mock_db):
        """Test pending validations retrieval"""
        with patch('app.repositories.validation_repository.ValidationRepository'):
            service = ValidationService(mock_db)
            service.repository = Mock()
            mock_validations = [Mock(id=1), Mock(id=2)]
            service.repository.get_pending_validations = AsyncMock(return_value=mock_validations)
            
            result = await service.get_pending_validations()
            assert len(result) == 2
    
    @pytest.mark.asyncio
    async def test_ecg_service_create_analysis(self, mock_db):
        """Test ECG analysis creation"""
        with patch('app.repositories.ecg_repository.ECGRepository'):
            with patch('app.services.ml_model_service.MLModelService'):
                with patch('app.services.validation_service.ValidationService'):
                    service = ECGAnalysisService(mock_db, Mock(), Mock())
                    service.repository = Mock()
                    service.ml_model_service = Mock()
                    service.processor = Mock()
                    
                    mock_analysis = Mock(id=1)
                    service.repository.create_analysis = AsyncMock(return_value=mock_analysis)
                    service.processor.load_ecg_file = AsyncMock(return_value=np.random.randn(5000, 12))
                    service.ml_model_service.analyze_ecg = AsyncMock(return_value={"predictions": {}})
                    
                    result = await service.create_analysis(1, "/fake/test.csv", "test.csv")
                    assert result == mock_analysis
    
    @pytest.mark.asyncio
    async def test_ecg_service_process_ecg_file(self, mock_db, sample_ecg_data):
        """Test ECG file processing"""
        with patch('app.repositories.ecg_repository.ECGRepository'):
            with patch('app.services.ml_model_service.MLModelService'):
                with patch('app.services.validation_service.ValidationService'):
                    service = ECGAnalysisService(mock_db, Mock(), Mock())
                    service.processor = Mock()
                    service.ml_model_service = Mock()
                    
                    service.processor.load_ecg_file = AsyncMock(return_value=sample_ecg_data)
                    service.processor.preprocess_signal = AsyncMock(return_value=sample_ecg_data)
                    service.ml_model_service.analyze_ecg = AsyncMock(return_value={"predictions": {}})
                    
                    result = await service._process_ecg_file("/fake/test.csv")
                    assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_ecg_service_validate_analysis_results(self, mock_db):
        """Test analysis results validation"""
        with patch('app.repositories.ecg_repository.ECGRepository'):
            with patch('app.services.ml_model_service.MLModelService'):
                with patch('app.services.validation_service.ValidationService'):
                    service = ECGAnalysisService(mock_db, Mock(), Mock())
                    
                    results = {"predictions": {"normal": 0.8}, "confidence": 0.9}
                    is_valid = await service._validate_analysis_results(results)
                    assert isinstance(is_valid, bool)
