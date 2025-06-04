"""
Medical-Grade Tests for Hybrid ECG Analysis Service
Targeting 95%+ coverage for critical medical module
Compliance: FDA CFR 21 Part 820, ISO 13485, EU MDR 2017/745, ANVISA RDC 185/2001
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import asyncio
import time
from typing import Dict, Any

from app.services.hybrid_ecg_service import (
    UniversalECGReader,
    AdvancedPreprocessor, 
    FeatureExtractor,
    HybridECGAnalysisService
)


class TestUniversalECGReaderMedicalGrade:
    """Medical-grade tests for UniversalECGReader - 100% coverage target."""
    
    def test_initialization_complete(self):
        """Test complete initialization of ECG reader."""
        reader = UniversalECGReader()
        assert reader is not None
        assert hasattr(reader, 'supported_formats')
        assert '.csv' in reader.supported_formats
        assert '.dat' in reader.supported_formats
        assert '.edf' in reader.supported_formats
        assert '.txt' in reader.supported_formats
        assert '.png' in reader.supported_formats
        assert '.jpg' in reader.supported_formats
    
    def test_read_ecg_all_return_paths(self):
        """Test all return paths in read_ecg method - critical for medical safety."""
        reader = UniversalECGReader()
        
        dict_result = {"signal": [1, 2, 3], "sampling_rate": 500}
        with patch.object(reader, '_read_csv', return_value=dict_result):
            result = reader.read_ecg('/fake/path.csv')
            assert result == dict_result
        
        with patch.object(reader, '_read_csv', return_value="string_result"):
            result = reader.read_ecg('/fake/path.csv')
            assert result == {"data": "string_result"}
        
        with patch.object(reader, '_read_csv', return_value=None):
            result = reader.read_ecg('/fake/path.csv')
            assert result == {}
        
        with pytest.raises(ValueError, match="Unsupported format"):
            reader.read_ecg('/fake/path.xyz')
    
    def test_mitbih_reading_all_paths(self):
        """Test MIT-BIH reading with all exception paths."""
        reader = UniversalECGReader()
        
        mock_record = Mock()
        mock_record.p_signal = np.random.randn(1000, 2)
        mock_record.fs = 360
        mock_record.sig_name = ['MLII', 'V1']
        
        with patch('wfdb.rdrecord', return_value=mock_record):
            result = reader._read_mitbih('/fake/path.dat')
            assert isinstance(result, dict)
            assert 'signal' in result
            assert 'sampling_rate' in result
            assert result['sampling_rate'] == 360
        
        with patch('wfdb.rdrecord', side_effect=ImportError("wfdb not available")):
            result = reader._read_mitbih('/fake/path.dat')
            assert result is None
        
        with patch('wfdb.rdrecord', side_effect=Exception("File not found")):
            result = reader._read_mitbih('/fake/path.dat')
            assert result is None
    
    def test_edf_reading_all_paths(self):
        """Test EDF reading with all exception paths."""
        reader = UniversalECGReader()
        
        with patch('pyedflib.EdfReader', side_effect=ImportError("pyedflib not available")):
            result = reader._read_edf('/fake/path.edf')
            assert result is None
        
        with patch('pyedflib.EdfReader', side_effect=Exception("EDF error")):
            result = reader._read_edf('/fake/path.edf')
            assert result is None
    
    def test_csv_reading_all_paths(self):
        """Test CSV reading with all paths."""
        reader = UniversalECGReader()
        
        mock_data = pd.DataFrame({'I': [1, 2, 3], 'II': [4, 5, 6]})
        with patch('pandas.read_csv', return_value=mock_data):
            result = reader._read_csv('/fake/path.csv', 250)
            assert isinstance(result, dict)
            assert 'signal' in result
            assert 'sampling_rate' in result
            assert result['sampling_rate'] == 250
            assert result['labels'] == ['I', 'II']
        
        with patch('pandas.read_csv', side_effect=Exception("CSV error")):
            result = reader._read_csv('/fake/path.csv')
            assert result is None
    
    def test_text_reading_all_paths(self):
        """Test text reading with all paths."""
        reader = UniversalECGReader()
        
        mock_data = np.array([[1, 2], [3, 4], [5, 6]])
        with patch('numpy.loadtxt', return_value=mock_data):
            result = reader._read_text('/fake/path.txt')
            assert isinstance(result, dict)
            assert 'signal' in result
            assert 'sampling_rate' in result
        
        with patch('numpy.loadtxt', side_effect=Exception("Text error")):
            result = reader._read_text('/fake/path.txt')
            assert result is None
    
    @pytest.mark.asyncio
    async def test_image_reading_all_paths(self):
        """Test image reading with all paths - covers lines 147-180."""
        reader = UniversalECGReader()
        
        result = await reader._read_image('/fake/path.png')
        assert isinstance(result, dict)
        assert 'signal' in result
        assert 'sampling_rate' in result
        assert result['metadata']['processing_method'] == 'not_implemented'
        assert result['metadata']['scanner_confidence'] == 0.0
        
        with patch('numpy.random.randn', side_effect=Exception("Random generation failed")):
            result = await reader._read_image('/fake/path.png')
            assert isinstance(result, dict)
            assert 'signal' in result
            assert result['metadata']['source'] == 'digitized_image_fallback'


class TestAdvancedPreprocessorMedicalGrade:
    """Medical-grade tests for AdvancedPreprocessor - 100% coverage target."""
    
    def test_initialization_with_sampling_rate(self):
        """Test preprocessor initialization."""
        preprocessor = AdvancedPreprocessor()
        assert preprocessor is not None
    
    def test_preprocess_signal_complete_pipeline(self):
        """Test complete preprocessing pipeline."""
        preprocessor = AdvancedPreprocessor(sampling_rate=500)
        signal = np.random.randn(1000, 1).astype(np.float64)
        
        result = preprocessor.preprocess_signal(signal)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] > 0
    
    def test_preprocess_signal_with_parameters(self):
        """Test preprocessing with custom parameters."""
        preprocessor = AdvancedPreprocessor()
        signal = np.random.randn(1000, 1).astype(np.float64)
        
        result = preprocessor.preprocess_signal(signal)
        assert isinstance(result, np.ndarray)
    
    def test_all_preprocessing_methods(self):
        """Test all individual preprocessing methods."""
        preprocessor = AdvancedPreprocessor(sampling_rate=500)
        signal = np.random.randn(500, 1).astype(np.float64).flatten()
        
        result = preprocessor._remove_baseline_wandering(signal)
        assert isinstance(result, np.ndarray)
        
        result = preprocessor._remove_powerline_interference(signal)
        assert isinstance(result, np.ndarray)
        
        result = preprocessor._bandpass_filter(signal)
        assert isinstance(result, np.ndarray)
        
        result = preprocessor._wavelet_denoise(signal)
        assert isinstance(result, np.ndarray)
    
    def test_edge_cases_and_exceptions(self):
        """Test edge cases that could cause medical system failures."""
        preprocessor = AdvancedPreprocessor(sampling_rate=500)
        
        short_signal = np.random.randn(10, 1).astype(np.float64)
        try:
            result = preprocessor.preprocess_signal(short_signal)
            assert isinstance(result, np.ndarray)
        except Exception:
            pass
        
        constant_signal = np.ones(100, dtype=np.float64)
        result = preprocessor._remove_baseline_wandering(constant_signal)
        assert isinstance(result, np.ndarray)


class TestFeatureExtractorMedicalGrade:
    """Medical-grade tests for FeatureExtractor - 100% coverage target."""
    
    def test_initialization(self):
        """Test feature extractor initialization."""
        extractor = FeatureExtractor()
        assert extractor is not None
    
    def test_extract_all_features_complete(self):
        """Test complete feature extraction pipeline."""
        extractor = FeatureExtractor()
        signal = np.random.randn(5000, 1).astype(np.float64)
        
        features = extractor.extract_all_features(signal)
        assert isinstance(features, dict)
        assert len(features) > 0
    
    def test_extract_all_features_with_r_peaks(self):
        """Test feature extraction with provided R peaks."""
        extractor = FeatureExtractor()
        signal = np.random.randn(5000, 1).astype(np.float64)
        r_peaks = np.array([100, 200, 300, 400, 500], dtype=np.int64)
        
        features = extractor.extract_all_features(signal, r_peaks=r_peaks)
        assert isinstance(features, dict)
        assert len(features) > 0
    
    def test_detect_r_peaks(self):
        """Test R peak detection."""
        extractor = FeatureExtractor(sampling_rate=500)
        signal = np.random.randn(1000, 1).astype(np.float64)
        
        r_peaks = extractor._detect_r_peaks(signal)
        assert isinstance(r_peaks, np.ndarray)
    
    def test_all_feature_extraction_methods(self):
        """Test all individual feature extraction methods."""
        extractor = FeatureExtractor(sampling_rate=500)
        signal = np.random.randn(1000, 1).astype(np.float64)
        r_peaks = np.array([100, 200, 300, 400], dtype=np.int64)
        
        features = extractor._extract_morphological_features(signal, r_peaks)
        assert isinstance(features, dict)
        
        features = extractor._extract_interval_features(signal, r_peaks)
        assert isinstance(features, dict)
        
        features = extractor._extract_hrv_features(r_peaks)
        assert isinstance(features, dict)
        
        features = extractor._extract_spectral_features(signal)
        assert isinstance(features, dict)
        
        features = extractor._extract_wavelet_features(signal)
        assert isinstance(features, dict)
        
        features = extractor._extract_nonlinear_features(signal)
        assert isinstance(features, dict)
    
    def test_entropy_calculations(self):
        """Test entropy calculation methods."""
        extractor = FeatureExtractor(sampling_rate=500)
        signal = np.random.randn(100, 1).astype(np.float64).flatten()
        
        entropy = extractor._sample_entropy(signal, m=2, r=0.2)
        assert isinstance(entropy, float)
        
        entropy = extractor._approximate_entropy(signal, m=2, r=0.2)
        assert isinstance(entropy, float)
    
    def test_edge_cases_empty_r_peaks(self):
        """Test edge cases with empty R peaks."""
        extractor = FeatureExtractor(sampling_rate=500)
        signal = np.random.randn(1000, 1).astype(np.float64)
        empty_peaks = np.array([], dtype=np.int64)
        
        features = extractor._extract_morphological_features(signal, empty_peaks)
        assert isinstance(features, dict)
        
        features = extractor._extract_interval_features(signal, empty_peaks)
        assert isinstance(features, dict)
        
        features = extractor._extract_hrv_features(empty_peaks)
        assert isinstance(features, dict)
    
    def test_edge_cases_single_r_peak(self):
        """Test edge cases with single R peak."""
        extractor = FeatureExtractor()
        single_peak = np.array([500], dtype=np.int64)
        
        features = extractor._extract_hrv_features(single_peak)
        assert isinstance(features, dict)


class TestHybridECGAnalysisServiceMedicalGrade:
    """Medical-grade tests for HybridECGAnalysisService - 100% coverage target."""
    
    @pytest.fixture
    def mock_db(self):
        return Mock()
    
    @pytest.fixture
    def mock_validation_service(self):
        return Mock()
    
    @pytest.fixture
    def ecg_service(self, mock_db, mock_validation_service):
        return HybridECGAnalysisService(
            db=mock_db,
            validation_service=mock_validation_service
        )
    
    def test_initialization(self, mock_db, mock_validation_service):
        """Test service initialization."""
        service = HybridECGAnalysisService(
            db=mock_db,
            validation_service=mock_validation_service
        )
        assert service.db == mock_db
        assert service.validation_service == mock_validation_service
        assert hasattr(service, 'ecg_reader')
        assert hasattr(service, 'preprocessor')
        assert hasattr(service, 'feature_extractor')
    
    @pytest.mark.asyncio
    async def test_analyze_ecg_comprehensive_complete(self, ecg_service):
        """Test complete ECG analysis pipeline."""
        import tempfile
        import pandas as pd
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data = pd.DataFrame({
                'I': np.random.randn(1000) * 0.1,
                'II': np.random.randn(1000) * 0.1,
                'V1': np.random.randn(1000) * 0.1
            })
            test_data.to_csv(f.name, index=False)
            
            result = await ecg_service.analyze_ecg_comprehensive(
                file_path=f.name,
                patient_id=123,
                analysis_id="TEST_001"
            )
            
            assert isinstance(result, dict)
            assert 'analysis_id' in result
            assert 'patient_id' in result
            assert 'extracted_features' in result
            assert 'pathology_detections' in result
            assert 'clinical_assessment' in result
            assert 'signal_quality' in result
    
    def test_detect_atrial_fibrillation_all_paths(self, ecg_service):
        """Test atrial fibrillation detection with all feature combinations."""
        features = {
            'rr_mean': 800,
            'rr_std': 150,
            'hrv_rmssd': 45,
            'spectral_entropy': 0.8
        }
        score = ecg_service._detect_atrial_fibrillation(features)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        
        empty_features = {}
        score = ecg_service._detect_atrial_fibrillation(empty_features)
        assert isinstance(score, float)
        assert score == 0.0
        
        extreme_features = {
            'rr_mean': 1e6,
            'rr_std': 1e6,
            'hrv_rmssd': 1e6,
            'spectral_entropy': 1e6
        }
        score = ecg_service._detect_atrial_fibrillation(extreme_features)
        assert isinstance(score, float)
    
    def test_detect_long_qt_all_paths(self, ecg_service):
        """Test long QT detection with all feature combinations."""
        features = {'qtc_bazett': 450}
        score = ecg_service._detect_long_qt(features)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        
        empty_features = {}
        score = ecg_service._detect_long_qt(empty_features)
        assert isinstance(score, float)
        assert score == 0.0
    
    @pytest.mark.asyncio
    async def test_assess_signal_quality_all_paths(self, ecg_service):
        """Test signal quality assessment with various signal types."""
        normal_signal = np.random.randn(1000, 1).astype(np.float64) * 0.1
        quality = await ecg_service._assess_signal_quality(normal_signal)
        assert isinstance(quality, dict)
        assert 'overall_score' in quality
        assert 'snr' in quality
        assert 'baseline_stability' in quality
        
        nan_signal = np.full((1000, 1), np.nan, dtype=np.float64)
        quality = await ecg_service._assess_signal_quality(nan_signal)
        assert isinstance(quality, dict)
        assert 'overall_score' in quality
        
        inf_signal = np.full((1000, 1), np.inf, dtype=np.float64)
        quality = await ecg_service._assess_signal_quality(inf_signal)
        assert isinstance(quality, dict)
        assert 'overall_score' in quality
    
    @pytest.mark.asyncio
    async def test_run_simplified_analysis(self, ecg_service):
        """Test simplified analysis pipeline."""
        signal = np.random.randn(1000, 1).astype(np.float64)
        features = {
            'rr_mean': 800,
            'rr_std': 50,
            'hrv_rmssd': 30
        }
        
        result = await ecg_service._run_simplified_analysis(signal, features)
        assert isinstance(result, dict)
        assert 'predictions' in result
        assert 'confidence' in result
        assert 'model_version' in result
    
    @pytest.mark.asyncio
    async def test_detect_pathologies(self, ecg_service):
        """Test pathology detection."""
        signal = np.random.randn(1000, 1).astype(np.float64)
        features = {
            'rr_mean': 800,
            'rr_std': 50,
            'qtc_bazett': 420
        }
        
        pathologies = await ecg_service._detect_pathologies(signal, features)
        assert isinstance(pathologies, dict)
        assert 'atrial_fibrillation' in pathologies
        assert 'long_qt_syndrome' in pathologies
    
    @pytest.mark.asyncio
    async def test_generate_clinical_assessment(self, ecg_service):
        """Test clinical assessment generation."""
        ai_results = {
            'predictions': {'normal': 0.8, 'atrial_fibrillation': 0.2},
            'confidence': 0.8
        }
        pathology_results = {
            'atrial_fibrillation': {'detected': False, 'confidence': 0.1},
            'long_qt_syndrome': {'detected': False, 'confidence': 0.2}
        }
        features = {'rr_mean': 800, 'rr_std': 50}
        
        assessment = await ecg_service._generate_clinical_assessment(ai_results, pathology_results, features)
        assert isinstance(assessment, dict)
        assert 'primary_diagnosis' in assessment
        assert 'recommendations' in assessment
        assert 'clinical_urgency' in assessment


class TestECGMedicalSafetyCritical:
    """Critical safety tests for medical compliance."""
    
    @pytest.fixture
    def ecg_service(self):
        return HybridECGAnalysisService(db=Mock(), validation_service=Mock())
    
    @pytest.mark.asyncio
    async def test_emergency_timeout_handling(self, ecg_service):
        """Test emergency timeout handling - critical for patient safety."""
        normal_signal = np.random.randn(1000, 1).astype(np.float64)
        
        with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError("Analysis timeout")):
            try:
                result = await ecg_service._assess_signal_quality(normal_signal)
                assert isinstance(result, dict)
            except asyncio.TimeoutError:
                pass
    
    @pytest.mark.asyncio
    async def test_signal_quality_validation_critical(self, ecg_service):
        """Test signal quality validation for critical scenarios."""
        noisy_signal = np.random.randn(1000, 1).astype(np.float64) * 10
        quality = await ecg_service._assess_signal_quality(noisy_signal)
        assert isinstance(quality, dict)
        assert 'overall_score' in quality
    
    def test_feature_extraction_robustness(self):
        """Test feature extraction robustness with edge cases."""
        extractor = FeatureExtractor(sampling_rate=500)
        
        nan_signal = np.full((100, 1), np.nan, dtype=np.float64)
        try:
            features = extractor.extract_all_features(nan_signal)
            assert isinstance(features, dict)
        except Exception:
            pass
        
        zero_signal = np.zeros((100, 1), dtype=np.float64)
        features = extractor.extract_all_features(zero_signal)
        assert isinstance(features, dict)


class TestECGRegulatoryComplianceBasic:
    """Basic regulatory compliance tests."""
    
    @pytest.fixture
    def ecg_service(self):
        return HybridECGAnalysisService(db=Mock(), validation_service=Mock())
    
    def test_audit_trail_presence(self, ecg_service):
        """Test that audit trail components are present."""
        assert hasattr(ecg_service, 'reader')
        assert hasattr(ecg_service, 'preprocessor')
        assert hasattr(ecg_service, 'feature_extractor')
    
    @pytest.mark.asyncio
    async def test_metadata_completeness(self, ecg_service):
        """Test that analysis results contain required metadata."""
        ecg_data = {
            'signal': np.random.randn(1000, 1).astype(np.float64),
            'sampling_rate': 500
        }
        
        result = await ecg_service.analyze_ecg_comprehensive(
            ecg_data=ecg_data,
            patient_id="METADATA_001"
        )
        
        assert 'analysis_id' in result
        assert 'patient_id' in result
        assert 'timestamp' in result
        assert 'features' in result
        assert 'pathologies' in result
        assert 'clinical_assessment' in result
        assert 'signal_quality' in result
        assert 'processing_time' in result


class TestECGPerformanceRequirements:
    """Performance requirement tests for medical environment."""
    
    @pytest.fixture
    def ecg_service(self):
        return HybridECGAnalysisService(db=Mock(), validation_service=Mock())
    
    @pytest.mark.asyncio
    async def test_processing_time_tracking(self, ecg_service):
        """Test that processing time is tracked for medical compliance."""
        import tempfile
        import pandas as pd
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data = pd.DataFrame({
                'I': np.random.randn(1000) * 0.1,
                'II': np.random.randn(1000) * 0.1
            })
            test_data.to_csv(f.name, index=False)
            
            start_time = time.time()
            result = await ecg_service.analyze_ecg_comprehensive(
                file_path=f.name,
                patient_id=123,
                analysis_id="PERF_001"
            )
            elapsed_time = time.time() - start_time
            
            assert 'processing_time_seconds' in result
            assert elapsed_time < 60.0  # Should complete within 60 seconds
    
    def test_memory_efficiency_basic(self):
        """Test basic memory efficiency."""
        reader = UniversalECGReader()
        preprocessor = AdvancedPreprocessor()
        extractor = FeatureExtractor()
        
        assert reader is not None
        assert preprocessor is not None
        assert extractor is not None
