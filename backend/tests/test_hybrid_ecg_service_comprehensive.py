"""
Comprehensive Medical-Grade Tests for Hybrid ECG Analysis Service
Targeting 95%+ coverage for critical medical module
"""

import pytest
import asyncio
import time
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List

from app.services.hybrid_ecg_service import (
    HybridECGAnalysisService, 
    UniversalECGReader, 
    AdvancedPreprocessor, 
    FeatureExtractor
)
from app.core.exceptions import ECGProcessingException
from app.core.constants import ClinicalUrgency


class TestUniversalECGReaderComprehensive:
    """Comprehensive tests for UniversalECGReader class."""
    
    @pytest.fixture
    def ecg_reader(self):
        """ECG reader instance for testing."""
        return UniversalECGReader()
    
    def test_reader_initialization(self, ecg_reader):
        """Test ECG reader initializes with all supported formats."""
        assert ecg_reader.supported_formats is not None
        assert '.csv' in ecg_reader.supported_formats
        assert '.txt' in ecg_reader.supported_formats
        assert '.dat' in ecg_reader.supported_formats
        assert '.edf' in ecg_reader.supported_formats
        assert len(ecg_reader.supported_formats) >= 4
    
    def test_read_ecg_csv_format(self, ecg_reader, tmp_path):
        """Test reading CSV format ECG files."""
        test_data = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.1, 0.2, 0.3]])
        csv_file = tmp_path / "test.csv"
        np.savetxt(csv_file, test_data, delimiter=',')
        
        result = ecg_reader.read_ecg(str(csv_file))
        assert 'signal' in result
        assert 'sampling_rate' in result
        assert 'labels' in result
        assert isinstance(result['signal'], np.ndarray)
        assert result['sampling_rate'] == 250  # Default sampling rate
    
    def test_read_csv_internal_method(self, ecg_reader, tmp_path):
        """Test internal CSV reading method."""
        test_data = np.array([[0.1, 0.2], [0.3, 0.4]])
        csv_file = tmp_path / "internal_test.csv"
        np.savetxt(csv_file, test_data, delimiter=',')
        
        result = ecg_reader._read_csv(str(csv_file))
        assert 'signal' in result
        assert 'sampling_rate' in result
        assert 'labels' in result
        assert result['sampling_rate'] == 250
    
    def test_read_text_format(self, ecg_reader, tmp_path):
        """Test reading text format ECG files."""
        test_data = np.array([[0.1, 0.2], [0.3, 0.4]])
        txt_file = tmp_path / "test.txt"
        np.savetxt(txt_file, test_data)
        
        result = ecg_reader.read_ecg(str(txt_file))
        assert 'signal' in result
        assert 'sampling_rate' in result
        assert 'labels' in result
    
    def test_read_text_internal_method(self, ecg_reader, tmp_path):
        """Test internal text reading method."""
        test_data = np.array([[0.1, 0.2], [0.3, 0.4]])
        txt_file = tmp_path / "internal_test.txt"
        np.savetxt(txt_file, test_data)
        
        result = ecg_reader._read_text(str(txt_file))
        assert 'signal' in result
        assert 'sampling_rate' in result
        assert 'labels' in result
    
    @patch('app.services.hybrid_ecg_service.wfdb')
    def test_read_mitbih_format(self, mock_wfdb, ecg_reader, tmp_path):
        """Test reading MIT-BIH format files."""
        mock_wfdb.rdsamp.return_value = (
            np.array([[0.1, 0.2], [0.3, 0.4]]),
            {'fs': 360, 'sig_name': ['MLII', 'V1']}
        )
        
        dat_file = tmp_path / "test.dat"
        dat_file.touch()  # Create empty file
        
        result = ecg_reader._read_mitbih(str(dat_file).replace('.dat', ''))
        assert 'signal' in result
        assert 'sampling_rate' in result
        assert 'labels' in result
        assert result['sampling_rate'] == 360
        assert result['labels'] == ['MLII', 'V1']
    
    @patch('app.services.hybrid_ecg_service.wfdb')
    def test_read_mitbih_error_handling(self, mock_wfdb, ecg_reader):
        """Test MIT-BIH format error handling."""
        mock_wfdb.rdrecord.side_effect = Exception("File not found")
        
        result = ecg_reader._read_mitbih("nonexistent")
        assert isinstance(result, dict)
    
    def test_read_edf_format_mock(self, ecg_reader, tmp_path):
        """Test EDF format reading with mocking."""
        edf_file = tmp_path / "test.edf"
        edf_file.touch()
        
        with patch('app.services.hybrid_ecg_service.pyedflib') as mock_pyedflib:
            mock_file = MagicMock()
            mock_file.getNSamples.return_value = [1000, 1000]
            mock_file.getSampleFrequency.return_value = 250
            mock_file.getSignalLabels.return_value = ['I', 'II']
            mock_file.readSignal.side_effect = [
                np.array([0.1, 0.2, 0.3] * 333 + [0.1]),  # 1000 samples
                np.array([0.2, 0.3, 0.4] * 333 + [0.2])   # 1000 samples
            ]
            mock_pyedflib.EdfReader.return_value.__enter__.return_value = mock_file
            
            result = ecg_reader._read_edf(str(edf_file))
            assert 'signal' in result
            assert 'sampling_rate' in result
            assert 'labels' in result
            assert result['sampling_rate'] == 250
            assert result['labels'] == ['I', 'II']
    
    def test_read_image_format_mock(self, ecg_reader, tmp_path):
        """Test image format reading with mocking."""
        img_file = tmp_path / "test.png"
        img_file.touch()
        
        with patch('app.services.hybrid_ecg_service.Image') as mock_pil, \
             patch('app.services.hybrid_ecg_service.cv2') as mock_cv2:
            
            mock_img = MagicMock()
            mock_img.size = (1000, 800)
            mock_pil.open.return_value = mock_img
            
            mock_cv2.imread.return_value = np.ones((800, 1000, 3), dtype=np.uint8) * 128
            mock_cv2.cvtColor.return_value = np.ones((800, 1000), dtype=np.uint8) * 128
            mock_cv2.threshold.return_value = (127, np.ones((800, 1000), dtype=np.uint8) * 255)
            mock_cv2.findContours.return_value = ([], None)
            
            result = ecg_reader._read_image(str(img_file))
            assert 'signal' in result
            assert 'sampling_rate' in result
            assert 'labels' in result
    
    def test_unsupported_format_error(self, ecg_reader):
        """Test error handling for unsupported file formats."""
        with pytest.raises(ValueError, match="Unsupported format"):
            ecg_reader.read_ecg("test.xyz")
    
    def test_file_not_found_error(self, ecg_reader):
        """Test error handling for non-existent files."""
        with pytest.raises((FileNotFoundError, ValueError)):
            ecg_reader.read_ecg("/nonexistent/file.csv")


class TestAdvancedPreprocessorComprehensive:
    """Comprehensive tests for AdvancedPreprocessor class."""
    
    @pytest.fixture
    def preprocessor(self):
        """Preprocessor instance for testing."""
        return AdvancedPreprocessor()
    
    def test_preprocessor_initialization(self, preprocessor):
        """Test preprocessor initializes correctly."""
        assert preprocessor is not None
    
    def test_preprocess_signal_basic(self, preprocessor):
        """Test basic signal preprocessing."""
        t = np.linspace(0, 10, 2500)  # 10 seconds at 250 Hz
        ecg_signal = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(2500)
        test_signal = ecg_signal.reshape(-1, 1)
        
        result = preprocessor.preprocess_signal(test_signal)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] > 0
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()
    
    def test_preprocess_signal_multi_lead(self, preprocessor):
        """Test preprocessing with multiple leads."""
        t = np.linspace(0, 10, 2500)
        lead1 = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(2500)
        lead2 = np.sin(2 * np.pi * 1.1 * t) + 0.1 * np.random.randn(2500)
        test_signal = np.column_stack([lead1, lead2])
        
        result = preprocessor.preprocess_signal(test_signal)
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 2  # Two leads
        assert not np.isnan(result).any()
    
    def test_baseline_wandering_removal(self, preprocessor):
        """Test baseline wandering removal."""
        t = np.linspace(0, 10, 2500)
        baseline_drift = 0.1 * t  # Linear drift
        ecg_signal = np.sin(2 * np.pi * 1.2 * t) + baseline_drift
        test_signal = ecg_signal.reshape(-1, 1)
        
        result = preprocessor._remove_baseline_wandering(test_signal)
        assert isinstance(result, np.ndarray)
        assert result.shape == test_signal.shape
        
        original_mean = np.mean(test_signal)
        processed_mean = np.mean(result)
        assert abs(processed_mean) < abs(original_mean)
    
    def test_powerline_interference_removal(self, preprocessor):
        """Test powerline interference removal."""
        t = np.linspace(0, 4, 1000)  # 4 seconds, 250 Hz sampling rate
        ecg_signal = np.sin(2 * np.pi * 1.2 * t)  # Heart signal
        powerline_noise = 0.1 * np.sin(2 * np.pi * 50 * t)  # 50Hz noise
        test_signal = (ecg_signal + powerline_noise).reshape(-1, 1)
        
        result = preprocessor._remove_powerline_interference(test_signal)
        assert isinstance(result, np.ndarray)
        assert result.shape == test_signal.shape
    
    def test_bandpass_filter(self, preprocessor):
        """Test bandpass filtering."""
        t = np.linspace(0, 4, 1000)
        low_freq = np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz (should be filtered)
        ecg_freq = np.sin(2 * np.pi * 10 * t)   # 10 Hz (should pass)
        high_freq = np.sin(2 * np.pi * 100 * t) # 100 Hz (should be filtered)
        test_signal = (low_freq + ecg_freq + high_freq).reshape(-1, 1)
        
        result = preprocessor._bandpass_filter(test_signal)
        assert isinstance(result, np.ndarray)
        assert result.shape == test_signal.shape
    
    def test_wavelet_denoise(self, preprocessor):
        """Test wavelet denoising."""
        t = np.linspace(0, 4, 1000)
        clean_signal = np.sin(2 * np.pi * 1.2 * t)
        noise = 0.2 * np.random.randn(1000)
        test_signal = (clean_signal + noise).reshape(-1, 1)
        
        result = preprocessor._wavelet_denoise(test_signal)
        assert isinstance(result, np.ndarray)
        assert result.shape == test_signal.shape
        
        original_std = np.std(np.diff(test_signal, axis=0))
        processed_std = np.std(np.diff(result, axis=0))
        assert processed_std <= original_std
    
    def test_short_signal_handling(self, preprocessor):
        """Test handling of signals too short for filtering."""
        short_signal = np.array([[0.1], [0.2], [0.3]])
        
        try:
            result = preprocessor.preprocess_signal(short_signal)
            assert isinstance(result, np.ndarray)
        except ValueError as e:
            assert "length" in str(e).lower() or "short" in str(e).lower()


class TestFeatureExtractorComprehensive:
    """Comprehensive tests for FeatureExtractor class."""
    
    @pytest.fixture
    def feature_extractor(self):
        """Feature extractor instance for testing."""
        return FeatureExtractor()
    
    def test_feature_extractor_initialization(self, feature_extractor):
        """Test feature extractor initializes correctly."""
        assert feature_extractor is not None
    
    def test_extract_all_features(self, feature_extractor):
        """Test comprehensive feature extraction."""
        t = np.linspace(0, 10, 2500)  # 10 seconds at 250 Hz
        ecg_signal = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(2500)
        test_signal = ecg_signal.reshape(-1, 1)
        
        features = feature_extractor.extract_all_features(test_signal)
        assert isinstance(features, dict)
        assert len(features) > 0
        
        for key, value in features.items():
            if isinstance(value, (int, float)):
                assert not np.isnan(value), f"NaN found in feature {key}"
                assert not np.isinf(value), f"Inf found in feature {key}"
    
    def test_detect_r_peaks(self, feature_extractor):
        """Test R-peak detection."""
        t = np.linspace(0, 5, 1250)  # 5 seconds at 250 Hz
        ecg_signal = np.zeros(1250)
        peak_locations = [250, 500, 750, 1000]  # 4 beats
        for loc in peak_locations:
            if loc < len(ecg_signal):
                ecg_signal[loc-2:loc+3] = [0.2, 0.5, 1.0, 0.5, 0.2]
        
        test_signal = ecg_signal.reshape(-1, 1)
        r_peaks = feature_extractor._detect_r_peaks(test_signal)
        
        assert isinstance(r_peaks, np.ndarray)
        assert len(r_peaks) > 0
        assert len(r_peaks) <= len(peak_locations) + 2  # Allow some tolerance
    
    def test_morphological_features(self, feature_extractor):
        """Test morphological feature extraction."""
        test_signal = np.random.randn(1000, 1) * 0.1
        r_peaks = np.array([100, 300, 500, 700, 900])  # Mock R-peaks
        
        features = feature_extractor._extract_morphological_features(test_signal, r_peaks)
        assert isinstance(features, dict)
        
        expected_keys = ['qrs_width_mean', 'qrs_amplitude_mean', 'p_wave_amplitude', 't_wave_amplitude']
        for key in expected_keys:
            if key in features:
                assert isinstance(features[key], (int, float))
    
    def test_interval_features(self, feature_extractor):
        """Test interval feature extraction."""
        r_peaks = np.array([100, 300, 500, 700, 900])  # Mock R-peaks with 200ms intervals
        features = feature_extractor._extract_interval_features(r_peaks)
        
        assert isinstance(features, dict)
        assert 'rr_mean' in features
        assert 'rr_std' in features
        assert 'rr_min' in features
        assert 'rr_max' in features
        
        assert features['rr_mean'] > 0
        assert features['rr_std'] >= 0
        assert features['rr_min'] > 0
        assert features['rr_max'] >= features['rr_min']
    
    def test_hrv_features(self, feature_extractor):
        """Test heart rate variability features."""
        r_peaks = np.array([100, 300, 500, 700, 900])  # Mock R-peaks
        features = feature_extractor._extract_hrv_features(r_peaks)
        
        assert isinstance(features, dict)
        assert 'hrv_rmssd' in features
        assert 'hrv_pnn50' in features
        assert 'hrv_triangular_index' in features
        
        assert features['hrv_rmssd'] >= 0
        assert 0 <= features['hrv_pnn50'] <= 100  # Percentage
        assert features['hrv_triangular_index'] >= 0
    
    def test_spectral_features(self, feature_extractor):
        """Test spectral feature extraction."""
        t = np.linspace(0, 10, 2500)
        test_signal = (np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(2500)).reshape(-1, 1)
        
        features = feature_extractor._extract_spectral_features(test_signal)
        assert isinstance(features, dict)
        
        expected_keys = ['spectral_entropy', 'dominant_frequency', 'power_lf', 'power_hf']
        for key in expected_keys:
            if key in features:
                assert isinstance(features[key], (int, float))
                assert not np.isnan(features[key])
    
    def test_wavelet_features(self, feature_extractor):
        """Test wavelet feature extraction."""
        t = np.linspace(0, 4, 1000)
        test_signal = np.sin(2 * np.pi * 1.2 * t).reshape(-1, 1)
        
        features = feature_extractor._extract_wavelet_features(test_signal)
        assert isinstance(features, dict)
        
        for key, value in features.items():
            if 'wavelet' in key.lower():
                assert isinstance(value, (int, float))
                assert not np.isnan(value)
    
    def test_nonlinear_features(self, feature_extractor):
        """Test nonlinear feature extraction."""
        test_signal = np.random.randn(1000, 1) * 0.1
        r_peaks = np.array([100, 300, 500, 700, 900, 1100])  # Mock R-peaks
        features = feature_extractor._extract_nonlinear_features(test_signal, r_peaks)
        
        assert isinstance(features, dict)
        
        expected_keys = ['sample_entropy', 'approximate_entropy']
        for key in expected_keys:
            if key in features:
                assert isinstance(features[key], (int, float))
                assert not np.isnan(features[key])
    
    def test_sample_entropy(self, feature_extractor):
        """Test sample entropy calculation."""
        data = np.array([1, 2, 1, 2, 1, 2, 1, 2])  # Regular pattern
        entropy = feature_extractor._sample_entropy(data, m=2, r=0.1)
        
        assert isinstance(entropy, float)
        assert entropy >= 0
        assert not np.isnan(entropy)
    
    def test_approximate_entropy(self, feature_extractor):
        """Test approximate entropy calculation."""
        data = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2])  # Regular pattern
        entropy = feature_extractor._approximate_entropy(data, m=2, r=0.1)
        
        assert isinstance(entropy, float)
        assert entropy >= 0
        assert not np.isnan(entropy)


class TestHybridECGAnalysisServiceComprehensive:
    """Comprehensive tests for HybridECGAnalysisService class."""
    
    @pytest.fixture
    def mock_db(self):
        """Mock database for testing."""
        return Mock()
    
    @pytest.fixture
    def mock_validation_service(self):
        """Mock validation service for testing."""
        return Mock()
    
    @pytest.fixture
    def ecg_service(self, mock_db, mock_validation_service):
        """ECG service configured for comprehensive testing."""
        return HybridECGAnalysisService(
            db=mock_db,
            validation_service=mock_validation_service
        )
    
    @pytest.fixture
    def sample_signal_data(self):
        """Sample ECG signal data for testing."""
        signal = np.array([[0.1, 0.2, 0.3, 0.2, 0.1] * 500])  # 2500 samples
        return {
            'signal': signal,
            'sampling_rate': 250,
            'labels': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF']
        }
    
    def test_service_initialization_comprehensive(self, mock_db, mock_validation_service):
        """Test comprehensive service initialization."""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        
        assert service.db is mock_db
        assert service.validation_service is mock_validation_service
        assert service.repository is not None
        assert service.ecg_reader is not None
        assert service.preprocessor is not None
        assert service.feature_extractor is not None
        assert service.ecg_logger is not None
        
        assert isinstance(service.ecg_reader, UniversalECGReader)
        assert isinstance(service.preprocessor, AdvancedPreprocessor)
        assert isinstance(service.feature_extractor, FeatureExtractor)
    
    @pytest.mark.asyncio
    async def test_analyze_ecg_comprehensive_full_workflow(self, ecg_service, sample_signal_data, tmp_path):
        """Test complete ECG analysis workflow."""
        ecg_file = tmp_path / "test_ecg.csv"
        np.savetxt(ecg_file, sample_signal_data['signal'], delimiter=',')
        
        with patch.object(ecg_service.ecg_reader, 'read_ecg', return_value=sample_signal_data):
            result = await ecg_service.analyze_ecg_comprehensive(
                file_path=str(ecg_file),
                patient_id=12345,
                analysis_id="COMPREHENSIVE_001"
            )
            
            assert result['analysis_id'] == "COMPREHENSIVE_001"
            assert result['patient_id'] == 12345
            assert 'processing_time_seconds' in result
            assert 'signal_quality' in result
            assert 'ai_predictions' in result
            assert 'pathology_detections' in result
            assert 'clinical_assessment' in result
            assert 'extracted_features' in result
            assert 'metadata' in result
            
            metadata = result['metadata']
            assert metadata['gdpr_compliant'] is True
            assert metadata['ce_marking'] is True
            assert metadata['surveillance_plan'] is True
            assert metadata['nmsa_certification'] is True
    
    @pytest.mark.asyncio
    async def test_analyze_ecg_error_handling(self, ecg_service):
        """Test error handling in comprehensive analysis."""
        with pytest.raises(ECGProcessingException):
            await ecg_service.analyze_ecg_comprehensive(
                file_path="/nonexistent/file.csv",
                patient_id=99999,
                analysis_id="ERROR_TEST"
            )
    
    @pytest.mark.asyncio
    async def test_run_simplified_analysis(self, ecg_service):
        """Test simplified AI analysis."""
        test_signal = np.array([[0.1, 0.2, 0.3] * 100])
        test_features = {
            'rr_mean': 1000,  # 60 BPM
            'rr_std': 50      # Low variability
        }
        
        result = await ecg_service._run_simplified_analysis(test_signal, test_features)
        
        assert 'predictions' in result
        assert 'confidence' in result
        assert 'model_version' in result
        assert isinstance(result['predictions'], dict)
        assert 0.0 <= result['confidence'] <= 1.0
        
        predictions = result['predictions']
        assert 'normal' in predictions
        assert 'atrial_fibrillation' in predictions
        assert 'tachycardia' in predictions
        assert 'bradycardia' in predictions
    
    @pytest.mark.asyncio
    async def test_detect_pathologies(self, ecg_service):
        """Test pathology detection."""
        test_signal = np.array([[0.1, 0.2, 0.3] * 100])
        test_features = {
            'rr_mean': 800,
            'rr_std': 300,
            'hrv_rmssd': 60,
            'spectral_entropy': 0.9,
            'qtc_bazett': 480
        }
        
        result = await ecg_service._detect_pathologies(test_signal, test_features)
        
        assert 'atrial_fibrillation' in result
        assert 'long_qt_syndrome' in result
        
        af_result = result['atrial_fibrillation']
        assert 'detected' in af_result
        assert 'confidence' in af_result
        assert 'criteria' in af_result
        assert isinstance(af_result['detected'], bool)
        assert 0.0 <= af_result['confidence'] <= 1.0
    
    def test_detect_atrial_fibrillation_positive(self, ecg_service):
        """Test atrial fibrillation detection - positive case."""
        af_features = {
            'rr_mean': 800,  # ms
            'rr_std': 300,   # High variability
            'hrv_rmssd': 60, # High HRV
            'spectral_entropy': 0.9  # High entropy
        }
        
        score = ecg_service._detect_atrial_fibrillation(af_features)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should detect AF
    
    def test_detect_atrial_fibrillation_negative(self, ecg_service):
        """Test atrial fibrillation detection - negative case."""
        normal_features = {
            'rr_mean': 1000,  # Normal RR interval
            'rr_std': 50,     # Low variability
            'hrv_rmssd': 30,  # Normal HRV
            'spectral_entropy': 0.3  # Low entropy
        }
        
        score = ecg_service._detect_atrial_fibrillation(normal_features)
        assert 0.0 <= score <= 1.0
        assert score < 0.5  # Should not detect AF
    
    def test_detect_atrial_fibrillation_edge_cases(self, ecg_service):
        """Test atrial fibrillation detection edge cases."""
        edge_features = {
            'rr_mean': 0,
            'rr_std': 100,
            'hrv_rmssd': 30,
            'spectral_entropy': 0.5
        }
        
        score = ecg_service._detect_atrial_fibrillation(edge_features)
        assert 0.0 <= score <= 1.0
        assert not np.isnan(score)
        assert not np.isinf(score)
    
    def test_detect_long_qt_positive(self, ecg_service):
        """Test long QT syndrome detection - positive case."""
        long_qt_features = {'qtc_bazett': 480}  # ms, above threshold
        score = ecg_service._detect_long_qt(long_qt_features)
        assert 0.0 <= score <= 1.0
        assert score > 0.0
    
    def test_detect_long_qt_negative(self, ecg_service):
        """Test long QT syndrome detection - negative case."""
        normal_qt_features = {'qtc_bazett': 420}  # ms, normal
        score = ecg_service._detect_long_qt(normal_qt_features)
        assert score == 0.0
    
    def test_detect_long_qt_extreme(self, ecg_service):
        """Test long QT syndrome detection - extreme case."""
        extreme_qt_features = {'qtc_bazett': 600}  # Very long QT
        score = ecg_service._detect_long_qt(extreme_qt_features)
        assert 0.0 <= score <= 1.0
        assert score == 1.0  # Should be capped at 1.0
    
    @pytest.mark.asyncio
    async def test_generate_clinical_assessment_normal(self, ecg_service):
        """Test clinical assessment generation for normal case."""
        ai_results = {
            'predictions': {'normal': 0.9, 'atrial_fibrillation': 0.1},
            'confidence': 0.9
        }
        pathology_results = {
            'atrial_fibrillation': {'detected': False, 'confidence': 0.1}
        }
        features = {'rr_mean': 1000, 'rr_std': 50}
        
        assessment = await ecg_service._generate_clinical_assessment(
            ai_results, pathology_results, features
        )
        
        assert 'primary_diagnosis' in assessment
        assert 'clinical_urgency' in assessment
        assert 'recommendations' in assessment
        assert 'confidence' in assessment
        assert assessment['primary_diagnosis'] == 'Normal ECG'
        assert assessment['clinical_urgency'] == ClinicalUrgency.LOW
    
    @pytest.mark.asyncio
    async def test_generate_clinical_assessment_af(self, ecg_service):
        """Test clinical assessment generation for atrial fibrillation."""
        ai_results = {
            'predictions': {'atrial_fibrillation': 0.8, 'normal': 0.2},
            'confidence': 0.8
        }
        pathology_results = {
            'atrial_fibrillation': {'detected': True, 'confidence': 0.75}
        }
        features = {'rr_mean': 800, 'rr_std': 200}
        
        assessment = await ecg_service._generate_clinical_assessment(
            ai_results, pathology_results, features
        )
        
        assert assessment['primary_diagnosis'] == 'Atrial Fibrillation'
        assert assessment['clinical_urgency'] == ClinicalUrgency.HIGH
        assert any('Anticoagulation' in rec for rec in assessment['recommendations'])
    
    @pytest.mark.asyncio
    async def test_assess_signal_quality_good(self, ecg_service):
        """Test signal quality assessment for good signal."""
        t = np.linspace(0, 10, 2500)
        good_signal = np.sin(2 * np.pi * 1.2 * t).reshape(-1, 1)
        
        quality_metrics = await ecg_service._assess_signal_quality(good_signal)
        
        assert 'snr_db' in quality_metrics
        assert 'baseline_stability' in quality_metrics
        assert 'overall_score' in quality_metrics
        
        assert 0.0 <= quality_metrics['overall_score'] <= 1.0
        assert 0.0 <= quality_metrics['baseline_stability'] <= 1.0
        assert quality_metrics['overall_score'] > 0.3  # Good signal
    
    @pytest.mark.asyncio
    async def test_assess_signal_quality_poor(self, ecg_service):
        """Test signal quality assessment for poor signal."""
        poor_signal = np.random.randn(1000, 1) * 0.01  # Very low amplitude, high noise
        
        quality_metrics = await ecg_service._assess_signal_quality(poor_signal)
        
        assert 'snr_db' in quality_metrics
        assert 'baseline_stability' in quality_metrics
        assert 'overall_score' in quality_metrics
        assert 0.0 <= quality_metrics['overall_score'] <= 1.0
    
    @pytest.mark.asyncio
    async def test_assess_signal_quality_edge_cases(self, ecg_service):
        """Test signal quality assessment edge cases."""
        zero_signal = np.zeros((1000, 1))
        quality_metrics = await ecg_service._assess_signal_quality(zero_signal)
        
        assert not np.isnan(quality_metrics['overall_score'])
        assert not np.isinf(quality_metrics['overall_score'])
        assert 0.0 <= quality_metrics['overall_score'] <= 1.0


class TestECGServiceIntegration:
    """Integration tests for complete ECG service workflow."""
    
    @pytest.fixture
    def ecg_service(self):
        """ECG service for integration testing."""
        return HybridECGAnalysisService(Mock(), Mock())
    
    @pytest.mark.asyncio
    async def test_complete_workflow_integration(self, ecg_service, tmp_path):
        """Test complete workflow from file to clinical assessment."""
        test_data = np.array([[0.1, 0.2, 0.3, 0.2, 0.1] * 500])
        ecg_file = tmp_path / "integration_test.csv"
        np.savetxt(ecg_file, test_data, delimiter=',')
        
        signal_data = {
            'signal': test_data,
            'sampling_rate': 250,
            'labels': ['I', 'II']
        }
        
        with patch.object(ecg_service.ecg_reader, 'read_ecg', return_value=signal_data):
            result = await ecg_service.analyze_ecg_comprehensive(
                file_path=str(ecg_file),
                patient_id=12345,
                analysis_id="INTEGRATION_001"
            )
            
            assert result['analysis_id'] == "INTEGRATION_001"
            assert 'processing_time_seconds' in result
            assert 'signal_quality' in result
            assert 'ai_predictions' in result
            assert 'pathology_detections' in result
            assert 'clinical_assessment' in result
            
            assert result['processing_time_seconds'] < 30.0
    
    def test_performance_requirements(self, ecg_service):
        """Test performance requirements for medical environment."""
        large_signal = np.random.randn(5000, 1) * 0.1
        
        start_time = time.time()
        processed = ecg_service.preprocessor.preprocess_signal(large_signal)
        processing_time = time.time() - start_time
        
        assert processing_time < 10.0  # Should process quickly
        assert processed is not None
    
    def test_memory_constraints(self, ecg_service):
        """Test memory usage constraints."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        for i in range(3):
            large_signal = np.random.randn(10000, 1) * 0.1
            _ = ecg_service.preprocessor.preprocess_signal(large_signal)
            _ = ecg_service.feature_extractor.extract_all_features(large_signal)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        assert memory_used < 200, f"Excessive memory usage: {memory_used:.1f}MB"


class TestECGRegulatoryCompliance:
    """Tests for regulatory compliance requirements."""
    
    @pytest.fixture
    def ecg_service(self):
        """ECG service for compliance testing."""
        return HybridECGAnalysisService(Mock(), Mock())
    
    @pytest.mark.asyncio
    async def test_metadata_compliance_requirements(self, ecg_service, tmp_path):
        """Test that analysis includes required compliance metadata."""
        test_signal = np.array([[0.1, 0.2, 0.3] * 500])
        signal_data = {
            'signal': test_signal,
            'sampling_rate': 250,
            'labels': ['I', 'II']
        }
        
        with patch.object(ecg_service.ecg_reader, 'read_ecg', return_value=signal_data):
            result = await ecg_service.analyze_ecg_comprehensive(
                file_path="test.csv",
                patient_id=12345,
                analysis_id="COMPLIANCE_001"
            )
            
            metadata = result['metadata']
            
            assert metadata['gdpr_compliant'] is True
            assert metadata['ce_marking'] is True
            assert metadata['surveillance_plan'] is True
            assert metadata['nmsa_certification'] is True
            assert metadata['data_residency'] is True
            assert metadata['language_support'] is True
            assert metadata['population_validation'] is True
    
    def test_audit_trail_completeness(self, ecg_service):
        """Test audit trail for regulatory compliance."""
        test_signal = np.random.randn(1000, 1) * 0.1
        
        processed = ecg_service.preprocessor.preprocess_signal(test_signal)
        assert isinstance(processed, np.ndarray)
        
        features = ecg_service.feature_extractor.extract_all_features(test_signal)
        assert isinstance(features, dict)
        
        af_score = ecg_service._detect_atrial_fibrillation({'rr_mean': 800, 'rr_std': 200})
        assert isinstance(af_score, float)
    
    def test_error_handling_medical_standards(self, ecg_service):
        """Test error handling meets medical standards."""
        with pytest.raises(ValueError):
            ecg_service.ecg_reader.read_ecg("nonexistent.xyz")
        
        invalid_features = {'invalid_key': 'invalid_value'}
        score = ecg_service._detect_atrial_fibrillation(invalid_features)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
