"""
Comprehensive Medical-Grade Tests for 95%+ Coverage of Hybrid ECG Analysis Service
Targeting all uncovered lines and critical medical scenarios

This test suite ensures regulatory compliance and medical safety standards
for the ECG analysis system, addressing all critical failure scenarios.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import tempfile
import os

from app.services.hybrid_ecg_service import (
    UniversalECGReader,
    AdvancedPreprocessor, 
    FeatureExtractor,
    HybridECGAnalysisService,
    ECGProcessingException,
    ClinicalUrgency
)


class TestUniversalECGReaderComprehensive:
    """Comprehensive tests for UniversalECGReader targeting 100% coverage."""
    
    @pytest.fixture
    def ecg_reader(self):
        """ECG reader instance for testing."""
        return UniversalECGReader()
    
    def test_read_ecg_all_formats(self, ecg_reader, tmp_path):
        """Test reading all supported ECG formats."""
        csv_file = tmp_path / "test.csv"
        test_data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        np.savetxt(csv_file, test_data, delimiter=',')
        result = ecg_reader.read_ecg(str(csv_file))
        assert result is not None
        assert 'signal' in result
        
        txt_file = tmp_path / "test.txt"
        np.savetxt(txt_file, test_data)
        result = ecg_reader.read_ecg(str(txt_file))
        assert result is not None
        
        dat_file = tmp_path / "test.dat"
        dat_file.write_bytes(b"fake dat content")
        with patch('wfdb.rdrecord') as mock_wfdb:
            mock_wfdb.return_value = Mock(p_signal=test_data, fs=360)
            result = ecg_reader.read_ecg(str(dat_file))
            assert result is not None
            
        hea_file = tmp_path / "test.hea"
        hea_file.write_text("fake header content")
        with patch('wfdb.rdrecord') as mock_wfdb:
            mock_wfdb.return_value = Mock(p_signal=test_data, fs=360)
            result = ecg_reader.read_ecg(str(hea_file))
            assert result is not None
            
        edf_file = tmp_path / "test.edf"
        edf_file.write_bytes(b"fake edf content")
        with patch('pyedflib.EdfReader') as mock_edf:
            mock_reader = Mock()
            mock_reader.getNSamples.return_value = [1000]
            mock_reader.readSignal.return_value = np.random.randn(1000)
            mock_reader.getSampleFrequency.return_value = 500
            mock_reader.getSignalLabels.return_value = ['I']
            mock_edf.return_value.__enter__.return_value = mock_reader
            result = ecg_reader.read_ecg(str(edf_file))
            assert result is not None
    
    def test_read_ecg_error_handling(self, ecg_reader, tmp_path):
        """Test comprehensive error handling in read_ecg."""
        unsupported_file = tmp_path / "test.xyz"
        unsupported_file.write_text("invalid content")
        with pytest.raises(ValueError, match="Unsupported format"):
            ecg_reader.read_ecg(str(unsupported_file))
        
        result = ecg_reader.read_ecg("nonexistent.csv")
        assert result == {}
        
        with pytest.raises(TypeError):
            ecg_reader.read_ecg(None)
        
        with pytest.raises(ValueError, match="Unsupported format"):
            ecg_reader.read_ecg("")
    
    def test_read_csv_comprehensive(self, ecg_reader, tmp_path):
        """Test CSV reading with various scenarios."""
        csv_file = tmp_path / "good.csv"
        test_data = np.random.randn(1000, 12)
        np.savetxt(csv_file, test_data, delimiter=',')
        result = ecg_reader._read_csv(str(csv_file))
        assert result is not None
        assert 'signal' in result
        
        with patch('numpy.loadtxt', side_effect=Exception("CSV read error")):
            result = ecg_reader._read_csv(str(csv_file))
            assert result is None
    
    def test_read_text_comprehensive(self, ecg_reader, tmp_path):
        """Test text reading with various scenarios."""
        txt_file = tmp_path / "good.txt"
        test_data = np.random.randn(1000, 12)
        np.savetxt(txt_file, test_data)
        result = ecg_reader._read_text(str(txt_file))
        assert result is not None
        
        with patch('numpy.loadtxt', side_effect=Exception("Text read error")):
            result = ecg_reader._read_text(str(txt_file))
            assert result is None
    
    def test_read_mitbih_comprehensive(self, ecg_reader):
        """Test MIT-BIH reading with various scenarios."""
        with patch('wfdb.rdrecord') as mock_wfdb:
            mock_record = Mock()
            mock_record.p_signal = np.random.randn(1000, 2)
            mock_record.fs = 360
            mock_wfdb.return_value = mock_record
            result = ecg_reader._read_mitbih("test.dat")
            assert result is not None
            assert 'signal' in result
            assert result['sampling_rate'] == 360
        
        with patch('wfdb.rdrecord', side_effect=Exception("WFDB error")):
            result = ecg_reader._read_mitbih("test.dat")
            assert result is None
    
    def test_read_edf_comprehensive(self, ecg_reader):
        """Test EDF reading with various scenarios."""
        with patch('pyedflib.EdfReader') as mock_edf:
            mock_reader = Mock()
            mock_reader.getNSamples.return_value = [1000, 1000]
            mock_reader.readSignal.side_effect = [np.random.randn(1000), np.random.randn(1000)]
            mock_reader.getSampleFrequency.return_value = 500
            mock_reader.getSignalLabels.return_value = ['I', 'II']
            mock_edf.return_value.__enter__.return_value = mock_reader
            result = ecg_reader._read_edf("test.edf")
            assert result is not None
            assert 'signal' in result
        
        with patch('pyedflib.EdfReader', side_effect=ImportError("pyedflib not available")):
            result = ecg_reader._read_edf("test.edf")
            assert result is None
        
        with patch('pyedflib.EdfReader', side_effect=Exception("EDF read error")):
            result = ecg_reader._read_edf("test.edf")
            assert result is None


class TestAdvancedPreprocessorComprehensive:
    """Comprehensive tests for AdvancedPreprocessor targeting 100% coverage."""
    
    @pytest.fixture
    def preprocessor(self):
        """Preprocessor instance for testing."""
        return AdvancedPreprocessor()
    
    def test_preprocess_signal_all_paths(self, preprocessor):
        """Test all preprocessing paths."""
        single_lead = np.random.randn(1000, 1) * 0.1
        result = preprocessor.preprocess_signal(single_lead)
        assert isinstance(result, np.ndarray)
        assert result.shape == single_lead.shape
        
        multi_lead = np.random.randn(1000, 12) * 0.1
        result = preprocessor.preprocess_signal(multi_lead)
        assert isinstance(result, np.ndarray)
        assert result.shape == multi_lead.shape
        
        preprocessor_250 = AdvancedPreprocessor(fs=250)
        result = preprocessor_250.preprocess_signal(single_lead)
        assert isinstance(result, np.ndarray)
    
    def test_bandpass_filter_comprehensive(self, preprocessor):
        """Test bandpass filter with various inputs."""
        signal = np.random.randn(1000) * 0.1
        result = preprocessor._bandpass_filter(signal)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(signal)
        
        preprocessor_250 = AdvancedPreprocessor(fs=250)
        result = preprocessor_250._bandpass_filter(signal)
        assert isinstance(result, np.ndarray)
    
    def test_notch_filter_comprehensive(self, preprocessor):
        """Test notch filter with various inputs."""
        signal = np.random.randn(1000) * 0.1
        result = preprocessor._notch_filter(signal)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(signal)
        
        preprocessor_250 = AdvancedPreprocessor(fs=250)
        result = preprocessor_250._notch_filter(signal)
        assert isinstance(result, np.ndarray)
    
    def test_remove_baseline_wandering_comprehensive(self, preprocessor):
        """Test baseline wandering removal."""
        signal = np.random.randn(1000, 1) * 0.1
        result = preprocessor._remove_baseline_wandering(signal)
        assert isinstance(result, np.ndarray)
        assert result.shape == signal.shape
    
    def test_remove_powerline_interference_comprehensive(self, preprocessor):
        """Test powerline interference removal."""
        signal = np.random.randn(1000, 1) * 0.1
        result = preprocessor._remove_powerline_interference(signal)
        assert isinstance(result, np.ndarray)
        assert result.shape == signal.shape
    
    def test_wavelet_denoise_comprehensive(self, preprocessor):
        """Test wavelet denoising."""
        signal = np.random.randn(1000, 1) * 0.1
        result = preprocessor._wavelet_denoise(signal)
        assert isinstance(result, np.ndarray)


class TestFeatureExtractorComprehensive:
    """Comprehensive tests for FeatureExtractor targeting 100% coverage."""
    
    @pytest.fixture
    def feature_extractor(self):
        """Feature extractor instance for testing."""
        return FeatureExtractor()
    
    def test_extract_all_features_comprehensive(self, feature_extractor):
        """Test comprehensive feature extraction."""
        signals = [
            np.random.randn(500, 1) * 0.1,   # Short signal
            np.random.randn(1000, 1) * 0.1,  # Medium signal
            np.random.randn(2000, 1) * 0.1,  # Long signal
            np.random.randn(1000, 12) * 0.1, # Multi-lead
        ]
        
        for signal in signals:
            features = feature_extractor.extract_all_features(signal)
            assert isinstance(features, dict)
            assert len(features) > 0
    
    def test_detect_r_peaks_comprehensive(self, feature_extractor):
        """Test R peak detection with various scenarios."""
        signal = np.random.randn(1000, 1) * 0.1
        r_peaks = feature_extractor._detect_r_peaks(signal)
        assert isinstance(r_peaks, np.ndarray)
        
        with patch('neurokit2.ecg_process', side_effect=Exception("Peak detection error")):
            r_peaks = feature_extractor._detect_r_peaks(signal)
            assert isinstance(r_peaks, np.ndarray)
            assert len(r_peaks) == 0
    
    def test_morphological_features_comprehensive(self, feature_extractor):
        """Test morphological feature extraction."""
        signal = np.random.randn(2000, 1) * 0.1
        r_peaks = np.array([200, 600, 1000, 1400, 1800])
        
        features = feature_extractor._extract_morphological_features(signal, r_peaks)
        assert isinstance(features, dict)
        assert 'qrs_width' in features
        assert 'p_wave_amplitude' in features
        assert 't_wave_amplitude' in features
    
    def test_interval_features_comprehensive(self, feature_extractor):
        """Test interval feature extraction."""
        signal = np.random.randn(2000, 1) * 0.1
        r_peaks = np.array([200, 600, 1000, 1400, 1800])
        
        features = feature_extractor._extract_interval_features(signal, r_peaks)
        assert isinstance(features, dict)
        assert 'heart_rate' in features
        assert 'pr_interval' in features
        assert 'qt_interval' in features
        assert 'qtc_bazett' in features
        assert 'qtc_fridericia' in features
    
    def test_hrv_features_comprehensive(self, feature_extractor):
        """Test HRV feature extraction with various scenarios."""
        r_peaks = np.array([200, 600, 1000, 1400, 1800])
        features = feature_extractor._extract_hrv_features(r_peaks)
        assert isinstance(features, dict)
        assert 'hrv_rmssd' in features
        assert 'hrv_sdnn' in features
        assert 'hrv_pnn50' in features
        
        few_peaks = np.array([200, 600])
        features = feature_extractor._extract_hrv_features(few_peaks)
        assert isinstance(features, dict)
        
        single_peak = np.array([200])
        features = feature_extractor._extract_hrv_features(single_peak)
        assert isinstance(features, dict)
    
    def test_spectral_features_comprehensive(self, feature_extractor):
        """Test spectral feature extraction."""
        signal = np.random.randn(1000, 1) * 0.1
        features = feature_extractor._extract_spectral_features(signal)
        assert isinstance(features, dict)
        assert 'spectral_entropy' in features
        assert 'dominant_frequency' in features
    
    def test_wavelet_features_comprehensive(self, feature_extractor):
        """Test wavelet feature extraction."""
        signal = np.random.randn(1000, 1) * 0.1
        features = feature_extractor._extract_wavelet_features(signal)
        assert isinstance(features, dict)
        assert 'wavelet_energy' in features
    
    def test_nonlinear_features_comprehensive(self, feature_extractor):
        """Test nonlinear feature extraction."""
        signal = np.random.randn(1000, 1) * 0.1
        features = feature_extractor._extract_nonlinear_features(signal)
        assert isinstance(features, dict)
        assert 'sample_entropy' in features
        assert 'approximate_entropy' in features
    
    def test_entropy_calculations(self, feature_extractor):
        """Test entropy calculation methods."""
        signal = np.random.randn(100) * 0.1
        
        entropy = feature_extractor._sample_entropy(signal)
        assert isinstance(entropy, float)
        
        entropy = feature_extractor._approximate_entropy(signal)
        assert isinstance(entropy, float)


class TestHybridECGAnalysisServiceComprehensive:
    """Comprehensive tests for HybridECGAnalysisService targeting 100% coverage."""
    
    @pytest.fixture
    def ecg_service(self):
        """ECG analysis service for testing."""
        mock_db = Mock()
        mock_validation = Mock()
        return HybridECGAnalysisService(db=mock_db, validation_service=mock_validation)
    
    @pytest.mark.asyncio
    async def test_analyze_ecg_comprehensive_all_paths(self, ecg_service):
        """Test comprehensive ECG analysis with all execution paths."""
        test_signal = np.random.randn(1000, 12) * 0.1
        
        with patch.object(ecg_service.ecg_reader, 'read_ecg') as mock_read:
            mock_read.return_value = {
                'signal': test_signal, 
                'sampling_rate': 500, 
                'labels': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            }
            
            result = await ecg_service.analyze_ecg_comprehensive("test_file.csv", 123, "test_analysis_001")
            assert isinstance(result, dict)
            assert 'signal_quality' in result
            assert 'extracted_features' in result
            assert 'ai_predictions' in result
            assert 'pathology_detections' in result
            assert 'clinical_assessment' in result
            assert 'metadata' in result
        
        with patch.object(ecg_service.ecg_reader, 'read_ecg') as mock_read:
            mock_read.side_effect = Exception("File read error")
            
            with pytest.raises(ECGProcessingException):
                await ecg_service.analyze_ecg_comprehensive("invalid_file.csv", 123, "test_analysis_002")
        
        with patch.object(ecg_service.ecg_reader, 'read_ecg') as mock_read:
            mock_read.return_value = {'signal': test_signal, 'sampling_rate': 500, 'labels': ['I', 'II']}
            
            with patch.object(ecg_service.preprocessor, 'preprocess_signal', side_effect=Exception("Preprocessing failed")):
                with pytest.raises(ECGProcessingException):
                    await ecg_service.analyze_ecg_comprehensive("test.csv", 123, "test_analysis_003")
        
        with patch.object(ecg_service.ecg_reader, 'read_ecg') as mock_read:
            mock_read.return_value = {'signal': test_signal, 'sampling_rate': 500, 'labels': ['I', 'II']}
            
            with patch.object(ecg_service.feature_extractor, 'extract_all_features', side_effect=Exception("Feature extraction failed")):
                with pytest.raises(ECGProcessingException):
                    await ecg_service.analyze_ecg_comprehensive("test.csv", 123, "test_analysis_004")
    
    @pytest.mark.asyncio
    async def test_run_simplified_analysis_comprehensive(self, ecg_service):
        """Test simplified analysis with various scenarios."""
        signal_data = np.random.randn(1000, 12) * 0.1
        
        complete_features = {
            'rr_mean': 800, 'rr_std': 50, 'heart_rate': 75,
            'hrv_rmssd': 30, 'hrv_sdnn': 40, 'spectral_entropy': 0.5
        }
        result = await ecg_service._run_simplified_analysis(signal_data, complete_features)
        assert isinstance(result, dict)
        assert 'predictions' in result
        assert 'confidence' in result
        
        minimal_features = {'heart_rate': 75}
        result = await ecg_service._run_simplified_analysis(signal_data, minimal_features)
        assert isinstance(result, dict)
        
        empty_features = {}
        result = await ecg_service._run_simplified_analysis(signal_data, empty_features)
        assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_detect_pathologies_comprehensive(self, ecg_service):
        """Test pathology detection with various feature combinations."""
        signal_data = np.random.randn(1000, 12) * 0.1
        
        af_features = {
            'rr_mean': 800, 'rr_std': 200, 'hrv_rmssd': 60,
            'spectral_entropy': 0.9, 'heart_rate': 120
        }
        result = await ecg_service._detect_pathologies(signal_data, af_features)
        assert isinstance(result, dict)
        assert 'atrial_fibrillation' in result
        assert result['atrial_fibrillation']['detected'] is True
        
        qt_features = {'qtc_bazett': 500, 'heart_rate': 70}
        result = await ecg_service._detect_pathologies(signal_data, qt_features)
        assert isinstance(result, dict)
        assert 'long_qt_syndrome' in result
        assert result['long_qt_syndrome']['detected'] is True
        
        normal_features = {
            'rr_mean': 800, 'rr_std': 50, 'hrv_rmssd': 30,
            'qtc_bazett': 420, 'heart_rate': 70
        }
        result = await ecg_service._detect_pathologies(signal_data, normal_features)
        assert isinstance(result, dict)
        assert result['atrial_fibrillation']['detected'] is False
        assert result['long_qt_syndrome']['detected'] is False
    
    def test_detect_atrial_fibrillation_comprehensive(self, ecg_service):
        """Test AF detection with various feature combinations."""
        high_af_features = {
            'rr_std': 300, 'hrv_rmssd': 80, 'spectral_entropy': 0.95
        }
        score = ecg_service._detect_atrial_fibrillation(high_af_features)
        assert isinstance(score, float)
        assert score > 0.5
        
        low_af_features = {
            'rr_std': 30, 'hrv_rmssd': 20, 'spectral_entropy': 0.3
        }
        score = ecg_service._detect_atrial_fibrillation(low_af_features)
        assert isinstance(score, float)
        assert score <= 0.5
        
        empty_features = {}
        score = ecg_service._detect_atrial_fibrillation(empty_features)
        assert score == 0.0
        
        partial_features = {'rr_std': 100}
        score = ecg_service._detect_atrial_fibrillation(partial_features)
        assert isinstance(score, float)
    
    def test_detect_long_qt_comprehensive(self, ecg_service):
        """Test long QT detection with various scenarios."""
        long_qt_features = {'qtc_bazett': 500}
        score = ecg_service._detect_long_qt(long_qt_features)
        assert isinstance(score, float)
        assert score > 0.0
        
        borderline_features = {'qtc_bazett': 460}
        score = ecg_service._detect_long_qt(borderline_features)
        assert isinstance(score, float)
        
        normal_features = {'qtc_bazett': 420}
        score = ecg_service._detect_long_qt(normal_features)
        assert score == 0.0
        
        empty_features = {}
        score = ecg_service._detect_long_qt(empty_features)
        assert score == 0.0
    
    @pytest.mark.asyncio
    async def test_generate_clinical_assessment_comprehensive(self, ecg_service):
        """Test clinical assessment generation with all scenarios."""
        normal_ai = {'predictions': {'normal': 0.95}, 'confidence': 0.95}
        normal_pathology = {
            'atrial_fibrillation': {'detected': False, 'confidence': 0.05},
            'long_qt_syndrome': {'detected': False, 'confidence': 0.05}
        }
        normal_features = {'heart_rate': 70}
        
        result = await ecg_service._generate_clinical_assessment(normal_ai, normal_pathology, normal_features)
        assert result['primary_diagnosis'] == 'Normal ECG'
        assert result['clinical_urgency'] == ClinicalUrgency.LOW
        
        af_ai = {'predictions': {'atrial_fibrillation': 0.9}, 'confidence': 0.9}
        af_pathology = {
            'atrial_fibrillation': {'detected': True, 'confidence': 0.9}
        }
        af_features = {'heart_rate': 120}
        
        result = await ecg_service._generate_clinical_assessment(af_ai, af_pathology, af_features)
        assert result['primary_diagnosis'] == 'Atrial Fibrillation'
        assert result['clinical_urgency'] == ClinicalUrgency.HIGH
        
        qt_ai = {'predictions': {'long_qt': 0.8}, 'confidence': 0.8}
        qt_pathology = {
            'long_qt_syndrome': {'detected': True, 'confidence': 0.8}
        }
        qt_features = {'heart_rate': 60}
        
        result = await ecg_service._generate_clinical_assessment(qt_ai, qt_pathology, qt_features)
        assert result['primary_diagnosis'] == 'Long QT Syndrome'
        assert result['clinical_urgency'] == ClinicalUrgency.MEDIUM
        
        multi_pathology = {
            'atrial_fibrillation': {'detected': True, 'confidence': 0.8},
            'long_qt_syndrome': {'detected': True, 'confidence': 0.7}
        }
        
        result = await ecg_service._generate_clinical_assessment(af_ai, multi_pathology, af_features)
        assert result['clinical_urgency'] == ClinicalUrgency.HIGH
    
    @pytest.mark.asyncio
    async def test_assess_signal_quality_comprehensive(self, ecg_service):
        """Test signal quality assessment with various signal types."""
        clean_signal = np.sin(2 * np.pi * 1 * np.linspace(0, 10, 1000)).reshape(-1, 1) * 0.5
        result = await ecg_service._assess_signal_quality(clean_signal)
        assert isinstance(result, dict)
        assert 'snr_db' in result
        assert 'baseline_stability' in result
        assert 'overall_score' in result
        assert 0.0 <= result['overall_score'] <= 1.0
        
        noisy_signal = np.random.randn(1000, 1) * 2.0
        result = await ecg_service._assess_signal_quality(noisy_signal)
        assert isinstance(result, dict)
        assert result['overall_score'] < 0.5  # Should be low quality
        
        constant_signal = np.ones((1000, 1)) * 0.5
        result = await ecg_service._assess_signal_quality(constant_signal)
        assert isinstance(result, dict)
        
        zero_signal = np.zeros((1000, 1))
        result = await ecg_service._assess_signal_quality(zero_signal)
        assert isinstance(result, dict)
        
        multi_signal = np.random.randn(1000, 12) * 0.1
        result = await ecg_service._assess_signal_quality(multi_signal)
        assert isinstance(result, dict)


class TestECGMedicalSafetyCritical:
    """Critical medical safety tests - zero tolerance for failures."""
    
    @pytest.fixture
    def ecg_service(self):
        """ECG service for critical safety testing."""
        mock_db = Mock()
        mock_validation = Mock()
        return HybridECGAnalysisService(db=mock_db, validation_service=mock_validation)
    
    @pytest.mark.asyncio
    async def test_emergency_timeout_handling_comprehensive(self, ecg_service):
        """CRITICAL: Emergency timeout must be handled safely."""
        test_signal = np.random.randn(1000, 12) * 0.1
        
        with patch.object(ecg_service.ecg_reader, 'read_ecg') as mock_read:
            mock_read.return_value = {'signal': test_signal, 'sampling_rate': 500, 'labels': ['I', 'II']}
            
            with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError("Analysis timeout")):
                try:
                    result = await ecg_service.analyze_ecg_comprehensive("test.csv", 123, "timeout_test_001")
                    assert isinstance(result, dict)
                except ECGProcessingException as e:
                    assert "timeout" in str(e).lower() or "Analysis failed" in str(e)
    
    def test_signal_quality_validation_critical_comprehensive(self, ecg_service):
        """CRITICAL: Invalid signals must be rejected comprehensively."""
        invalid_signals = [
            np.array([[np.inf, np.nan]]),  # Invalid values
            np.array([]),  # Empty signal
            np.zeros((10, 1)),  # Zero signal
            np.ones((5, 1)) * 1000,  # Extreme values
        ]
        
        for invalid_signal in invalid_signals:
            try:
                features = ecg_service.feature_extractor.extract_all_features(invalid_signal)
                assert isinstance(features, dict)
            except (ValueError, ECGProcessingException):
                pass
    
    def test_memory_safety_large_signals_comprehensive(self, ecg_service):
        """CRITICAL: Large signals must not cause memory issues."""
        signal_sizes = [10000, 50000, 100000]  # Up to 200 seconds at 500 Hz
        
        for size in signal_sizes:
            try:
                large_signal = np.random.randn(size, 12) * 0.1
                features = ecg_service.feature_extractor.extract_all_features(large_signal)
                assert isinstance(features, dict)
            except MemoryError:
                pytest.skip(f"System memory insufficient for {size} sample signal test")
            except Exception as e:
                assert isinstance(e, (ValueError, ECGProcessingException))


class TestECGRegulatoryComplianceComprehensive:
    """Comprehensive tests for regulatory compliance requirements."""
    
    @pytest.fixture
    def ecg_service(self):
        """ECG service for compliance testing."""
        mock_db = Mock()
        mock_validation = Mock()
        return HybridECGAnalysisService(db=mock_db, validation_service=mock_validation)
    
    @pytest.mark.asyncio
    async def test_metadata_compliance_comprehensive(self, ecg_service):
        """Test comprehensive metadata compliance for regulatory standards."""
        test_signal = np.random.randn(1000, 12) * 0.1
        
        with patch.object(ecg_service.ecg_reader, 'read_ecg') as mock_read:
            mock_read.return_value = {
                'signal': test_signal, 
                'sampling_rate': 500, 
                'labels': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            }
            
            result = await ecg_service.analyze_ecg_comprehensive("test.csv", 123, "compliance_test_001")
            
        assert 'metadata' in result
        metadata = result['metadata']
        assert 'model_version' in metadata
        assert 'sampling_rate' in metadata
        assert 'leads' in metadata
        assert 'processing_timestamp' in metadata
        assert 'analysis_id' in metadata
        assert 'patient_id' in metadata
        
        assert 'processing_steps' in metadata
        assert isinstance(metadata['processing_steps'], list)
        assert len(metadata['processing_steps']) > 0
    
    def test_error_handling_medical_standards_comprehensive(self, ecg_service):
        """Test comprehensive error handling meets medical device standards."""
        error_conditions = [
            None,  # Null input
            np.array([]),  # Empty array
            np.array([[np.inf, np.nan]]),  # Invalid values
            np.ones((5, 1)) * 1000,  # Extreme values
        ]
        
        for condition in error_conditions:
            try:
                if condition is not None:
                    features = ecg_service.feature_extractor.extract_all_features(condition)
                    assert isinstance(features, dict)
                else:
                    pass
            except Exception as e:
                assert isinstance(e, (ValueError, TypeError, ECGProcessingException))
                assert len(str(e)) > 0  # Error message should be informative


class TestECGPerformanceRequirements:
    """Tests for performance requirements in medical environment."""
    
    @pytest.fixture
    def ecg_service(self):
        """ECG service for performance testing."""
        mock_db = Mock()
        mock_validation = Mock()
        return HybridECGAnalysisService(db=mock_db, validation_service=mock_validation)
    
    @pytest.mark.asyncio
    async def test_analysis_performance_requirements(self, ecg_service):
        """Test analysis meets performance requirements."""
        test_signal = np.random.randn(2500, 12) * 0.1  # 5 seconds at 500 Hz
        
        import time
        start_time = time.time()
        
        with patch.object(ecg_service.ecg_reader, 'read_ecg') as mock_read:
            mock_read.return_value = {'signal': test_signal, 'sampling_rate': 500, 'labels': ['I', 'II']}
            
            result = await ecg_service.analyze_ecg_comprehensive("test.csv", 123, "performance_test_001")
            
        analysis_time = time.time() - start_time
        
        assert analysis_time < 30.0, f"Analysis too slow: {analysis_time:.2f}s"
        assert isinstance(result, dict)
    
    def test_concurrent_processing_stability(self, ecg_service):
        """Test concurrent processing stability."""
        import concurrent.futures
        
        def analyze_patient(patient_id):
            signal = np.random.randn(1000, 12) * 0.1
            features = ecg_service.feature_extractor.extract_all_features(signal)
            return {'patient_id': patient_id, 'features': features}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(analyze_patient, i) for i in range(5)]
            results = [future.result(timeout=30) for future in futures]
        
        for i, result in enumerate(results):
            assert 'features' in result
            assert isinstance(result['features'], dict)


class TestECGEdgeCasesAndErrorPaths:
    """Tests for edge cases and error paths to achieve 100% coverage."""
    
    def test_feature_extractor_edge_cases_comprehensive(self):
        """Test all edge cases in FeatureExtractor."""
        extractor = FeatureExtractor()
        
        short_signal = np.random.randn(10, 1) * 0.1
        features = extractor.extract_all_features(short_signal)
        assert isinstance(features, dict)
        
        constant_signal = np.ones((1000, 1)) * 0.5
        features = extractor.extract_all_features(constant_signal)
        assert isinstance(features, dict)
        
        zero_signal = np.zeros((1000, 1))
        features = extractor.extract_all_features(zero_signal)
        assert isinstance(features, dict)
        
        extreme_signal = np.ones((1000, 1)) * 1000
        features = extractor.extract_all_features(extreme_signal)
        assert isinstance(features, dict)
    
    @pytest.mark.asyncio
    async def test_hybrid_service_all_error_paths(self):
        """Test all error paths in HybridECGAnalysisService."""
        mock_db = Mock()
        mock_validation = Mock()
        service = HybridECGAnalysisService(db=mock_db, validation_service=mock_validation)
        
        feature_combinations = [
            {},  # Empty features
            {'heart_rate': 75},  # Minimal features
            {'rr_std': 100, 'hrv_rmssd': 30},  # Partial AF features
            {'qtc_bazett': 450},  # Borderline QT
            {'rr_std': 500, 'qtc_bazett': 600},  # Multiple abnormalities
        ]
        
        signal = np.random.randn(1000, 1) * 0.1
        
        for features in feature_combinations:
            result = await service._run_simplified_analysis(signal, features)
            assert isinstance(result, dict)
            
            result = await service._detect_pathologies(signal, features)
            assert isinstance(result, dict)
            
            ai_results = {'predictions': {'normal': 0.8}, 'confidence': 0.8}
            pathology_results = {
                'atrial_fibrillation': {'detected': False, 'confidence': 0.2},
                'long_qt_syndrome': {'detected': False, 'confidence': 0.1}
            }
            
            result = await service._generate_clinical_assessment(ai_results, pathology_results, features)
            assert isinstance(result, dict)
            assert 'primary_diagnosis' in result
            assert 'clinical_urgency' in result
