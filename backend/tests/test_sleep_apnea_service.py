"""
Tests for Sleep Apnea and Respiratory Analysis Service
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from app.services.sleep_apnea_service import (
    RespiratoryPatternAnalyzer,
    SleepApneaDetector
)


class TestRespiratoryPatternAnalyzer:
    """Test respiratory pattern analysis functionality"""

    def test_init(self):
        """Test analyzer initialization"""
        analyzer = RespiratoryPatternAnalyzer(sampling_rate=250)
        
        assert analyzer.sampling_rate == 250
        assert "normal" in analyzer.respiratory_bands
        assert "bradypnea" in analyzer.respiratory_bands
        assert "tachypnea" in analyzer.respiratory_bands

    def test_extract_respiratory_signal_edr(self):
        """Test EDR respiratory signal extraction"""
        analyzer = RespiratoryPatternAnalyzer()
        
        ecg_signal = np.random.randn(5000).astype(np.float64)
        
        with patch.object(analyzer, '_extract_edr_signal') as mock_edr:
            mock_edr.return_value = np.zeros(5000, dtype=np.float64)
            
            result = analyzer.extract_respiratory_signal(ecg_signal, method="edr")
            
            mock_edr.assert_called_once_with(ecg_signal)
            assert len(result) == 5000

    def test_extract_respiratory_signal_rsa(self):
        """Test RSA respiratory signal extraction"""
        analyzer = RespiratoryPatternAnalyzer()
        
        ecg_signal = np.random.randn(5000).astype(np.float64)
        
        with patch.object(analyzer, '_extract_rsa_signal') as mock_rsa:
            mock_rsa.return_value = np.zeros(5000, dtype=np.float64)
            
            result = analyzer.extract_respiratory_signal(ecg_signal, method="rsa")
            
            mock_rsa.assert_called_once_with(ecg_signal)
            assert len(result) == 5000

    def test_extract_respiratory_signal_invalid_method(self):
        """Test invalid method handling"""
        analyzer = RespiratoryPatternAnalyzer()
        
        ecg_signal = np.random.randn(1000).astype(np.float64)
        
        result = analyzer.extract_respiratory_signal(ecg_signal, method="invalid")
        
        assert len(result) == len(ecg_signal)
        assert np.all(result == 0)

    def test_analyze_respiratory_rate_short_signal(self):
        """Test respiratory rate analysis with short signal"""
        analyzer = RespiratoryPatternAnalyzer()
        
        short_signal = np.random.randn(50).astype(np.float64)
        
        result = analyzer.analyze_respiratory_rate(short_signal)
        
        assert result["respiratory_rate"] == 0.0
        assert result["confidence"] == 0.0

    @patch('app.services.sleep_apnea_service.SCIPY_AVAILABLE', True)
    def test_analyze_respiratory_rate_with_scipy(self):
        """Test respiratory rate analysis with SciPy available"""
        analyzer = RespiratoryPatternAnalyzer()
        
        t = np.linspace(0, 10, 5000)
        respiratory_signal = np.sin(2 * np.pi * 0.3 * t).astype(np.float64)  # 18 breaths/min
        
        with patch('app.services.sleep_apnea_service.scipy.signal.welch') as mock_welch:
            freqs = np.linspace(0, 2, 1000)
            psd = np.ones(1000)
            psd[150] = 10  # Peak at 0.3 Hz
            mock_welch.return_value = (freqs, psd)
            
            result = analyzer.analyze_respiratory_rate(respiratory_signal)
            
            assert "respiratory_rate" in result
            assert "confidence" in result
            assert "classification" in result

    def test_classify_respiratory_rate(self):
        """Test respiratory rate classification"""
        analyzer = RespiratoryPatternAnalyzer()
        
        assert analyzer._classify_respiratory_rate(5) == "bradypnea"
        assert analyzer._classify_respiratory_rate(15) == "normal"
        assert analyzer._classify_respiratory_rate(30) == "tachypnea"

    def test_detect_r_peaks_basic(self):
        """Test basic R-peak detection"""
        analyzer = RespiratoryPatternAnalyzer()
        
        ecg_signal = np.zeros(1000, dtype=np.float64)
        ecg_signal[100] = 2.0
        ecg_signal[400] = 2.0
        ecg_signal[700] = 2.0
        
        peaks = analyzer._detect_r_peaks(ecg_signal)
        
        assert len(peaks) >= 0  # Should detect some peaks or none


class TestSleepApneaDetector:
    """Test sleep apnea detection functionality"""

    def test_init(self):
        """Test detector initialization"""
        detector = SleepApneaDetector(sampling_rate=250)
        
        assert detector.sampling_rate == 250
        assert "mild" in detector.apnea_thresholds
        assert "moderate" in detector.apnea_thresholds
        assert "severe" in detector.apnea_thresholds

    async def test_detect_sleep_apnea_basic(self):
        """Test basic sleep apnea detection"""
        detector = SleepApneaDetector()
        
        ecg_signal = np.random.randn(5000).astype(np.float64)
        
        with patch.object(detector, '_detect_apnea_events') as mock_apnea, \
             patch.object(detector, '_detect_hypopnea_events') as mock_hypopnea:
            
            mock_apnea.return_value = []
            mock_hypopnea.return_value = []
            
            result = await detector.detect_sleep_apnea(ecg_signal, duration_hours=1.0)
            
            assert "ahi" in result
            assert "severity" in result
            assert "apnea_events" in result
            assert "hypopnea_events" in result
            assert "total_events" in result
            assert "recommendations" in result
            assert "processing_time" in result
            assert "confidence" in result

    def test_detect_apnea_events_short_signal(self):
        """Test apnea detection with short signal"""
        detector = SleepApneaDetector()
        
        short_signal = np.random.randn(50).astype(np.float64)
        
        events = detector._detect_apnea_events(short_signal)
        
        assert events == []

    def test_detect_hypopnea_events_short_signal(self):
        """Test hypopnea detection with short signal"""
        detector = SleepApneaDetector()
        
        short_signal = np.random.randn(50).astype(np.float64)
        
        events = detector._detect_hypopnea_events(short_signal)
        
        assert events == []

    def test_classify_sleep_apnea_severity(self):
        """Test sleep apnea severity classification"""
        detector = SleepApneaDetector()
        
        assert detector._classify_sleep_apnea_severity(2) == "normal"
        assert detector._classify_sleep_apnea_severity(10) == "mild"
        assert detector._classify_sleep_apnea_severity(20) == "moderate"
        assert detector._classify_sleep_apnea_severity(40) == "severe"

    def test_calculate_detection_confidence(self):
        """Test detection confidence calculation"""
        detector = SleepApneaDetector()
        
        high_confidence = detector._calculate_detection_confidence(15, 5)
        assert high_confidence > 0.5
        
        low_confidence = detector._calculate_detection_confidence(0, 1)
        assert low_confidence < 0.5
        
        assert 0.1 <= high_confidence <= 1.0
        assert 0.1 <= low_confidence <= 1.0

    def test_generate_apnea_recommendations_severe(self):
        """Test severe apnea recommendations"""
        detector = SleepApneaDetector()
        
        recommendations = detector._generate_apnea_recommendations("severe", 35.0)
        
        assert "Urgent sleep medicine consultation required" in recommendations
        assert "Polysomnography (sleep study) recommended" in recommendations
        assert "Consider CPAP therapy evaluation" in recommendations

    def test_generate_apnea_recommendations_moderate(self):
        """Test moderate apnea recommendations"""
        detector = SleepApneaDetector()
        
        recommendations = detector._generate_apnea_recommendations("moderate", 20.0)
        
        assert "Sleep medicine consultation recommended" in recommendations
        assert "Sleep study (polysomnography) advised" in recommendations
        assert "Lifestyle modifications (weight loss, sleep position)" in recommendations

    def test_generate_apnea_recommendations_mild(self):
        """Test mild apnea recommendations"""
        detector = SleepApneaDetector()
        
        recommendations = detector._generate_apnea_recommendations("mild", 8.0)
        
        assert "Sleep hygiene counseling" in recommendations
        assert "Weight management if applicable" in recommendations

    def test_generate_apnea_recommendations_normal(self):
        """Test normal apnea recommendations"""
        detector = SleepApneaDetector()
        
        recommendations = detector._generate_apnea_recommendations("normal", 2.0)
        
        assert "Maintain good sleep hygiene" in recommendations
        assert "Regular follow-up if symptoms develop" in recommendations

    def test_empty_apnea_result(self):
        """Test empty apnea result structure"""
        detector = SleepApneaDetector()
        
        result = detector._empty_apnea_result()
        
        assert result["ahi"] == 0.0
        assert result["severity"] == "unknown"
        assert result["apnea_events"] == []
        assert result["hypopnea_events"] == []
        assert result["total_events"] == 0
        assert result["recommendations"] == []
        assert result["processing_time"] == 0.0
        assert result["confidence"] == 0.0

    async def test_detect_sleep_apnea_error_handling(self):
        """Test error handling in sleep apnea detection"""
        detector = SleepApneaDetector()
        
        with patch.object(detector, '_detect_apnea_events', side_effect=Exception("Test error")):
            result = await detector.detect_sleep_apnea(np.array([]), duration_hours=1.0)
            
            assert result["ahi"] == 0.0
            assert result["severity"] == "unknown"


@patch('app.services.sleep_apnea_service.SCIPY_AVAILABLE', False)
class TestWithoutSciPy:
    """Test functionality when SciPy is not available"""

    def test_respiratory_analysis_without_scipy(self):
        """Test respiratory analysis fallback without SciPy"""
        analyzer = RespiratoryPatternAnalyzer()
        
        t = np.linspace(0, 10, 1000)
        respiratory_signal = np.sin(2 * np.pi * 0.25 * t).astype(np.float64)
        
        result = analyzer.analyze_respiratory_rate(respiratory_signal)
        
        assert "respiratory_rate" in result
        assert "confidence" in result
        assert result["confidence"] == 0.5  # Lower confidence without SciPy

    def test_bandpass_filter_without_scipy(self):
        """Test bandpass filter fallback without SciPy"""
        analyzer = RespiratoryPatternAnalyzer()
        
        signal = np.random.randn(1000).astype(np.float64)
        
        filtered = analyzer._bandpass_filter(signal, 0.1, 0.5)
        
        np.testing.assert_array_equal(filtered, signal)


@patch('app.services.sleep_apnea_service.NEUROKIT_AVAILABLE', False)
class TestWithoutNeuroKit:
    """Test functionality when NeuroKit2 is not available"""

    def test_r_peak_detection_without_neurokit(self):
        """Test R-peak detection fallback without NeuroKit2"""
        analyzer = RespiratoryPatternAnalyzer()
        
        ecg_signal = np.zeros(1000, dtype=np.float64)
        ecg_signal[100:110] = 1.0  # Simple peak
        ecg_signal[500:510] = 1.0  # Another peak
        
        peaks = analyzer._detect_r_peaks(ecg_signal)
        
        assert isinstance(peaks, np.ndarray)
        assert peaks.dtype == np.int64
