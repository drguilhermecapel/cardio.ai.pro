"""
Comprehensive tests for Hybrid ECG Analysis Service
Tests regulatory compliance for FDA, ANVISA, NMSA, and EU MDR standards
"""

import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from pathlib import Path

from app.services.hybrid_ecg_service import (
    HybridECGAnalysisService,
    UniversalECGReader,
    AdvancedPreprocessor,
    FeatureExtractor
)
from app.services.regulatory_validation import (
    RegulatoryValidationService,
    RegulatoryStandard,
    ValidationResult
)
from app.core.exceptions import ECGProcessingException


class TestUniversalECGReader:
    """Test ECG file reading capabilities"""
    
    def setup_method(self):
        self.reader = UniversalECGReader()
    
    def test_csv_reading(self, tmp_path):
        """Test CSV file reading"""
        csv_file = tmp_path / "test_ecg.csv"
        test_data = "I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6\n"
        for i in range(100):
            test_data += ",".join([str(np.random.randn()) for _ in range(12)]) + "\n"
        
        csv_file.write_text(test_data)
        
        result = self.reader.read_ecg(str(csv_file))
        
        assert 'signal' in result
        assert 'sampling_rate' in result
        assert 'labels' in result
        assert result['signal'].shape[1] == 12
        assert len(result['labels']) == 12
    
    def test_text_reading(self, tmp_path):
        """Test text file reading"""
        txt_file = tmp_path / "test_ecg.txt"
        test_data = np.random.randn(100, 12)
        np.savetxt(txt_file, test_data)
        
        result = self.reader.read_ecg(str(txt_file))
        
        assert result['signal'].shape == (100, 12)
        assert result['sampling_rate'] == 500
    
    def test_unsupported_format(self):
        """Test handling of unsupported file formats"""
        with pytest.raises(ValueError, match="Unsupported format"):
            self.reader.read_ecg("test.xyz")


class TestAdvancedPreprocessor:
    """Test ECG signal preprocessing"""
    
    def setup_method(self):
        self.preprocessor = AdvancedPreprocessor(sampling_rate=500)
    
    def test_signal_preprocessing(self):
        """Test complete preprocessing pipeline"""
        t = np.linspace(0, 10, 5000)
        clean_signal = np.sin(2 * np.pi * 1.2 * t)  # 1.2 Hz heart rate
        noise = np.random.randn(5000) * 0.1
        noisy_signal = clean_signal + noise
        
        powerline = 0.05 * np.sin(2 * np.pi * 50 * t)
        signal_with_interference = noisy_signal + powerline
        
        signal_data = np.column_stack([signal_with_interference] * 12)
        
        processed = self.preprocessor.preprocess_signal(signal_data)
        
        assert processed.shape == signal_data.shape
        assert np.std(processed) < np.std(signal_data)  # Should reduce noise
    
    def test_baseline_wandering_removal(self):
        """Test baseline wandering removal"""
        t = np.linspace(0, 10, 5000)
        baseline = 0.5 * np.sin(2 * np.pi * 0.1 * t)  # Low frequency baseline
        signal = np.sin(2 * np.pi * 1.2 * t) + baseline
        
        corrected = self.preprocessor._remove_baseline_wandering(signal)
        
        assert np.mean(np.abs(corrected)) < np.mean(np.abs(signal))
    
    def test_powerline_interference_removal(self):
        """Test powerline interference removal"""
        t = np.linspace(0, 10, 5000)
        clean_signal = np.sin(2 * np.pi * 1.2 * t)
        powerline_50hz = 0.1 * np.sin(2 * np.pi * 50 * t)
        powerline_60hz = 0.1 * np.sin(2 * np.pi * 60 * t)
        
        contaminated = clean_signal + powerline_50hz + powerline_60hz
        filtered = self.preprocessor._remove_powerline_interference(contaminated)
        
        freqs = np.fft.fftfreq(len(filtered), 1/500)
        fft_contaminated = np.abs(np.fft.fft(contaminated))
        fft_filtered = np.abs(np.fft.fft(filtered))
        
        idx_50 = np.argmin(np.abs(freqs - 50))
        idx_60 = np.argmin(np.abs(freqs - 60))
        
        assert fft_filtered[idx_50] < fft_contaminated[idx_50]
        assert fft_filtered[idx_60] < fft_contaminated[idx_60]


class TestFeatureExtractor:
    """Test ECG feature extraction"""
    
    def setup_method(self):
        self.extractor = FeatureExtractor(sampling_rate=500)
    
    def test_feature_extraction_complete(self):
        """Test complete feature extraction"""
        t = np.linspace(0, 10, 5000)
        ecg_signal = self._generate_synthetic_ecg(t)
        signal_data = np.column_stack([ecg_signal] * 12)
        
        features = self.extractor.extract_all_features(signal_data)
        
        expected_features = [
            'r_peak_amplitude_mean', 'r_peak_amplitude_std',
            'signal_amplitude_range', 'signal_mean', 'signal_std',
            'rr_mean', 'rr_std', 'rr_min', 'rr_max',
            'hrv_rmssd', 'hrv_sdnn', 'hrv_pnn50',
            'dominant_frequency', 'spectral_entropy', 'power_total',
            'sample_entropy', 'approximate_entropy'
        ]
        
        for feature in expected_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))
    
    def test_r_peak_detection(self):
        """Test R peak detection"""
        t = np.linspace(0, 10, 5000)
        ecg_signal = self._generate_synthetic_ecg(t)
        
        r_peaks = self.extractor._detect_r_peaks(ecg_signal)
        
        assert 8 <= len(r_peaks) <= 16
    
    def test_hrv_features(self):
        """Test HRV feature extraction"""
        r_peaks = np.array([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500])
        
        features = self.extractor._extract_hrv_features(r_peaks)
        
        assert 'hrv_rmssd' in features
        assert 'hrv_sdnn' in features
        assert 'hrv_pnn50' in features
        assert features['hrv_sdnn'] == 0.0  # Regular rhythm should have 0 SDNN
    
    def test_spectral_features(self):
        """Test spectral feature extraction"""
        t = np.linspace(0, 10, 5000)
        signal = np.sin(2 * np.pi * 1.2 * t)  # 1.2 Hz dominant frequency
        signal_data = signal.reshape(-1, 1)
        
        features = self.extractor._extract_spectral_features(signal_data)
        
        assert 'dominant_frequency' in features
        assert 'spectral_entropy' in features
        assert 'power_total' in features
        assert abs(features['dominant_frequency'] - 1.2) < 0.1
    
    def _generate_synthetic_ecg(self, t):
        """Generate synthetic ECG signal"""
        heart_rate = 72  # bpm
        beat_interval = 60 / heart_rate
        
        ecg = np.zeros_like(t)
        for beat_time in np.arange(0, t[-1], beat_interval):
            qrs_start = beat_time - 0.05
            qrs_end = beat_time + 0.05
            mask = (t >= qrs_start) & (t <= qrs_end)
            ecg[mask] += np.exp(-((t[mask] - beat_time) / 0.02)**2)
        
        return ecg


class TestHybridECGAnalysisService:
    """Test Hybrid ECG Analysis Service"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock()
    
    @pytest.fixture
    def mock_validation_service(self):
        return Mock()
    
    @pytest.fixture
    def service(self, mock_db, mock_validation_service):
        return HybridECGAnalysisService(mock_db, mock_validation_service)
    
    @pytest.mark.asyncio
    async def test_comprehensive_analysis(self, service, tmp_path):
        """Test comprehensive ECG analysis"""
        csv_file = tmp_path / "test_ecg.csv"
        test_data = "I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6\n"
        for i in range(5000):  # 10 seconds at 500 Hz
            test_data += ",".join([str(np.random.randn() * 0.1) for _ in range(12)]) + "\n"
        
        csv_file.write_text(test_data)
        
        service.repository = Mock()
        
        result = await service.analyze_ecg_comprehensive(
            file_path=str(csv_file),
            patient_id=123,
            analysis_id="TEST_001"
        )
        
        assert 'analysis_id' in result
        assert 'patient_id' in result
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
    async def test_atrial_fibrillation_detection(self, service):
        """Test atrial fibrillation detection"""
        features = {
            'rr_mean': 800,  # ms
            'rr_std': 300,   # High variability
            'hrv_rmssd': 80, # High RMSSD
            'spectral_entropy': 0.9  # High entropy
        }
        
        af_score = service._detect_atrial_fibrillation(features)
        
        assert af_score > 0.5  # Should detect AF
    
    @pytest.mark.asyncio
    async def test_long_qt_detection(self, service):
        """Test long QT syndrome detection"""
        features = {
            'qtc_bazett': 480  # ms (prolonged)
        }
        
        qt_score = service._detect_long_qt(features)
        
        assert qt_score > 0.0  # Should detect prolonged QT
    
    @pytest.mark.asyncio
    async def test_signal_quality_assessment(self, service):
        """Test signal quality assessment"""
        high_quality_signal = np.random.randn(5000, 12) * 0.01  # Low noise
        
        quality_metrics = await service._assess_signal_quality(high_quality_signal)
        
        assert 'snr_db' in quality_metrics
        assert 'baseline_stability' in quality_metrics
        assert 'overall_score' in quality_metrics
        assert quality_metrics['overall_score'] > 0.5
    
    @pytest.mark.asyncio
    async def test_clinical_assessment_normal(self, service):
        """Test clinical assessment for normal ECG"""
        ai_results = {
            'predictions': {'normal': 0.9, 'atrial_fibrillation': 0.1},
            'confidence': 0.9
        }
        pathology_results = {
            'atrial_fibrillation': {'detected': False, 'confidence': 0.1}
        }
        features = {'rr_mean': 800, 'qtc_bazett': 400}
        
        assessment = await service._generate_clinical_assessment(
            ai_results, pathology_results, features
        )
        
        assert assessment['primary_diagnosis'] == 'Normal ECG'
        assert assessment['clinical_urgency'].value == 'low'
        assert not assessment['requires_immediate_attention']
    
    @pytest.mark.asyncio
    async def test_clinical_assessment_af(self, service):
        """Test clinical assessment for atrial fibrillation"""
        ai_results = {
            'predictions': {'normal': 0.1, 'atrial_fibrillation': 0.8},
            'confidence': 0.8
        }
        pathology_results = {
            'atrial_fibrillation': {'detected': True, 'confidence': 0.8}
        }
        features = {'rr_mean': 800, 'qtc_bazett': 400}
        
        assessment = await service._generate_clinical_assessment(
            ai_results, pathology_results, features
        )
        
        assert assessment['primary_diagnosis'] == 'Atrial Fibrillation'
        assert assessment['clinical_urgency'].value == 'high'
        assert 'Anticoagulation assessment recommended' in assessment['recommendations']


class TestRegulatoryValidationService:
    """Test Regulatory Validation Service"""
    
    def setup_method(self):
        self.validation_service = RegulatoryValidationService()
    
    @pytest.mark.asyncio
    async def test_fda_validation_compliant(self):
        """Test FDA validation for compliant analysis"""
        analysis_results = {
            'ai_predictions': {
                'confidence': 0.90,
                'predictions': {'normal': 0.9},
                'model_version': 'hybrid_v1.0'
            },
            'signal_quality': {'overall_score': 0.8},
            'clinical_assessment': {
                'requires_immediate_attention': False,
                'recommendations': ['Regular follow-up recommended']
            },
            'metadata': {
                'gdpr_compliant': True,
                'ce_marking': True
            }
        }
        
        result = await self.validation_service._validate_single_standard(
            RegulatoryStandard.FDA, analysis_results
        )
        
        assert result.standard == RegulatoryStandard.FDA
        assert result.compliant is True
        assert len(result.validation_errors) == 0
    
    @pytest.mark.asyncio
    async def test_fda_validation_non_compliant(self):
        """Test FDA validation for non-compliant analysis"""
        analysis_results = {
            'ai_predictions': {
                'confidence': 0.70,  # Below FDA threshold
                'predictions': {'normal': 0.7}
            },
            'signal_quality': {'overall_score': 0.6},  # Below FDA threshold
            'clinical_assessment': {
                'requires_immediate_attention': True,
                'recommendations': []
            }
        }
        
        result = await self.validation_service._validate_single_standard(
            RegulatoryStandard.FDA, analysis_results
        )
        
        assert result.standard == RegulatoryStandard.FDA
        assert result.compliant is False
        assert len(result.validation_errors) > 0
    
    @pytest.mark.asyncio
    async def test_eu_mdr_validation(self):
        """Test EU MDR specific validation"""
        analysis_results = {
            'ai_predictions': {'confidence': 0.90, 'predictions': {'normal': 0.9}},
            'signal_quality': {'overall_score': 0.8},
            'clinical_assessment': {'requires_immediate_attention': False},
            'metadata': {
                'gdpr_compliant': True,
                'ce_marking': True,
                'surveillance_plan': True
            }
        }
        
        result = await self.validation_service._validate_single_standard(
            RegulatoryStandard.EU_MDR, analysis_results
        )
        
        assert result.standard == RegulatoryStandard.EU_MDR
        assert result.compliant is True
    
    @pytest.mark.asyncio
    async def test_anvisa_validation(self):
        """Test ANVISA specific validation"""
        analysis_results = {
            'ai_predictions': {'confidence': 0.85, 'predictions': {'normal': 0.85}},
            'signal_quality': {'overall_score': 0.7},
            'clinical_assessment': {'requires_immediate_attention': False},
            'metadata': {
                'language_support': True,
                'population_validation': True
            }
        }
        
        result = await self.validation_service._validate_single_standard(
            RegulatoryStandard.ANVISA, analysis_results
        )
        
        assert result.standard == RegulatoryStandard.ANVISA
        assert result.compliant is True
    
    @pytest.mark.asyncio
    async def test_nmsa_validation(self):
        """Test NMSA (China) specific validation"""
        analysis_results = {
            'ai_predictions': {'confidence': 0.85, 'predictions': {'normal': 0.85}},
            'signal_quality': {'overall_score': 0.7},
            'clinical_assessment': {'requires_immediate_attention': False},
            'metadata': {
                'nmsa_certification': True,
                'data_residency': True
            }
        }
        
        result = await self.validation_service._validate_single_standard(
            RegulatoryStandard.NMSA, analysis_results
        )
        
        assert result.standard == RegulatoryStandard.NMSA
        assert result.compliant is True
    
    @pytest.mark.asyncio
    async def test_comprehensive_validation(self):
        """Test comprehensive validation against all standards"""
        analysis_results = {
            'analysis_id': 'TEST_001',
            'patient_id': 123,
            'processing_time_seconds': 2.5,
            'ai_predictions': {
                'confidence': 0.90,
                'predictions': {'normal': 0.9},
                'model_version': 'hybrid_v1.0'
            },
            'signal_quality': {'overall_score': 0.8},
            'clinical_assessment': {
                'requires_immediate_attention': False,
                'recommendations': ['Regular follow-up recommended']
            },
            'metadata': {
                'sampling_rate': 500,
                'gdpr_compliant': True,
                'ce_marking': True,
                'surveillance_plan': True,
                'nmsa_certification': True,
                'data_residency': True,
                'language_support': True,
                'population_validation': True
            }
        }
        
        results = await self.validation_service.validate_analysis_comprehensive(
            analysis_results
        )
        
        assert len(results) == 4  # All four standards
        assert RegulatoryStandard.FDA in results
        assert RegulatoryStandard.ANVISA in results
        assert RegulatoryStandard.NMSA in results
        assert RegulatoryStandard.EU_MDR in results
        
        for standard, result in results.items():
            assert result.compliant is True
    
    @pytest.mark.asyncio
    async def test_validation_report_generation(self):
        """Test validation report generation"""
        validation_results = {
            RegulatoryStandard.FDA: ValidationResult(
                standard=RegulatoryStandard.FDA,
                compliant=True,
                confidence_score=0.95,
                validation_errors=[],
                validation_warnings=[],
                test_results={'confidence': 0.90},
                timestamp=datetime.utcnow()
            ),
            RegulatoryStandard.EU_MDR: ValidationResult(
                standard=RegulatoryStandard.EU_MDR,
                compliant=False,
                confidence_score=0.70,
                validation_errors=['Missing GDPR compliance'],
                validation_warnings=[],
                test_results={'confidence': 0.85},
                timestamp=datetime.utcnow()
            )
        }
        
        report = await self.validation_service.generate_validation_report(
            validation_results
        )
        
        assert 'validation_timestamp' in report
        assert 'overall_compliance' in report
        assert 'standards_summary' in report
        assert 'recommendations' in report
        assert 'next_steps' in report
        
        assert report['overall_compliance'] is False  # One standard failed
        assert len(report['recommendations']) > 0
        assert 'Address validation errors' in report['next_steps'][0]
    
    @pytest.mark.asyncio
    async def test_performance_metrics_calculation(self):
        """Test performance metrics calculation with ground truth"""
        analysis_results = {
            'ai_predictions': {
                'predictions': {
                    'normal': 0.9,
                    'atrial_fibrillation': 0.1,
                    'ventricular_tachycardia': 0.05
                }
            }
        }
        
        ground_truth = {
            'labels': {
                'normal': 1,
                'atrial_fibrillation': 0,
                'ventricular_tachycardia': 0
            }
        }
        
        metrics = await self.validation_service._calculate_performance_metrics(
            analysis_results, ground_truth
        )
        
        assert 'sensitivity' in metrics
        assert 'specificity' in metrics
        assert 'precision' in metrics
        assert 'false_positive_rate' in metrics
        
        assert metrics['normal_sensitivity'] == 1.0
        assert metrics['normal_specificity'] == 1.0


class TestIntegrationCompliance:
    """Integration tests for regulatory compliance"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_fda_compliance(self, tmp_path):
        """Test end-to-end FDA compliance"""
        csv_file = tmp_path / "fda_test_ecg.csv"
        test_data = "I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6\n"
        
        t = np.linspace(0, 10, 5000)
        for i in range(5000):
            values = [str(np.sin(2 * np.pi * 1.2 * t[i]) + np.random.randn() * 0.01) for _ in range(12)]
            test_data += ",".join(values) + "\n"
        
        csv_file.write_text(test_data)
        
        mock_db = Mock()
        mock_validation_service = Mock()
        hybrid_service = HybridECGAnalysisService(mock_db, mock_validation_service)
        regulatory_service = RegulatoryValidationService()
        
        hybrid_service.repository = Mock()
        
        analysis_results = await hybrid_service.analyze_ecg_comprehensive(
            file_path=str(csv_file),
            patient_id=123,
            analysis_id="FDA_TEST_001"
        )
        
        validation_results = await regulatory_service.validate_analysis_comprehensive(
            analysis_results
        )
        
        fda_result = validation_results[RegulatoryStandard.FDA]
        assert fda_result.compliant is True
        assert fda_result.confidence_score > 0.8
        assert len(fda_result.validation_errors) == 0
    
    @pytest.mark.asyncio
    async def test_critical_condition_validation(self, tmp_path):
        """Test validation for critical conditions"""
        csv_file = tmp_path / "critical_test_ecg.csv"
        test_data = "I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6\n"
        
        for i in range(5000):
            irregular_factor = 1 + 0.5 * np.random.randn()
            values = [str(irregular_factor * np.random.randn() * 0.2) for _ in range(12)]
            test_data += ",".join(values) + "\n"
        
        csv_file.write_text(test_data)
        
        mock_db = Mock()
        mock_validation_service = Mock()
        hybrid_service = HybridECGAnalysisService(mock_db, mock_validation_service)
        regulatory_service = RegulatoryValidationService()
        
        hybrid_service.repository = Mock()
        
        analysis_results = await hybrid_service.analyze_ecg_comprehensive(
            file_path=str(csv_file),
            patient_id=123,
            analysis_id="CRITICAL_TEST_001"
        )
        
        validation_results = await regulatory_service.validate_analysis_comprehensive(
            analysis_results
        )
        
        report = await regulatory_service.generate_validation_report(validation_results)
        
        assert 'validation_timestamp' in report
        assert 'overall_compliance' in report
        assert 'standards_summary' in report
        
        assert len(report['standards_summary']) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
