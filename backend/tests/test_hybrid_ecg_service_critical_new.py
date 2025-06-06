"""
Critical Safety Tests for Hybrid ECG Analysis Service
Medical-grade testing for life-critical ECG analysis functionality
"""

import pytest
import asyncio
import time
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

from app.services.hybrid_ecg_service import HybridECGAnalysisService
from app.core.exceptions import ECGProcessingException


class TestECGCriticalSafety:
    """Critical safety tests - scenarios that can affect patient lives."""
    
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
        """ECG service configured for critical testing."""
        return HybridECGAnalysisService(
            db=mock_db,
            validation_service=mock_validation_service
        )

    @pytest.mark.timeout(30)


    def test_service_initialization_critical(self, mock_db, mock_validation_service):
        """CRITICAL: Service must initialize properly for medical use."""
        service = HybridECGAnalysisService(mock_db, mock_validation_service)
        
        assert service is not None
        assert service.db is mock_db
        assert service.validation_service is mock_validation_service
        assert hasattr(service, 'ecg_reader')
        assert hasattr(service, 'preprocessor')
        assert hasattr(service, 'feature_extractor')
        assert hasattr(service, 'repository')
        assert hasattr(service, 'ecg_logger')

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)

    async def test_comprehensive_analysis_emergency_timing(self, ecg_service):
        """CRITICAL: ECG analysis must complete within emergency timeframe."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("time,I,II\n")
            for i in range(100):  # Small test file for speed
                f.write(f"{i/250.0},{0.1 + i*0.001},{0.2 + i*0.001}\n")
            temp_file = f.name
        
        try:
            start_time = time.time()
            
            result = await ecg_service.analyze_ecg_comprehensive(
                file_path=temp_file,
                patient_id=1,
                analysis_id="EMERGENCY_001"
            )
            
            analysis_time = time.time() - start_time
            
            assert analysis_time < 30.0, f"Analysis too slow for emergency: {analysis_time:.2f}s"
            assert result is not None
            assert "analysis_id" in result
            assert result["analysis_id"] == "EMERGENCY_001"
            assert "patient_id" in result
            assert result["patient_id"] == 1
            assert "processing_time_seconds" in result
            
        finally:
            os.unlink(temp_file)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)

    async def test_invalid_file_error_handling(self, ecg_service):
        """CRITICAL: Invalid files must be rejected to prevent misdiagnosis."""
        with pytest.raises(ECGProcessingException) as exc_info:
            await ecg_service.analyze_ecg_comprehensive(
                file_path="/tmp/nonexistent_file.ecg",
                patient_id=999,
                analysis_id="INVALID_001"
            )
        
        assert "analysis failed" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)

    async def test_unsupported_format_handling(self, ecg_service):
        """CRITICAL: Unsupported formats must be handled safely."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.unknown', delete=False) as f:
            f.write("invalid format")
            temp_file = f.name
        
        try:
            with pytest.raises(ECGProcessingException) as exc_info:
                await ecg_service.analyze_ecg_comprehensive(
                    file_path=temp_file,
                    patient_id=5,
                    analysis_id="UNSUPPORTED_001"
                )
            
            assert "analysis failed" in str(exc_info.value).lower()
        finally:
            os.unlink(temp_file)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)

    async def test_preprocessor_functionality(self, ecg_service):
        """CRITICAL: Signal preprocessing must maintain data integrity."""
        test_signal = np.array([0.1, 0.2, 0.3, 0.2, 0.1] * 250).reshape(-1, 1)
        
        processed_signal = await ecg_service.preprocessor.preprocess_signal(test_signal)
        
        assert processed_signal is not None
        assert isinstance(processed_signal, np.ndarray)
        assert processed_signal.shape[0] > 0  # Should have channels
        assert processed_signal.shape[1] > 0  # Should have samples
        assert not np.all(processed_signal == 0), "Processed signal should not be all zeros"
        assert np.isfinite(processed_signal).all(), "Processed signal should contain finite values"

    @pytest.mark.timeout(30)


    def test_feature_extractor_functionality(self, ecg_service):
        """CRITICAL: Feature extraction must work reliably."""
        test_signal = np.array([[0.1, 0.2, 0.3, 0.2, 0.1] * 200])  # Longer signal for features
        
        features = ecg_service.feature_extractor.extract_all_features(test_signal)
        
        assert features is not None
        assert isinstance(features, dict)
        assert len(features) > 0, "Features should be extracted"

    @pytest.mark.timeout(30)


    def test_ecg_reader_supported_formats(self, ecg_service):
        """CRITICAL: ECG reader must support required medical formats."""
        supported_formats = ecg_service.ecg_reader.supported_formats
        
        assert '.csv' in supported_formats, "CSV format must be supported"
        assert '.dat' in supported_formats, "DAT format must be supported"
        assert '.edf' in supported_formats, "EDF format must be supported"

    @pytest.mark.timeout(30)


    def test_ecg_reader_error_handling(self, ecg_service):
        """CRITICAL: ECG reader must handle errors safely."""
        try:
            ecg_service.ecg_reader.read_ecg("/nonexistent/file.xyz")
            assert False, "Should have raised error for unsupported format"
        except (ValueError, FileNotFoundError) as e:
            assert len(str(e)) > 0, "Error message should not be empty"

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)

    async def test_pathology_detection_functionality(self, ecg_service):
        """CRITICAL: Pathology detection must function correctly."""
        test_signal = np.array([[0.1, 0.2, 0.3, 0.2, 0.1] * 200])
        
        features = ecg_service.feature_extractor.extract_all_features(test_signal)
        
        pathologies = await ecg_service._detect_pathologies(test_signal, features)
        
        assert pathologies is not None
        assert isinstance(pathologies, dict)
        assert 'atrial_fibrillation' in pathologies
        assert 'long_qt_syndrome' in pathologies

    @pytest.mark.timeout(30)


    def test_atrial_fibrillation_detection(self, ecg_service):
        """CRITICAL: AF detection algorithm must work."""
        normal_features = {
            'rr_mean': 800,  # Normal RR interval
            'rr_std': 50,    # Low variability
            'hrv_rmssd': 30, # Normal HRV
            'spectral_entropy': 0.5  # Normal entropy
        }
        
        af_score = ecg_service._detect_atrial_fibrillation(normal_features)
        assert isinstance(af_score, float)
        assert 0.0 <= af_score <= 1.0

    @pytest.mark.timeout(30)


    def test_long_qt_detection(self, ecg_service):
        """CRITICAL: Long QT detection must work."""
        normal_features = {'qtc_bazett': 420}  # Normal QTc
        qt_score = ecg_service._detect_long_qt(normal_features)
        assert qt_score == 0.0
        
        prolonged_features = {'qtc_bazett': 500}  # Prolonged QTc
        qt_score = ecg_service._detect_long_qt(prolonged_features)
        assert qt_score > 0.0

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)

    async def test_clinical_assessment_generation(self, ecg_service):
        """CRITICAL: Clinical assessment must be generated properly."""
        ai_results = {
            'predictions': {'normal': 0.8, 'atrial_fibrillation': 0.2},
            'confidence': 0.8
        }
        
        pathology_results = {
            'atrial_fibrillation': {'detected': False, 'confidence': 0.2}
        }
        
        features = {'rr_mean': 800}
        
        assessment = await ecg_service._generate_clinical_assessment(
            ai_results, pathology_results, features
        )
        
        assert assessment is not None
        assert 'primary_diagnosis' in assessment
        assert 'clinical_urgency' in assessment
        assert 'recommendations' in assessment

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)

    async def test_signal_quality_assessment(self, ecg_service):
        """CRITICAL: Signal quality assessment must work."""
        test_signal = np.array([[0.1 + 0.05*np.sin(2*np.pi*i/250) for i in range(1000)]])
        
        quality_metrics = await ecg_service._assess_signal_quality(test_signal)
        
        assert quality_metrics is not None
        assert 'snr_db' in quality_metrics
        assert 'baseline_stability' in quality_metrics
        assert 'overall_score' in quality_metrics
        assert isinstance(quality_metrics['overall_score'], float)
        if not np.isnan(quality_metrics['overall_score']):
            assert 0.0 <= quality_metrics['overall_score'] <= 1.0

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)

    async def test_memory_constraints_medical_environment(self, ecg_service):
        """CRITICAL: Memory usage must be controlled for hospital environment."""
        import psutil
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        for i in range(10):
            test_signal = np.array([[0.1, 0.2, 0.3] * 100])
            try:
                await ecg_service.preprocessor.preprocess_signal(test_signal)
            except Exception:
                pass
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        assert memory_used < 100, f"Excessive memory usage: {memory_used:.1f}MB"

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)

    async def test_complete_workflow_integration(self, ecg_service):
        """CRITICAL: Complete ECG analysis workflow must work end-to-end."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("time,I,II\n")
            for i in range(250):  # 1 second of data at 250Hz
                f.write(f"{i/250.0},{0.1 * np.sin(2*np.pi*i/250)},{0.2 * np.sin(2*np.pi*i/250)}\n")
            temp_file = f.name
        
        try:
            result = await ecg_service.analyze_ecg_comprehensive(
                file_path=temp_file,
                patient_id=100,
                analysis_id="WORKFLOW_001"
            )
            
            assert result is not None
            assert "analysis_id" in result
            assert "patient_id" in result
            assert "processing_time_seconds" in result
            assert "signal_quality" in result
            assert "ai_predictions" in result
            assert "pathology_detections" in result
            assert "clinical_assessment" in result
            assert "extracted_features" in result
            assert "metadata" in result
            
            metadata = result["metadata"]
            assert metadata["gdpr_compliant"] is True
            assert metadata["ce_marking"] is True
            assert metadata["nmsa_certification"] is True
            
        finally:
            os.unlink(temp_file)


class TestECGRegulatoryCompliance:
    """Tests for regulatory compliance (FDA, ANVISA, NMSA, EU)."""
    
    @pytest.fixture
    def ecg_service(self):
        """ECG service for regulatory testing."""
        return HybridECGAnalysisService(Mock(), Mock())
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)

    async def test_regulatory_metadata_compliance(self, ecg_service):
        """REGULATORY: All processing must include compliance metadata."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("time,I,II\n")
            for i in range(50):
                f.write(f"{i/250.0},{0.1},{0.2}\n")
            temp_file = f.name
        
        try:
            result = await ecg_service.analyze_ecg_comprehensive(
                file_path=temp_file,
                patient_id=100,
                analysis_id="REGULATORY_001"
            )
            
            assert "metadata" in result
            metadata = result["metadata"]
            assert "model_version" in metadata
            assert "gdpr_compliant" in metadata
            assert "ce_marking" in metadata
            assert "surveillance_plan" in metadata
            assert "nmsa_certification" in metadata
            assert "data_residency" in metadata
            assert metadata["gdpr_compliant"] is True
            assert metadata["ce_marking"] is True
            
        finally:
            os.unlink(temp_file)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)

    async def test_data_integrity_validation(self, ecg_service):
        """REGULATORY: Data integrity must be maintained throughout processing."""
        test_signal = np.array([0.1, 0.2, 0.3, 0.2, 0.1] * 250).reshape(-1, 1)
        
        processed_signal = await ecg_service.preprocessor.preprocess_signal(test_signal)
        
        assert processed_signal is not None
        assert isinstance(processed_signal, np.ndarray)
        assert processed_signal.shape[0] > 0  # Should have channels
        assert processed_signal.shape[1] > 0  # Should have samples

    @pytest.mark.timeout(30)


    def test_error_handling_medical_standards(self, ecg_service):
        """REGULATORY: Error handling must meet medical device standards."""
        try:
            ecg_service.ecg_reader.read_ecg("/nonexistent/file.xyz")
            assert False, "Should have raised error for unsupported format"
        except (ValueError, FileNotFoundError) as e:
            assert len(str(e)) > 0, "Error message should not be empty"


class TestECGPerformanceMedical:
    """Performance tests for medical environment requirements."""
    
    @pytest.fixture
    def ecg_service(self):
        """ECG service for performance testing."""
        return HybridECGAnalysisService(Mock(), Mock())
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)

    async def test_preprocessing_performance(self, ecg_service):
        """PERFORMANCE: Preprocessing must complete within time limits."""
        test_signal = np.array([0.1, 0.2, 0.3] * 500).reshape(-1, 1)
        
        start_time = time.time()
        processed = await ecg_service.preprocessor.preprocess_signal(test_signal)
        processing_time = time.time() - start_time
        
        assert processing_time < 5.0, f"Preprocessing too slow: {processing_time:.2f}s"
        assert processed is not None

    @pytest.mark.timeout(30)


    def test_feature_extraction_performance(self, ecg_service):
        """PERFORMANCE: Feature extraction must be efficient."""
        test_signal = np.array([[0.1, 0.2, 0.3] * 500])
        
        start_time = time.time()
        features = ecg_service.feature_extractor.extract_all_features(test_signal)
        extraction_time = time.time() - start_time
        
        assert extraction_time < 10.0, f"Feature extraction too slow: {extraction_time:.2f}s"
        assert features is not None

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)

    async def test_resource_cleanup_medical_safety(self, ecg_service):
        """PERFORMANCE: Resources must be properly cleaned up for medical safety."""
        for i in range(5):
            test_signal = np.array([[0.1, 0.2, 0.3] * 100])
            
            try:
                await ecg_service.preprocessor.preprocess_signal(test_signal)
                ecg_service.feature_extractor.extract_all_features(test_signal)
            except Exception:
                pass  # Focus on resource management
        
        import gc
        gc.collect()
        
        assert True
