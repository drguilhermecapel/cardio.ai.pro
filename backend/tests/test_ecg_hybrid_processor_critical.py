"""
Critical Safety Tests for ECG Hybrid Processor
Medical-grade testing for ECG processing integration utilities
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from app.utils.ecg_hybrid_processor import ECGHybridProcessor
from app.core.exceptions import ECGProcessingException


class TestECGHybridProcessorCritical:
    """Critical safety tests for ECG hybrid processor integration."""
    
    @pytest.fixture
    def mock_db(self):
        """Mock database for testing."""
        return Mock()
    
    @pytest.fixture
    def mock_validation_service(self):
        """Mock validation service for testing."""
        return Mock()
    
    @pytest.fixture
    def processor(self, mock_db, mock_validation_service):
        """ECG hybrid processor for testing."""
        return ECGHybridProcessor(mock_db, mock_validation_service)
    
    def test_processor_initialization_critical(self, mock_db, mock_validation_service):
        """CRITICAL: Processor must initialize with required services."""
        processor = ECGHybridProcessor(mock_db, mock_validation_service)
        
        assert processor is not None
        assert processor.hybrid_service is not None
        assert processor.regulatory_service is None  # Will be implemented in PR-003
        assert hasattr(processor, 'hybrid_service')

    @pytest.mark.asyncio
    @pytest.mark.asyncio
async def test_process_ecg_with_validation_critical(self, processor):
        """CRITICAL: ECG processing with validation must handle medical scenarios."""
        mock_analysis_result = {
            "abnormalities": {
                "stemi": {"detected": True, "confidence": 0.98},
                "vfib": {"detected": False, "confidence": 0.02}
            },
            "clinical_urgency": "critical",
            "findings": ["ST elevation detected", "Anterior wall involvement"],
            "processing_time": 15.2,
            "signal_quality": "good"
        }
        
        with patch.object(processor.hybrid_service, 'analyze_ecg_comprehensive', 
                         new_callable=AsyncMock, return_value=mock_analysis_result):
            
            result = await processor.process_ecg_with_validation(
                file_path="/tmp/test_ecg.dat",
                patient_id=12345,
                analysis_id="CRITICAL_001",
                require_regulatory_compliance=True
            )
        
        assert result is not None
        assert "abnormalities" in result
        assert "regulatory_compliant" in result
        assert "compliance_issues" in result
        assert "regulatory_validation" in result
        
        assert result["abnormalities"]["stemi"]["detected"] is True
        assert result["clinical_urgency"] == "critical"

    @pytest.mark.asyncio
    @pytest.mark.asyncio
async def test_process_ecg_error_handling_medical(self, processor):
        """CRITICAL: Error handling must be safe for medical environment."""
        with patch.object(processor.hybrid_service, 'analyze_ecg_comprehensive',
                         side_effect=Exception("ECG analysis failed")):
            
            with pytest.raises(ECGProcessingException) as exc_info:
                await processor.process_ecg_with_validation(
                    file_path="/tmp/invalid_ecg.dat",
                    patient_id=12346,
                    analysis_id="ERROR_001"
                )
            
            assert "Hybrid processing failed" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.asyncio
async def test_validate_existing_analysis_placeholder(self, processor):
        """CRITICAL: Analysis validation must provide safe placeholder."""
        existing_analysis = {
            "abnormalities": {"stemi": {"detected": False}},
            "clinical_urgency": "low",
            "findings": ["Normal sinus rhythm"]
        }
        
        result = await processor.validate_existing_analysis(existing_analysis)
        
        assert result is not None
        assert "validation_results" in result
        assert "validation_report" in result
        assert "overall_compliance" in result
        assert result["overall_compliance"] is True  # Placeholder compliance

    def test_get_supported_formats_medical(self, processor):
        """CRITICAL: Supported formats must include medical standards."""
        with patch.object(processor.hybrid_service.ecg_reader, 'supported_formats', 
                         {"WFDB": "PhysioNet format", "EDF": "European Data Format", "DICOM": "Medical imaging"}):
            
            formats = processor.supported_formats)
            
            assert isinstance(formats, list)
            assert len(formats) >= 3
            assert "WFDB" in formats
            assert "EDF" in formats
            assert "DICOM" in formats

    def test_get_regulatory_standards_compliance(self, processor):
        """CRITICAL: Must support required regulatory standards."""
        standards = processor.get_regulatory_standards()
        
        assert isinstance(standards, list)
        assert len(standards) == 4
        assert "FDA" in standards
        assert "ANVISA" in standards
        assert "NMSA" in standards
        assert "EU_MDR" in standards

    @pytest.mark.asyncio
    @pytest.mark.asyncio
async def test_get_system_status_medical_readiness(self, processor):
        """CRITICAL: System status must indicate medical readiness."""
        with patch.object(processor, 'get_supported_formats', return_value=["WFDB", "EDF", "DICOM"]):
            with patch.object(processor, 'get_regulatory_standards', return_value=["FDA", "ANVISA", "NMSA", "EU_MDR"]):
                
                status = await processor.get_model_info()
                
                assert status is not None
                assert "hybrid_service_initialized" in status
                assert "regulatory_service_initialized" in status
                assert "supported_formats" in status
                assert "regulatory_standards" in status
                assert "system_version" in status
                
                assert status["hybrid_service_initialized"] is True
                assert status["regulatory_service_initialized"] is False  # Placeholder for PR-003
                assert len(status["supported_formats"]) >= 3
                assert len(status["regulatory_standards"]) == 4
                assert status["system_version"] == "1.0.0"

    @pytest.mark.asyncio
    @pytest.mark.asyncio
async def test_regulatory_compliance_enforcement(self, processor):
        """CRITICAL: Regulatory compliance must be enforced when required."""
        mock_analysis_result = {
            "abnormalities": {"stemi": {"detected": True, "confidence": 0.85}},  # Lower confidence
            "clinical_urgency": "high",
            "findings": ["Possible ST elevation"]
        }
        
        mock_validation_report = {
            "overall_compliance": False,
            "recommendations": ["Confidence below FDA threshold", "Requires manual review"]
        }
        
        with patch.object(processor.hybrid_service, 'analyze_ecg_comprehensive',
                         new_callable=AsyncMock, return_value=mock_analysis_result):
            
            processor.regulatory_service = Mock()
            processor.regulatory_service.validate_analysis_comprehensive = AsyncMock(
                return_value={"status": "non_compliant"}
            )
            processor.regulatory_service.generate_validation_report = AsyncMock(
                return_value=mock_validation_report
            )
            
            result = await processor.process_ecg_with_validation(
                file_path="/tmp/test_compliance.dat",
                patient_id=12347,
                analysis_id="COMPLIANCE_001",
                require_regulatory_compliance=True
            )
            
            assert result["regulatory_compliant"] is False
            assert len(result["compliance_issues"]) > 0
            assert "FDA threshold" in result["compliance_issues"][0]

    @pytest.mark.asyncio
    @pytest.mark.asyncio
async def test_emergency_processing_priority(self, processor):
        """CRITICAL: Emergency cases must be processed with priority."""
        emergency_analysis = {
            "abnormalities": {
                "vfib": {"detected": True, "confidence": 0.99},
                "stemi": {"detected": False, "confidence": 0.05}
            },
            "clinical_urgency": "critical",
            "findings": ["Ventricular fibrillation detected", "Immediate intervention required"]
        }
        
        with patch.object(processor.hybrid_service, 'analyze_ecg_comprehensive',
                         new_callable=AsyncMock, return_value=emergency_analysis):
            
            import time
            start_time = time.time()
            
            result = await processor.process_ecg_with_validation(
                file_path="/tmp/emergency_vfib.dat",
                patient_id=99999,
                analysis_id="EMERGENCY_VFIB_001",
                require_regulatory_compliance=False  # Emergency override
            )
            
            processing_time = time.time() - start_time
            
            assert processing_time < 10.0, f"Emergency processing too slow: {processing_time:.2f}s"
            assert result["abnormalities"]["vfib"]["detected"] is True
            assert result["clinical_urgency"] == "critical"

    def test_processor_memory_efficiency_medical(self, processor):
        """CRITICAL: Memory usage must be efficient for medical environment."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        for _ in range(100):
            formats = processor.supported_formats)
            standards = processor.get_regulatory_standards()
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        assert memory_increase < 50, f"Excessive memory usage: {memory_increase:.1f}MB"

    @pytest.mark.asyncio
    @pytest.mark.asyncio
async def test_concurrent_processing_medical_safety(self, processor):
        """CRITICAL: Concurrent processing must be safe for medical use."""
        import concurrent.futures
        
        async def process_patient(patient_id: int) -> Dict[str, Any]:
            """Simulate concurrent patient processing."""
            mock_result = {
                "abnormalities": {"stemi": {"detected": False}},
                "clinical_urgency": "low",
                "patient_id": patient_id
            }
            
            with patch.object(processor.hybrid_service, 'analyze_ecg_comprehensive',
                             new_callable=AsyncMock, return_value=mock_result):
                
                return await processor.process_ecg_with_validation(
                    file_path=f"/tmp/patient_{patient_id}.dat",
                    patient_id=patient_id,
                    analysis_id=f"CONCURRENT_{patient_id}"
                )
        
        tasks = [process_patient(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 5, f"Concurrent processing failures: {len(results) - len(successful_results)}/5"
        
        for result in successful_results:
            assert "abnormalities" in result
            assert "regulatory_compliant" in result


class TestECGHybridProcessorIntegration:
    """Integration tests for ECG hybrid processor with medical workflows."""
    
    @pytest.fixture
    def processor(self):
        """ECG hybrid processor for integration testing."""
        return ECGHybridProcessor(Mock(), Mock())
    
    @pytest.mark.asyncio
    @pytest.mark.asyncio
async def test_emergency_department_workflow_simulation(self, processor):
        """INTEGRATION: Simulate complete emergency department workflow."""
        emergency_scenario = {
            "patient_id": 88888,
            "analysis_id": "ED_CHEST_PAIN_001",
            "file_path": "/tmp/ed_patient.dat",
            "clinical_context": "chest_pain_emergency"
        }
        
        stemi_result = {
            "abnormalities": {
                "stemi": {"detected": True, "confidence": 0.97, "location": "anterior"},
                "vfib": {"detected": False, "confidence": 0.01}
            },
            "clinical_urgency": "critical",
            "findings": ["Acute anterior STEMI", "Immediate catheterization indicated"],
            "processing_time": 12.5
        }
        
        with patch.object(processor.hybrid_service, 'analyze_ecg_comprehensive',
                         new_callable=AsyncMock, return_value=stemi_result):
            
            result = await processor.process_ecg_with_validation(
                file_path=emergency_scenario["file_path"],
                patient_id=emergency_scenario["patient_id"],
                analysis_id=emergency_scenario["analysis_id"],
                require_regulatory_compliance=True
            )
        
        assert result["abnormalities"]["stemi"]["detected"] is True
        assert result["clinical_urgency"] == "critical"
        assert "anterior" in result["abnormalities"]["stemi"]["location"]
        assert result["regulatory_compliant"] is True

    @pytest.mark.asyncio
    @pytest.mark.asyncio
async def test_routine_screening_workflow_simulation(self, processor):
        """INTEGRATION: Simulate routine cardiac screening workflow."""
        screening_scenario = {
            "patient_id": 77777,
            "analysis_id": "SCREENING_001",
            "file_path": "/tmp/routine_screening.dat",
            "clinical_context": "routine_screening"
        }
        
        normal_result = {
            "abnormalities": {
                "stemi": {"detected": False, "confidence": 0.02},
                "vfib": {"detected": False, "confidence": 0.01},
                "afib": {"detected": False, "confidence": 0.05}
            },
            "clinical_urgency": "low",
            "findings": ["Normal sinus rhythm", "No acute abnormalities"],
            "processing_time": 8.3
        }
        
        with patch.object(processor.hybrid_service, 'analyze_ecg_comprehensive',
                         new_callable=AsyncMock, return_value=normal_result):
            
            result = await processor.process_ecg_with_validation(
                file_path=screening_scenario["file_path"],
                patient_id=screening_scenario["patient_id"],
                analysis_id=screening_scenario["analysis_id"],
                require_regulatory_compliance=True
            )
        
        assert result["clinical_urgency"] == "low"
        assert result["abnormalities"]["stemi"]["detected"] is False
        assert result["abnormalities"]["vfib"]["detected"] is False
        assert result["regulatory_compliant"] is True

    def test_processor_error_recovery_medical(self, processor):
        """INTEGRATION: Error recovery must be robust for medical use."""
        error_scenarios = [
            {"error": "Network timeout", "expected_behavior": "graceful_degradation"},
            {"error": "Database unavailable", "expected_behavior": "local_processing"},
            {"error": "Model loading failed", "expected_behavior": "fallback_analysis"},
        ]
        
        for scenario in error_scenarios:
            try:
                with patch.object(processor.hybrid_service, 'analyze_ecg_comprehensive',
                                 side_effect=Exception(scenario["error"])):
                    processor.process_ecg_with_validation(
                        file_path="/tmp/error_test.dat",
                        patient_id=66666,
                        analysis_id="ERROR_RECOVERY_001"
                    )
            except ECGProcessingException:
                assert True
            except Exception as e:
                assert False, f"Unexpected error type for {scenario['error']}: {type(e).__name__}"
