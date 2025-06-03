"""
Tests for FHIR R4 Service and Healthcare Interoperability
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from app.services.fhir_service import (
    FHIRResourceMapper,
    EpicSystemsIntegration,
    FHIRService
)


class TestFHIRResourceMapper:
    """Test FHIR resource mapping functionality"""

    def test_init(self):
        """Test mapper initialization"""
        mapper = FHIRResourceMapper()
        
        assert "ecg_12_lead" in mapper.loinc_codes
        assert "heart_rate" in mapper.loinc_codes
        assert "atrial_fibrillation" in mapper.snomed_codes
        assert "ventricular_tachycardia" in mapper.snomed_codes

    def test_create_ecg_observation_basic(self):
        """Test basic ECG observation creation"""
        mapper = FHIRResourceMapper()
        
        ecg_data = np.random.randn(5000).astype(np.float32)
        analysis_results = {
            "heart_rate": 75.0,
            "rhythm": "normal_sinus_rhythm",
            "qt_interval": 400.0,
            "pr_interval": 160.0,
            "qrs_duration": 100.0
        }
        
        observation = mapper.create_ecg_observation(
            patient_id="patient-123",
            ecg_data=ecg_data,
            analysis_results=analysis_results
        )
        
        assert observation["resourceType"] == "Observation"
        assert observation["status"] == "final"
        assert observation["subject"]["reference"] == "Patient/patient-123"
        assert "component" in observation
        assert len(observation["component"]) > 0

    def test_create_ecg_observation_with_arrhythmias(self):
        """Test ECG observation creation with arrhythmia detection"""
        mapper = FHIRResourceMapper()
        
        ecg_data = np.random.randn(5000).astype(np.float32)
        analysis_results = {
            "heart_rate": 120.0,
            "arrhythmias": {
                "atrial_fibrillation": True,
                "ventricular_tachycardia": False
            }
        }
        
        observation = mapper.create_ecg_observation(
            patient_id="patient-456",
            ecg_data=ecg_data,
            analysis_results=analysis_results
        )
        
        assert observation["resourceType"] == "Observation"
        
        arrhythmia_components = [
            comp for comp in observation["component"]
            if "valueCodeableConcept" in comp
        ]
        assert len(arrhythmia_components) > 0

    def test_create_ecg_observation_with_sleep_apnea(self):
        """Test ECG observation creation with sleep apnea analysis"""
        mapper = FHIRResourceMapper()
        
        ecg_data = np.random.randn(5000).astype(np.float32)
        analysis_results = {
            "heart_rate": 65.0,
            "sleep_apnea_analysis": {
                "ahi": 15.5,
                "severity": "moderate"
            }
        }
        
        observation = mapper.create_ecg_observation(
            patient_id="patient-789",
            ecg_data=ecg_data,
            analysis_results=analysis_results
        )
        
        assert observation["resourceType"] == "Observation"
        
        sleep_components = [
            comp for comp in observation["component"]
            if comp.get("valueQuantity", {}).get("unit") == "events/hour"
        ]
        assert len(sleep_components) > 0

    def test_create_diagnostic_report(self):
        """Test diagnostic report creation"""
        mapper = FHIRResourceMapper()
        
        analysis_results = {
            "heart_rate": 80.0,
            "rhythm": "normal_sinus_rhythm"
        }
        
        report = mapper.create_diagnostic_report(
            patient_id="patient-123",
            observation_ids=["obs-1", "obs-2"],
            analysis_results=analysis_results
        )
        
        assert report["resourceType"] == "DiagnosticReport"
        assert report["status"] == "final"
        assert report["subject"]["reference"] == "Patient/patient-123"
        assert len(report["result"]) == 2
        assert "conclusion" in report

    def test_create_media_resource(self):
        """Test media resource creation"""
        mapper = FHIRResourceMapper()
        
        ecg_data = np.random.randn(1000).astype(np.float32)
        
        media = mapper.create_media_resource(
            patient_id="patient-123",
            ecg_waveform_data=ecg_data
        )
        
        assert media["resourceType"] == "Media"
        assert media["status"] == "completed"
        assert media["subject"]["reference"] == "Patient/patient-123"
        assert "content" in media
        assert "data" in media["content"]

    def test_create_bundle(self):
        """Test bundle creation"""
        mapper = FHIRResourceMapper()
        
        resources = [
            {"resourceType": "Observation", "id": "obs-1"},
            {"resourceType": "DiagnosticReport", "id": "report-1"}
        ]
        
        bundle = mapper.create_bundle(resources)
        
        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "collection"
        assert bundle["total"] == 2
        assert len(bundle["entry"]) == 2

    def test_generate_clinical_conclusion(self):
        """Test clinical conclusion generation"""
        mapper = FHIRResourceMapper()
        
        analysis_results = {
            "heart_rate": 45.0,  # Bradycardia
            "rhythm": "sinus_bradycardia",
            "arrhythmias": {
                "atrial_fibrillation": True
            },
            "sleep_apnea_analysis": {
                "severity": "severe"
            }
        }
        
        conclusion = mapper._generate_clinical_conclusion(analysis_results)
        
        assert "Rhythm: sinus_bradycardia" in conclusion
        assert "Bradycardia detected" in conclusion
        assert "Arrhythmias detected: atrial_fibrillation" in conclusion
        assert "Sleep apnea severity: severe" in conclusion

    @patch('app.services.fhir_service.FHIR_AVAILABLE', False)
    def test_create_basic_observation_without_fhir(self):
        """Test basic observation creation without FHIR library"""
        mapper = FHIRResourceMapper()
        
        analysis_results = {"heart_rate": 75.0}
        
        observation = mapper._create_basic_observation("patient-123", analysis_results)
        
        assert observation["resourceType"] == "Observation"
        assert observation["status"] == "final"
        assert observation["subject"]["reference"] == "Patient/patient-123"


class TestEpicSystemsIntegration:
    """Test Epic Systems integration functionality"""

    def test_init(self):
        """Test Epic integration initialization"""
        integration = EpicSystemsIntegration()
        
        assert integration.fhir_mapper is not None
        assert hasattr(integration, 'base_url')
        assert hasattr(integration, 'client_id')

    async def test_submit_ecg_results(self):
        """Test ECG results submission to Epic"""
        integration = EpicSystemsIntegration()
        
        ecg_data = np.random.randn(5000).astype(np.float32)
        analysis_results = {
            "heart_rate": 75.0,
            "rhythm": "normal_sinus_rhythm"
        }
        
        result = await integration.submit_ecg_results(
            patient_id="patient-123",
            ecg_data=ecg_data,
            analysis_results=analysis_results
        )
        
        assert result["status"] == "success"
        assert "bundle_id" in result
        assert "resources_created" in result
        assert "bundle" in result

    async def test_query_patient_data(self):
        """Test patient data query from Epic"""
        integration = EpicSystemsIntegration()
        
        result = await integration.query_patient_data("patient-123")
        
        assert result["status"] == "success"
        assert result["patient_id"] == "patient-123"
        assert "data" in result

    async def test_submit_ecg_results_error_handling(self):
        """Test error handling in Epic submission"""
        integration = EpicSystemsIntegration()
        
        with patch.object(integration.fhir_mapper, 'create_ecg_observation', side_effect=Exception("Test error")):
            result = await integration.submit_ecg_results(
                patient_id="patient-123",
                ecg_data=np.array([]),
                analysis_results={}
            )
            
            assert result["status"] == "error"
            assert "error" in result


class TestFHIRService:
    """Test main FHIR service functionality"""

    def test_init(self):
        """Test FHIR service initialization"""
        service = FHIRService()
        
        assert service.resource_mapper is not None
        assert service.epic_integration is not None

    async def test_process_ecg_for_fhir(self):
        """Test ECG processing for FHIR"""
        service = FHIRService()
        
        ecg_data = np.random.randn(5000).astype(np.float32)
        analysis_results = {
            "heart_rate": 75.0,
            "rhythm": "normal_sinus_rhythm"
        }
        
        result = await service.process_ecg_for_fhir(
            patient_id="patient-123",
            ecg_data=ecg_data,
            analysis_results=analysis_results,
            submit_to_epic=False
        )
        
        assert result["status"] == "success"
        assert "fhir_bundle" in result
        assert "observation_id" in result
        assert "diagnostic_report_id" in result
        assert "media_resource_id" in result

    async def test_process_ecg_for_fhir_with_epic(self):
        """Test ECG processing with Epic submission"""
        service = FHIRService()
        
        ecg_data = np.random.randn(5000).astype(np.float32)
        analysis_results = {
            "heart_rate": 75.0,
            "rhythm": "normal_sinus_rhythm"
        }
        
        result = await service.process_ecg_for_fhir(
            patient_id="patient-123",
            ecg_data=ecg_data,
            analysis_results=analysis_results,
            submit_to_epic=True
        )
        
        assert result["status"] == "success"
        assert "epic_submission" in result

    def test_validate_fhir_resource_valid(self):
        """Test FHIR resource validation with valid resource"""
        service = FHIRService()
        
        valid_observation = {
            "resourceType": "Observation",
            "status": "final",
            "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4"}]},
            "subject": {"reference": "Patient/123"}
        }
        
        result = service.validate_fhir_resource(valid_observation)
        
        assert result["valid"] is True

    def test_validate_fhir_resource_invalid(self):
        """Test FHIR resource validation with invalid resource"""
        service = FHIRService()
        
        invalid_observation = {
            "resourceType": "Observation",
            "status": "final"
        }
        
        result = service.validate_fhir_resource(invalid_observation)
        
        assert result["valid"] is False
        assert "errors" in result

    def test_validate_fhir_resource_missing_type(self):
        """Test FHIR resource validation with missing resourceType"""
        service = FHIRService()
        
        invalid_resource = {
            "status": "final",
            "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4"}]}
        }
        
        result = service.validate_fhir_resource(invalid_resource)
        
        assert result["valid"] is False
        assert "Missing resourceType" in result["errors"]

    async def test_process_ecg_error_handling(self):
        """Test error handling in ECG processing"""
        service = FHIRService()
        
        with patch.object(service.resource_mapper, 'create_ecg_observation', side_effect=Exception("Test error")):
            result = await service.process_ecg_for_fhir(
                patient_id="patient-123",
                ecg_data=np.array([]),
                analysis_results={}
            )
            
            assert result["status"] == "error"
            assert "error" in result

    @patch('app.services.fhir_service.FHIR_AVAILABLE', False)
    def test_validate_without_fhir_library(self):
        """Test validation without FHIR library"""
        service = FHIRService()
        
        resource = {"resourceType": "Observation"}
        
        result = service.validate_fhir_resource(resource)
        
        assert result["valid"] is True
        assert "FHIR validation library not available" in result["warnings"]


class TestFHIRIntegration:
    """Test FHIR integration scenarios"""

    async def test_complete_ecg_workflow(self):
        """Test complete ECG to FHIR workflow"""
        service = FHIRService()
        
        ecg_data = np.random.randn(5000).astype(np.float32)
        analysis_results = {
            "heart_rate": 75.0,
            "rhythm": "normal_sinus_rhythm",
            "qt_interval": 400.0,
            "pr_interval": 160.0,
            "qrs_duration": 100.0,
            "arrhythmias": {
                "atrial_fibrillation": False,
                "ventricular_tachycardia": False
            },
            "sleep_apnea_analysis": {
                "ahi": 3.2,
                "severity": "normal"
            },
            "risk_prediction": {
                "cardiovascular_risk": 0.15,
                "sudden_cardiac_death_risk": 0.02
            }
        }
        
        result = await service.process_ecg_for_fhir(
            patient_id="patient-comprehensive",
            ecg_data=ecg_data,
            analysis_results=analysis_results,
            submit_to_epic=True
        )
        
        assert result["status"] == "success"
        assert "fhir_bundle" in result
        assert "epic_submission" in result
        
        bundle = result["fhir_bundle"]
        assert bundle["resourceType"] == "Bundle"
        assert len(bundle["entry"]) == 3  # Observation, DiagnosticReport, Media
        
        observation = next(
            entry["resource"] for entry in bundle["entry"]
            if entry["resource"]["resourceType"] == "Observation"
        )
        assert len(observation["component"]) > 0
        
        diagnostic_report = next(
            entry["resource"] for entry in bundle["entry"]
            if entry["resource"]["resourceType"] == "DiagnosticReport"
        )
        assert "conclusion" in diagnostic_report
        
        media = next(
            entry["resource"] for entry in bundle["entry"]
            if entry["resource"]["resourceType"] == "Media"
        )
        assert "content" in media
