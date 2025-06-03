"""
Comprehensive regulatory compliance tests for ECG analysis system

This module provides tests to validate compliance with FDA, ANVISA, NMSA (China),
and European Union regulatory standards for medical device software.
"""

import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.exceptions import ValidationException
from app.services.regulatory_validation import RegulatoryValidationService
from app.services.ecg_service import ECGAnalysisService


class TestFDACompliance:
    """Test suite for FDA regulatory compliance."""

    @pytest.fixture
    def regulatory_service(self) -> RegulatoryValidationService:
        """Create regulatory validation service instance."""
        return RegulatoryValidationService()

    @pytest.mark.asyncio
    async def test_fda_510k_validation_requirements(self, regulatory_service: RegulatoryValidationService):
        """Test FDA 510(k) validation requirements."""
        analysis_result = {
            "analysis_id": "test-fda-001",
            "ai_confidence": 0.95,
            "predictions": {
                "rhythm": "normal_sinus_rhythm",
                "heart_rate": 72,
                "intervals": {
                    "pr_interval": 160,
                    "qrs_duration": 90,
                    "qt_interval": 400
                }
            },
            "clinical_findings": {
                "primary_diagnosis": "Normal ECG",
                "secondary_diagnoses": [],
                "clinical_urgency": "low"
            }
        }
        
        validation_result = await regulatory_service.validate_fda_compliance(analysis_result)
        
        assert validation_result["compliant"] is True
        assert "fda_510k_requirements" in validation_result
        assert validation_result["fda_510k_requirements"]["ai_transparency"] is True
        assert validation_result["fda_510k_requirements"]["clinical_validation"] is True
        assert validation_result["fda_510k_requirements"]["risk_management"] is True

    @pytest.mark.asyncio
    async def test_fda_software_as_medical_device_requirements(self, regulatory_service: RegulatoryValidationService):
        """Test FDA Software as Medical Device (SaMD) requirements."""
        risk_categories = ["low", "moderate", "high"]
        
        for risk_category in risk_categories:
            analysis_result = {
                "analysis_id": f"test-samd-{risk_category}",
                "risk_category": risk_category,
                "ai_confidence": 0.88,
                "clinical_urgency": risk_category
            }
            
            validation_result = await regulatory_service.validate_samd_requirements(analysis_result)
            
            assert validation_result["compliant"] is True
            assert validation_result["risk_category"] == risk_category
            assert "documentation_requirements" in validation_result
            assert "clinical_evaluation_requirements" in validation_result

    @pytest.mark.asyncio
    async def test_fda_ai_ml_guidance_compliance(self, regulatory_service: RegulatoryValidationService):
        """Test compliance with FDA AI/ML guidance."""
        ai_model_info = {
            "model_type": "hybrid_neural_network",
            "training_data_size": 100000,
            "validation_data_size": 20000,
            "test_data_size": 10000,
            "performance_metrics": {
                "sensitivity": 0.95,
                "specificity": 0.92,
                "ppv": 0.89,
                "npv": 0.97
            },
            "bias_assessment": {
                "demographic_bias": "assessed",
                "clinical_bias": "assessed",
                "technical_bias": "assessed"
            }
        }
        
        validation_result = await regulatory_service.validate_ai_ml_guidance(ai_model_info)
        
        assert validation_result["compliant"] is True
        assert validation_result["algorithm_transparency"] is True
        assert validation_result["performance_validation"] is True
        assert validation_result["bias_assessment"] is True


class TestANVISACompliance:
    """Test suite for ANVISA (Brazil) regulatory compliance."""

    @pytest.fixture
    def regulatory_service(self) -> RegulatoryValidationService:
        """Create regulatory validation service instance."""
        return RegulatoryValidationService()

    @pytest.mark.asyncio
    async def test_anvisa_software_medical_device_requirements(self, regulatory_service: RegulatoryValidationService):
        """Test ANVISA software medical device requirements."""
        device_info = {
            "device_class": "II",
            "intended_use": "ECG analysis and interpretation",
            "clinical_evidence": {
                "clinical_studies": 3,
                "patient_population": 5000,
                "clinical_sites": 10
            },
            "quality_management": {
                "iso_13485": True,
                "iso_14971": True,
                "iec_62304": True
            }
        }
        
        validation_result = await regulatory_service.validate_anvisa_compliance(device_info)
        
        assert validation_result["compliant"] is True
        assert validation_result["registration_requirements"]["class_ii_requirements"] is True
        assert validation_result["quality_standards"]["iso_compliance"] is True

    @pytest.mark.asyncio
    async def test_anvisa_ai_specific_requirements(self, regulatory_service: RegulatoryValidationService):
        """Test ANVISA AI-specific requirements."""
        ai_system_info = {
            "ai_type": "machine_learning",
            "learning_type": "supervised",
            "data_governance": {
                "data_quality": "validated",
                "data_security": "encrypted",
                "data_privacy": "anonymized"
            },
            "algorithm_validation": {
                "internal_validation": True,
                "external_validation": True,
                "clinical_validation": True
            }
        }
        
        validation_result = await regulatory_service.validate_anvisa_ai_requirements(ai_system_info)
        
        assert validation_result["compliant"] is True
        assert validation_result["ai_transparency"] is True
        assert validation_result["data_governance"] is True


class TestNMSACompliance:
    """Test suite for NMSA (China) regulatory compliance."""

    @pytest.fixture
    def regulatory_service(self) -> RegulatoryValidationService:
        """Create regulatory validation service instance."""
        return RegulatoryValidationService()

    @pytest.mark.asyncio
    async def test_nmsa_medical_device_registration(self, regulatory_service: RegulatoryValidationService):
        """Test NMSA medical device registration requirements."""
        device_registration = {
            "device_category": "Class_II",
            "registration_type": "domestic",
            "clinical_trial_data": {
                "trial_sites": 5,
                "patient_enrollment": 1000,
                "primary_endpoints_met": True
            },
            "manufacturing_quality": {
                "gmp_compliance": True,
                "quality_system": "ISO_13485"
            }
        }
        
        validation_result = await regulatory_service.validate_nmsa_compliance(device_registration)
        
        assert validation_result["compliant"] is True
        assert validation_result["registration_pathway"]["class_ii_pathway"] is True
        assert validation_result["clinical_requirements"]["adequate_clinical_data"] is True

    @pytest.mark.asyncio
    async def test_nmsa_ai_algorithm_requirements(self, regulatory_service: RegulatoryValidationService):
        """Test NMSA AI algorithm specific requirements."""
        ai_algorithm_info = {
            "algorithm_type": "deep_learning",
            "training_methodology": "federated_learning",
            "data_requirements": {
                "chinese_population_data": True,
                "data_representativeness": "validated",
                "data_size": 50000
            },
            "performance_validation": {
                "chinese_clinical_sites": 8,
                "validation_population": "chinese_patients",
                "performance_benchmarks_met": True
            }
        }
        
        validation_result = await regulatory_service.validate_nmsa_ai_requirements(ai_algorithm_info)
        
        assert validation_result["compliant"] is True
        assert validation_result["population_specific_validation"] is True
        assert validation_result["local_clinical_evidence"] is True


class TestEUMDRCompliance:
    """Test suite for EU MDR (Medical Device Regulation) compliance."""

    @pytest.fixture
    def regulatory_service(self) -> RegulatoryValidationService:
        """Create regulatory validation service instance."""
        return RegulatoryValidationService()

    @pytest.mark.asyncio
    async def test_eu_mdr_classification_requirements(self, regulatory_service: RegulatoryValidationService):
        """Test EU MDR classification requirements."""
        device_classification = {
            "device_class": "IIa",
            "classification_rules": ["Rule_11", "Rule_12"],
            "intended_purpose": "diagnosis_support",
            "risk_classification": "medium_risk",
            "conformity_assessment": {
                "notified_body_required": True,
                "ce_marking": True,
                "technical_documentation": "complete"
            }
        }
        
        validation_result = await regulatory_service.validate_eu_mdr_compliance(device_classification)
        
        assert validation_result["compliant"] is True
        assert validation_result["classification"]["class_iia_requirements"] is True
        assert validation_result["conformity_assessment"]["notified_body_involvement"] is True

    @pytest.mark.asyncio
    async def test_eu_mdr_software_requirements(self, regulatory_service: RegulatoryValidationService):
        """Test EU MDR software-specific requirements."""
        software_requirements = {
            "software_lifecycle": "IEC_62304",
            "cybersecurity": {
                "security_by_design": True,
                "vulnerability_management": True,
                "data_protection": "GDPR_compliant"
            },
            "clinical_evaluation": {
                "clinical_evidence": "sufficient",
                "post_market_surveillance": "planned",
                "clinical_follow_up": "ongoing"
            },
            "unique_device_identification": {
                "udi_assigned": True,
                "udi_database_registration": True
            }
        }
        
        validation_result = await regulatory_service.validate_eu_mdr_software_requirements(software_requirements)
        
        assert validation_result["compliant"] is True
        assert validation_result["software_lifecycle_compliance"] is True
        assert validation_result["cybersecurity_requirements"] is True
        assert validation_result["gdpr_compliance"] is True

    @pytest.mark.asyncio
    async def test_eu_ai_act_compliance(self, regulatory_service: RegulatoryValidationService):
        """Test EU AI Act compliance for high-risk AI systems."""
        ai_system_info = {
            "ai_system_category": "high_risk",
            "use_case": "medical_diagnosis",
            "risk_management": {
                "risk_assessment": "comprehensive",
                "risk_mitigation": "implemented",
                "ongoing_monitoring": True
            },
            "data_governance": {
                "training_data_quality": "validated",
                "bias_monitoring": "continuous",
                "data_representativeness": "ensured"
            },
            "transparency": {
                "algorithm_explainability": True,
                "user_information": "comprehensive",
                "decision_transparency": True
            },
            "human_oversight": {
                "human_in_the_loop": True,
                "override_capability": True,
                "competent_supervision": True
            }
        }
        
        validation_result = await regulatory_service.validate_eu_ai_act_compliance(ai_system_info)
        
        assert validation_result["compliant"] is True
        assert validation_result["high_risk_requirements"]["risk_management"] is True
        assert validation_result["high_risk_requirements"]["data_governance"] is True
        assert validation_result["high_risk_requirements"]["transparency"] is True
        assert validation_result["high_risk_requirements"]["human_oversight"] is True


class TestCrossRegulatoryCompliance:
    """Test suite for cross-regulatory compliance validation."""

    @pytest.fixture
    def regulatory_service(self) -> RegulatoryValidationService:
        """Create regulatory validation service instance."""
        return RegulatoryValidationService()

    @pytest.mark.asyncio
    async def test_multi_jurisdiction_compliance(self, regulatory_service: RegulatoryValidationService):
        """Test compliance across multiple jurisdictions."""
        system_info = {
            "target_markets": ["US", "EU", "Brazil", "China"],
            "device_classification": {
                "fda_class": "II",
                "eu_class": "IIa",
                "anvisa_class": "II",
                "nmsa_class": "II"
            },
            "clinical_evidence": {
                "us_clinical_data": True,
                "eu_clinical_data": True,
                "brazil_clinical_data": True,
                "china_clinical_data": True
            }
        }
        
        validation_result = await regulatory_service.validate_multi_jurisdiction_compliance(system_info)
        
        assert validation_result["overall_compliant"] is True
        assert validation_result["fda_compliant"] is True
        assert validation_result["eu_mdr_compliant"] is True
        assert validation_result["anvisa_compliant"] is True
        assert validation_result["nmsa_compliant"] is True

    @pytest.mark.asyncio
    async def test_harmonized_standards_compliance(self, regulatory_service: RegulatoryValidationService):
        """Test compliance with harmonized international standards."""
        standards_compliance = {
            "iso_13485": True,  # Quality management systems
            "iso_14971": True,  # Risk management
            "iec_62304": True,  # Medical device software lifecycle
            "iso_27001": True,  # Information security management
            "iec_62366": True,  # Usability engineering
            "iso_15189": True,  # Medical laboratories quality
        }
        
        validation_result = await regulatory_service.validate_harmonized_standards(standards_compliance)
        
        assert validation_result["compliant"] is True
        assert validation_result["quality_management"] is True
        assert validation_result["risk_management"] is True
        assert validation_result["software_lifecycle"] is True
        assert validation_result["information_security"] is True

    @pytest.mark.asyncio
    async def test_post_market_surveillance_requirements(self, regulatory_service: RegulatoryValidationService):
        """Test post-market surveillance requirements across jurisdictions."""
        surveillance_plan = {
            "adverse_event_reporting": {
                "fda_maude": True,
                "eu_eudamed": True,
                "anvisa_notivisa": True,
                "nmsa_nifdc": True
            },
            "performance_monitoring": {
                "real_world_evidence": True,
                "algorithm_performance_tracking": True,
                "bias_monitoring": True
            },
            "corrective_actions": {
                "field_safety_notices": "planned",
                "software_updates": "controlled",
                "recall_procedures": "documented"
            }
        }
        
        validation_result = await regulatory_service.validate_post_market_surveillance(surveillance_plan)
        
        assert validation_result["compliant"] is True
        assert validation_result["adverse_event_systems"] is True
        assert validation_result["performance_monitoring"] is True
        assert validation_result["corrective_action_procedures"] is True


class TestRegulatoryDocumentation:
    """Test suite for regulatory documentation requirements."""

    @pytest.fixture
    def regulatory_service(self) -> RegulatoryValidationService:
        """Create regulatory validation service instance."""
        return RegulatoryValidationService()

    @pytest.mark.asyncio
    async def test_technical_documentation_completeness(self, regulatory_service: RegulatoryValidationService):
        """Test technical documentation completeness."""
        documentation = {
            "device_description": "complete",
            "intended_use": "defined",
            "clinical_evidence": "sufficient",
            "risk_analysis": "comprehensive",
            "software_documentation": "iec_62304_compliant",
            "usability_engineering": "iec_62366_compliant",
            "cybersecurity_documentation": "complete"
        }
        
        validation_result = await regulatory_service.validate_technical_documentation(documentation)
        
        assert validation_result["complete"] is True
        assert validation_result["regulatory_ready"] is True

    @pytest.mark.asyncio
    async def test_clinical_evaluation_documentation(self, regulatory_service: RegulatoryValidationService):
        """Test clinical evaluation documentation."""
        clinical_evaluation = {
            "clinical_evidence_plan": "approved",
            "literature_review": "systematic",
            "clinical_studies": {
                "study_count": 3,
                "patient_count": 2500,
                "primary_endpoints": "met",
                "safety_profile": "acceptable"
            },
            "post_market_clinical_follow_up": "planned",
            "clinical_evaluation_report": "complete"
        }
        
        validation_result = await regulatory_service.validate_clinical_evaluation(clinical_evaluation)
        
        assert validation_result["adequate"] is True
        assert validation_result["evidence_quality"] == "high"
        assert validation_result["regulatory_acceptable"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
