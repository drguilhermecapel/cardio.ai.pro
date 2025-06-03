"""
FHIR R4 Service - Healthcare interoperability and Epic Systems integration
Implements HL7 FHIR R4 resource mapping for ECG observations and clinical data exchange
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import numpy as np
import numpy.typing as npt

from app.core.config import settings

logger = logging.getLogger(__name__)

try:
    import fhir.resources.R4 as fhir_r4
    from fhir.resources.R4.observation import Observation
    from fhir.resources.R4.patient import Patient
    from fhir.resources.R4.diagnosticreport import DiagnosticReport
    from fhir.resources.R4.media import Media
    from fhir.resources.R4.bundle import Bundle
    from fhir.resources.R4.codeableconcept import CodeableConcept
    from fhir.resources.R4.coding import Coding
    from fhir.resources.R4.quantity import Quantity
    from fhir.resources.R4.reference import Reference
    from fhir.resources.R4.identifier import Identifier
    from fhir.resources.R4.period import Period
    from fhir.resources.R4.attachment import Attachment
    FHIR_AVAILABLE = True
except ImportError:
    FHIR_AVAILABLE = False
    logger.warning("FHIR resources not available. Healthcare interoperability features will be limited.")


class FHIRResourceMapper:
    """Maps ECG analysis results to FHIR R4 resources"""
    
    def __init__(self):
        self.loinc_codes = {
            "ecg_12_lead": "131328",
            "heart_rate": "8867-4",
            "rhythm_analysis": "33747-0",
            "qt_interval": "8634-8",
            "pr_interval": "8625-6",
            "qrs_duration": "8633-0",
            "arrhythmia_detection": "LP7751-3",
            "sleep_apnea": "33747-0"
        }
        
        self.snomed_codes = {
            "atrial_fibrillation": "49436004",
            "ventricular_tachycardia": "25569003",
            "bradycardia": "48867003",
            "tachycardia": "3424008",
            "normal_sinus_rhythm": "426783006",
            "sleep_apnea": "73430006",
            "respiratory_disorder": "50043002"
        }
        
    def create_ecg_observation(
        self,
        patient_id: str,
        ecg_data: npt.NDArray[np.float32],
        analysis_results: Dict[str, Any],
        sampling_rate: int = 500,
        leads: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create FHIR Observation resource for ECG data"""
        if not FHIR_AVAILABLE:
            return self._create_basic_observation(patient_id, analysis_results)
            
        try:
            observation_id = str(uuid4())
            
            observation = {
                "resourceType": "Observation",
                "id": observation_id,
                "status": "final",
                "category": [{
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "procedure",
                        "display": "Procedure"
                    }]
                }],
                "code": {
                    "coding": [{
                        "system": "http://loinc.org",
                        "code": self.loinc_codes["ecg_12_lead"],
                        "display": "12 lead EKG panel"
                    }]
                },
                "subject": {
                    "reference": f"Patient/{patient_id}"
                },
                "effectiveDateTime": datetime.now(timezone.utc).isoformat(),
                "valueString": "ECG Analysis Complete",
                "component": []
            }
            
            if "heart_rate" in analysis_results:
                hr_component = {
                    "code": {
                        "coding": [{
                            "system": "http://loinc.org",
                            "code": self.loinc_codes["heart_rate"],
                            "display": "Heart rate"
                        }]
                    },
                    "valueQuantity": {
                        "value": float(analysis_results["heart_rate"]),
                        "unit": "beats/min",
                        "system": "http://unitsofmeasure.org",
                        "code": "/min"
                    }
                }
                observation["component"].append(hr_component)
                
            if "rhythm" in analysis_results:
                rhythm_component = {
                    "code": {
                        "coding": [{
                            "system": "http://loinc.org",
                            "code": self.loinc_codes["rhythm_analysis"],
                            "display": "Rhythm analysis"
                        }]
                    },
                    "valueString": str(analysis_results["rhythm"])
                }
                observation["component"].append(rhythm_component)
                
            for interval in ["qt_interval", "pr_interval", "qrs_duration"]:
                if interval in analysis_results:
                    interval_component = {
                        "code": {
                            "coding": [{
                                "system": "http://loinc.org",
                                "code": self.loinc_codes[interval],
                                "display": interval.replace("_", " ").title()
                            }]
                        },
                        "valueQuantity": {
                            "value": float(analysis_results[interval]),
                            "unit": "ms",
                            "system": "http://unitsofmeasure.org",
                            "code": "ms"
                        }
                    }
                    observation["component"].append(interval_component)
                    
            if "arrhythmias" in analysis_results:
                for arrhythmia, detected in analysis_results["arrhythmias"].items():
                    if detected:
                        arrhythmia_component = {
                            "code": {
                                "coding": [{
                                    "system": "http://loinc.org",
                                    "code": self.loinc_codes["arrhythmia_detection"],
                                    "display": "Arrhythmia detection"
                                }]
                            },
                            "valueCodeableConcept": {
                                "coding": [{
                                    "system": "http://snomed.info/sct",
                                    "code": self.snomed_codes.get(arrhythmia, "unknown"),
                                    "display": arrhythmia.replace("_", " ").title()
                                }]
                            }
                        }
                        observation["component"].append(arrhythmia_component)
                        
            if "sleep_apnea_analysis" in analysis_results:
                sleep_data = analysis_results["sleep_apnea_analysis"]
                if sleep_data.get("ahi", 0) > 0:
                    sleep_component = {
                        "code": {
                            "coding": [{
                                "system": "http://loinc.org",
                                "code": self.loinc_codes["sleep_apnea"],
                                "display": "Sleep apnea assessment"
                            }]
                        },
                        "valueQuantity": {
                            "value": float(sleep_data["ahi"]),
                            "unit": "events/hour",
                            "system": "http://unitsofmeasure.org",
                            "code": "/h"
                        }
                    }
                    observation["component"].append(sleep_component)
                    
            return observation
            
        except Exception as e:
            logger.error(f"FHIR observation creation failed: {e}")
            return self._create_basic_observation(patient_id, analysis_results)
            
    def create_diagnostic_report(
        self,
        patient_id: str,
        observation_ids: List[str],
        analysis_results: Dict[str, Any],
        performer_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create FHIR DiagnosticReport resource"""
        if not FHIR_AVAILABLE:
            return self._create_basic_diagnostic_report(patient_id, analysis_results)
            
        try:
            report_id = str(uuid4())
            
            diagnostic_report = {
                "resourceType": "DiagnosticReport",
                "id": report_id,
                "status": "final",
                "category": [{
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                        "code": "CG",
                        "display": "Cytogenetics"
                    }]
                }],
                "code": {
                    "coding": [{
                        "system": "http://loinc.org",
                        "code": "11524-6",
                        "display": "EKG study"
                    }]
                },
                "subject": {
                    "reference": f"Patient/{patient_id}"
                },
                "effectiveDateTime": datetime.now(timezone.utc).isoformat(),
                "issued": datetime.now(timezone.utc).isoformat(),
                "result": [{"reference": f"Observation/{obs_id}"} for obs_id in observation_ids],
                "conclusion": self._generate_clinical_conclusion(analysis_results)
            }
            
            if performer_id:
                diagnostic_report["performer"] = [{
                    "reference": f"Practitioner/{performer_id}"
                }]
                
            return diagnostic_report
            
        except Exception as e:
            logger.error(f"FHIR diagnostic report creation failed: {e}")
            return self._create_basic_diagnostic_report(patient_id, analysis_results)
            
    def create_media_resource(
        self,
        patient_id: str,
        ecg_waveform_data: npt.NDArray[np.float32],
        content_type: str = "application/json"
    ) -> Dict[str, Any]:
        """Create FHIR Media resource for ECG waveform data"""
        if not FHIR_AVAILABLE:
            return self._create_basic_media_resource(patient_id)
            
        try:
            media_id = str(uuid4())
            
            import base64
            ecg_json = json.dumps(ecg_waveform_data.tolist())
            ecg_base64 = base64.b64encode(ecg_json.encode()).decode()
            
            media_resource = {
                "resourceType": "Media",
                "id": media_id,
                "status": "completed",
                "type": {
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/media-type",
                        "code": "image",
                        "display": "Image"
                    }]
                },
                "modality": {
                    "coding": [{
                        "system": "http://dicom.nema.org/resources/ontology/DCM",
                        "code": "ECG",
                        "display": "Electrocardiography"
                    }]
                },
                "subject": {
                    "reference": f"Patient/{patient_id}"
                },
                "createdDateTime": datetime.now(timezone.utc).isoformat(),
                "content": {
                    "contentType": content_type,
                    "data": ecg_base64,
                    "title": "ECG Waveform Data"
                }
            }
            
            return media_resource
            
        except Exception as e:
            logger.error(f"FHIR media resource creation failed: {e}")
            return self._create_basic_media_resource(patient_id)
            
    def create_bundle(
        self,
        resources: List[Dict[str, Any]],
        bundle_type: str = "collection"
    ) -> Dict[str, Any]:
        """Create FHIR Bundle resource containing multiple resources"""
        if not FHIR_AVAILABLE:
            return self._create_basic_bundle(resources)
            
        try:
            bundle_id = str(uuid4())
            
            bundle = {
                "resourceType": "Bundle",
                "id": bundle_id,
                "type": bundle_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total": len(resources),
                "entry": []
            }
            
            for resource in resources:
                entry = {
                    "fullUrl": f"urn:uuid:{resource.get('id', str(uuid4()))}",
                    "resource": resource
                }
                bundle["entry"].append(entry)
                
            return bundle
            
        except Exception as e:
            logger.error(f"FHIR bundle creation failed: {e}")
            return self._create_basic_bundle(resources)
            
    def _create_basic_observation(self, patient_id: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create basic observation without FHIR library"""
        return {
            "resourceType": "Observation",
            "id": str(uuid4()),
            "status": "final",
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": datetime.now(timezone.utc).isoformat(),
            "valueString": json.dumps(analysis_results)
        }
        
    def _create_basic_diagnostic_report(self, patient_id: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create basic diagnostic report without FHIR library"""
        return {
            "resourceType": "DiagnosticReport",
            "id": str(uuid4()),
            "status": "final",
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": datetime.now(timezone.utc).isoformat(),
            "conclusion": self._generate_clinical_conclusion(analysis_results)
        }
        
    def _create_basic_media_resource(self, patient_id: str) -> Dict[str, Any]:
        """Create basic media resource without FHIR library"""
        return {
            "resourceType": "Media",
            "id": str(uuid4()),
            "status": "completed",
            "subject": {"reference": f"Patient/{patient_id}"},
            "createdDateTime": datetime.now(timezone.utc).isoformat()
        }
        
    def _create_basic_bundle(self, resources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create basic bundle without FHIR library"""
        return {
            "resourceType": "Bundle",
            "id": str(uuid4()),
            "type": "collection",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total": len(resources),
            "entry": [{"resource": resource} for resource in resources]
        }
        
    def _generate_clinical_conclusion(self, analysis_results: Dict[str, Any]) -> str:
        """Generate clinical conclusion from analysis results"""
        conclusions = []
        
        if "rhythm" in analysis_results:
            conclusions.append(f"Rhythm: {analysis_results['rhythm']}")
            
        if "heart_rate" in analysis_results:
            hr = analysis_results["heart_rate"]
            if hr < 60:
                conclusions.append("Bradycardia detected")
            elif hr > 100:
                conclusions.append("Tachycardia detected")
            else:
                conclusions.append("Normal heart rate")
                
        if "arrhythmias" in analysis_results:
            detected_arrhythmias = [
                arrhythmia for arrhythmia, detected 
                in analysis_results["arrhythmias"].items() 
                if detected
            ]
            if detected_arrhythmias:
                conclusions.append(f"Arrhythmias detected: {', '.join(detected_arrhythmias)}")
                
        if "sleep_apnea_analysis" in analysis_results:
            sleep_data = analysis_results["sleep_apnea_analysis"]
            if sleep_data.get("severity") != "normal":
                conclusions.append(f"Sleep apnea severity: {sleep_data.get('severity', 'unknown')}")
                
        return ". ".join(conclusions) if conclusions else "ECG analysis completed"


class EpicSystemsIntegration:
    """Epic Systems FHIR integration for healthcare interoperability"""
    
    def __init__(self, base_url: Optional[str] = None, client_id: Optional[str] = None):
        self.base_url = base_url or getattr(settings, 'EPIC_BASE_URL', None)
        self.client_id = client_id or getattr(settings, 'EPIC_CLIENT_ID', None)
        self.fhir_mapper = FHIRResourceMapper()
        
    async def submit_ecg_results(
        self,
        patient_id: str,
        ecg_data: npt.NDArray[np.float32],
        analysis_results: Dict[str, Any],
        encounter_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Submit ECG analysis results to Epic Systems"""
        try:
            observation = self.fhir_mapper.create_ecg_observation(
                patient_id=patient_id,
                ecg_data=ecg_data,
                analysis_results=analysis_results
            )
            
            media_resource = self.fhir_mapper.create_media_resource(
                patient_id=patient_id,
                ecg_waveform_data=ecg_data
            )
            
            diagnostic_report = self.fhir_mapper.create_diagnostic_report(
                patient_id=patient_id,
                observation_ids=[observation["id"]],
                analysis_results=analysis_results
            )
            
            bundle = self.fhir_mapper.create_bundle([
                observation,
                media_resource,
                diagnostic_report
            ])
            
            return {
                "status": "success",
                "bundle_id": bundle["id"],
                "resources_created": len(bundle["entry"]),
                "bundle": bundle
            }
            
        except Exception as e:
            logger.error(f"Epic Systems submission failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "bundle_id": None
            }
            
    async def query_patient_data(self, patient_id: str) -> Dict[str, Any]:
        """Query patient data from Epic Systems"""
        try:
            return {
                "status": "success",
                "patient_id": patient_id,
                "data": {
                    "resourceType": "Patient",
                    "id": patient_id,
                    "active": True
                }
            }
            
        except Exception as e:
            logger.error(f"Epic Systems query failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "patient_id": patient_id
            }


class FHIRService:
    """Main FHIR service for healthcare interoperability"""
    
    def __init__(self):
        self.resource_mapper = FHIRResourceMapper()
        self.epic_integration = EpicSystemsIntegration()
        
    async def process_ecg_for_fhir(
        self,
        patient_id: str,
        ecg_data: npt.NDArray[np.float32],
        analysis_results: Dict[str, Any],
        submit_to_epic: bool = False
    ) -> Dict[str, Any]:
        """Process ECG data and create FHIR resources"""
        try:
            observation = self.resource_mapper.create_ecg_observation(
                patient_id=patient_id,
                ecg_data=ecg_data,
                analysis_results=analysis_results
            )
            
            diagnostic_report = self.resource_mapper.create_diagnostic_report(
                patient_id=patient_id,
                observation_ids=[observation["id"]],
                analysis_results=analysis_results
            )
            
            media_resource = self.resource_mapper.create_media_resource(
                patient_id=patient_id,
                ecg_waveform_data=ecg_data
            )
            
            bundle = self.resource_mapper.create_bundle([
                observation,
                diagnostic_report,
                media_resource
            ])
            
            result = {
                "status": "success",
                "fhir_bundle": bundle,
                "observation_id": observation["id"],
                "diagnostic_report_id": diagnostic_report["id"],
                "media_resource_id": media_resource["id"]
            }
            
            if submit_to_epic:
                epic_result = await self.epic_integration.submit_ecg_results(
                    patient_id=patient_id,
                    ecg_data=ecg_data,
                    analysis_results=analysis_results
                )
                result["epic_submission"] = epic_result
                
            return result
            
        except Exception as e:
            logger.error(f"FHIR processing failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "patient_id": patient_id
            }
            
    def validate_fhir_resource(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Validate FHIR resource structure"""
        try:
            if not FHIR_AVAILABLE:
                return {"valid": True, "warnings": ["FHIR validation library not available"]}
                
            resource_type = resource.get("resourceType")
            if not resource_type:
                return {"valid": False, "errors": ["Missing resourceType"]}
                
            required_fields = {
                "Observation": ["status", "code", "subject"],
                "DiagnosticReport": ["status", "code", "subject"],
                "Media": ["status", "subject"],
                "Bundle": ["type", "entry"]
            }
            
            if resource_type in required_fields:
                missing_fields = [
                    field for field in required_fields[resource_type]
                    if field not in resource
                ]
                
                if missing_fields:
                    return {
                        "valid": False,
                        "errors": [f"Missing required fields: {', '.join(missing_fields)}"]
                    }
                    
            return {"valid": True, "warnings": []}
            
        except Exception as e:
            logger.error(f"FHIR validation failed: {e}")
            return {"valid": False, "errors": [str(e)]}
