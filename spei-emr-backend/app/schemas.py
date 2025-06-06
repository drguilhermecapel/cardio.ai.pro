from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class UserRole(str, Enum):
    PHYSICIAN = "physician"
    NURSE = "nurse"
    ADMIN = "admin"
    PATIENT = "patient"

class DocumentType(str, Enum):
    ANAMNESE = "anamnese"
    EVOLUCAO = "evolucao"
    ADMISSAO = "admissao"
    ALTA = "alta"
    CIRURGICO = "cirurgico"
    CONSULTA = "consulta"

class ConsultationType(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    CHAT = "chat"

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user_info: Dict[str, Any]

class UserCreateRequest(BaseModel):
    username: str
    email: str
    password: str
    full_name: str
    role: UserRole
    specialty: Optional[str] = None
    license_number: Optional[str] = None

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    full_name: str
    role: str
    specialty: Optional[str]
    license_number: Optional[str]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

class PatientCreateRequest(BaseModel):
    cpf: str
    name: str
    birth_date: datetime
    gender: str
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[Dict[str, Any]] = None
    emergency_contact: Optional[Dict[str, Any]] = None
    blood_type: Optional[str] = None
    allergies: Optional[List[str]] = None
    medical_history: Optional[List[str]] = None
    family_history: Optional[List[str]] = None
    social_history: Optional[Dict[str, Any]] = None
    current_medications: Optional[List[str]] = None

class PatientResponse(BaseModel):
    id: str
    cpf: str
    name: str
    birth_date: datetime
    gender: str
    phone: Optional[str]
    email: Optional[str]
    address: Optional[Dict[str, Any]]
    emergency_contact: Optional[Dict[str, Any]]
    blood_type: Optional[str]
    allergies: Optional[List[str]]
    medical_history: Optional[List[str]]
    family_history: Optional[List[str]]
    social_history: Optional[Dict[str, Any]]
    current_medications: Optional[List[str]]
    created_at: datetime

    class Config:
        from_attributes = True

class MedicalRecordCreateRequest(BaseModel):
    patient_id: str
    encounter_id: str
    document_type: DocumentType
    chief_complaint: str
    history_present_illness: Optional[str] = None
    review_of_systems: Optional[Dict[str, Any]] = None
    physical_examination: Optional[Dict[str, Any]] = None
    assessment: Optional[str] = None
    plan: Optional[str] = None

class MedicalRecordUpdateRequest(BaseModel):
    chief_complaint: Optional[str] = None
    history_present_illness: Optional[str] = None
    review_of_systems: Optional[Dict[str, Any]] = None
    physical_examination: Optional[Dict[str, Any]] = None
    assessment: Optional[str] = None
    plan: Optional[str] = None

class MedicalRecordResponse(BaseModel):
    id: str
    patient_id: str
    encounter_id: str
    document_type: str
    status: str
    chief_complaint: Optional[str]
    history_present_illness: Optional[str]
    review_of_systems: Optional[Dict[str, Any]]
    physical_examination: Optional[Dict[str, Any]]
    assessment: Optional[str]
    plan: Optional[str]
    extracted_symptoms: Optional[List[Dict[str, Any]]]
    extracted_diagnoses: Optional[List[Dict[str, Any]]]
    severity_score: Optional[float]
    quality_score: Optional[float]
    created_by: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class DiagnosisRequest(BaseModel):
    patient_id: str
    chief_complaint: str
    history_present_illness: str
    symptoms: List[Dict[str, Any]]
    vital_signs: Optional[Dict[str, float]] = None
    physical_examination: Optional[Dict[str, Any]] = None
    laboratory_results: Optional[Dict[str, Any]] = None
    imaging_results: Optional[Dict[str, Any]] = None

class DiagnosisHypothesis(BaseModel):
    icd_code: str
    icd_version: str
    description: str
    confidence: float
    evidence: List[Dict[str, Any]]
    differential_features: List[str]
    required_tests: List[str]
    treatment_implications: List[str]
    prognosis: str

class DiagnosisResponse(BaseModel):
    session_id: str
    primary_diagnoses: List[DiagnosisHypothesis]
    differential_diagnoses: List[DiagnosisHypothesis]
    ruled_out_diagnoses: List[str]
    ai_confidence_score: float
    processing_time: float
    recommendations: List[str]
    red_flags: List[str]

class SymptomAnalysisRequest(BaseModel):
    symptoms_text: str
    patient_demographics: Optional[Dict[str, Any]] = None

class SymptomAnalysisResponse(BaseModel):
    extracted_symptoms: List[Dict[str, Any]]
    severity_assessment: Dict[str, Any]
    temporal_analysis: Dict[str, Any]
    associated_conditions: List[str]

class DifferentialDiagnosisRequest(BaseModel):
    primary_symptoms: List[str]
    patient_age: int
    patient_gender: str
    medical_history: Optional[List[str]] = None

class DifferentialDiagnosisResponse(BaseModel):
    differential_list: List[DiagnosisHypothesis]
    decision_tree: Dict[str, Any]
    next_steps: List[str]

class ConsultationCreateRequest(BaseModel):
    patient_id: str
    consultation_type: ConsultationType
    scheduled_time: datetime
    chief_complaint: str

class ConsultationResponse(BaseModel):
    id: str
    patient_id: str
    physician_id: str
    consultation_type: str
    status: str
    scheduled_time: datetime
    actual_start_time: Optional[datetime]
    actual_end_time: Optional[datetime]
    room_id: Optional[str]
    chief_complaint: str
    consultation_notes: Optional[str]
    diagnosis: Optional[Dict[str, Any]]
    ai_insights: Optional[Dict[str, Any]]
    created_at: datetime

    class Config:
        from_attributes = True
