from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional


class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class PatientCreate(BaseModel):
    cpf: str
    name: str
    birth_date: str
    gender: str
    phone: Optional[str] = None
    email: Optional[str] = None


class PatientResponse(BaseModel):
    id: int
    cpf: str
    name: str
    birth_date: str
    gender: str
    phone: Optional[str]
    email: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class MedicalRecordCreate(BaseModel):
    patient_id: int
    document_type: str
    chief_complaint: str
    history_present_illness: str
    assessment: str
    treatment_plan: str


class MedicalRecordResponse(BaseModel):
    id: int
    patient_id: int
    document_type: str
    chief_complaint: str
    history_present_illness: str
    assessment: str
    treatment_plan: str
    created_at: datetime

    class Config:
        from_attributes = True


class ConsultationCreate(BaseModel):
    patient_id: int
    consultation_type: str
    scheduled_date: datetime


class ConsultationResponse(BaseModel):
    id: int
    patient_id: int
    consultation_type: str
    scheduled_date: datetime
    status: str
    notes: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class DiagnosticRequest(BaseModel):
    symptoms: str
    vital_signs: Optional[dict] = None
    patient_history: Optional[str] = None


class DiagnosisItem(BaseModel):
    condition: str
    confidence: float
    type: str


class DiagnosticResponse(BaseModel):
    diagnoses: List[DiagnosisItem]
    treatment_plan: str
    confidence_score: float
    compliance_notice: str
