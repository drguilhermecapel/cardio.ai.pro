from sqlalchemy import Column, String, DateTime, JSON, Float, Integer, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

from .database import Base

class User(Base):
    __tablename__ = 'users'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    role = Column(String, default='physician')  # physician, nurse, admin, patient
    specialty = Column(String)
    license_number = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Patient(Base):
    __tablename__ = 'patients'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    cpf = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    birth_date = Column(DateTime)
    gender = Column(String)
    phone = Column(String)
    email = Column(String)
    address = Column(JSON)
    emergency_contact = Column(JSON)
    
    blood_type = Column(String)
    allergies = Column(JSON)
    medical_history = Column(JSON)
    family_history = Column(JSON)
    social_history = Column(JSON)
    current_medications = Column(JSON)
    
    created_by = Column(String, ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    medical_records = relationship("MedicalRecord", back_populates="patient")
    consultations = relationship("TelemedicineConsultation", back_populates="patient")

class MedicalRecord(Base):
    __tablename__ = 'medical_records'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String, ForeignKey('patients.id'), nullable=False, index=True)
    encounter_id = Column(String, nullable=False, index=True)
    document_type = Column(String, nullable=False)  # anamnese, evolucao, alta, etc.
    status = Column(String, default='draft')  # draft, final, amended
    
    chief_complaint = Column(Text)
    history_present_illness = Column(Text)
    review_of_systems = Column(JSON)
    physical_examination = Column(JSON)
    assessment = Column(Text)
    plan = Column(Text)
    
    extracted_symptoms = Column(JSON)
    extracted_diagnoses = Column(JSON)
    extracted_medications = Column(JSON)
    severity_score = Column(Float)
    urgency_level = Column(Integer)
    
    quality_score = Column(Float)
    completeness_score = Column(Float)
    
    created_by = Column(String, ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    signed_at = Column(DateTime)
    signed_by = Column(String, ForeignKey('users.id'))
    
    version = Column(Integer, default=1)
    parent_version = Column(String)
    
    fhir_resource_id = Column(String)
    fhir_resource_version = Column(String)
    
    patient = relationship("Patient", back_populates="medical_records")
    diagnoses = relationship("DiagnosticSession", back_populates="medical_record")

class DiagnosticSession(Base):
    __tablename__ = 'diagnostic_sessions'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String, ForeignKey('patients.id'), nullable=False, index=True)
    medical_record_id = Column(String, ForeignKey('medical_records.id'))
    clinician_id = Column(String, ForeignKey('users.id'), nullable=False)
    
    chief_complaint = Column(Text)
    history_present_illness = Column(Text)
    clinical_presentation = Column(JSON)
    
    patient_demographics = Column(JSON)
    medical_history = Column(JSON)
    family_history = Column(JSON)
    social_history = Column(JSON)
    
    vital_signs = Column(JSON)
    physical_examination = Column(JSON)
    
    laboratory_results = Column(JSON)
    imaging_results = Column(JSON)
    other_tests = Column(JSON)
    
    ai_diagnoses = Column(JSON)
    differential_diagnoses = Column(JSON)
    ruled_out_diagnoses = Column(JSON)
    
    ai_confidence_score = Column(Float)
    ai_model_version = Column(String)
    processing_time = Column(Float)
    
    clinician_selected_diagnosis = Column(JSON)
    clinician_feedback = Column(JSON)
    diagnostic_accuracy = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    finalized_at = Column(DateTime)
    
    medical_record = relationship("MedicalRecord", back_populates="diagnoses")

class TelemedicineConsultation(Base):
    __tablename__ = 'telemedicine_consultations'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String, ForeignKey('patients.id'), nullable=False, index=True)
    physician_id = Column(String, ForeignKey('users.id'), nullable=False, index=True)
    
    consultation_type = Column(String, nullable=False)  # video, audio, chat
    status = Column(String, default='scheduled')  # scheduled, in_progress, completed, cancelled
    
    scheduled_time = Column(DateTime)
    actual_start_time = Column(DateTime)
    actual_end_time = Column(DateTime)
    duration_minutes = Column(Integer)
    
    room_id = Column(String, unique=True)
    access_token_patient = Column(Text)
    access_token_physician = Column(Text)
    
    chief_complaint = Column(Text)
    consultation_notes = Column(Text)
    diagnosis = Column(JSON)
    prescriptions = Column(JSON)
    
    ai_insights = Column(JSON)
    symptom_analysis = Column(JSON)
    
    recording_enabled = Column(Boolean, default=True)
    recording_url = Column(String)
    transcript_url = Column(String)
    
    connected_devices = Column(JSON)
    vital_signs_data = Column(JSON)
    
    patient_rating = Column(Integer)
    patient_feedback = Column(Text)
    
    consent_recorded = Column(Boolean, default=False)
    identity_verified = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    patient = relationship("Patient", back_populates="consultations")

class CIDMapping(Base):
    __tablename__ = 'cid_mappings'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    cid10_code = Column(String, index=True)
    cid11_code = Column(String, index=True)
    description_pt = Column(Text)
    description_en = Column(Text)
    
    typical_symptoms = Column(JSON)
    physical_signs = Column(JSON)
    diagnostic_criteria = Column(JSON)
    differential_diagnoses = Column(JSON)
    
    prevalence = Column(Float)
    incidence = Column(Float)
    risk_factors = Column(JSON)
    age_distribution = Column(JSON)
    gender_distribution = Column(JSON)
    
    clinical_embedding = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class AuditLog(Base):
    __tablename__ = 'audit_logs'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.id'))
    action = Column(String, nullable=False)
    resource_type = Column(String)
    resource_id = Column(String)
    details = Column(JSON)
    ip_address = Column(String)
    user_agent = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
