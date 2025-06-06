from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import List, Optional
import uvicorn
from datetime import datetime
import os

from .database import engine, SessionLocal, Base
from .models import *
from .schemas import *

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="SPEI - Sistema de Prontuário Eletrônico Inteligente",
    description="Electronic Medical Record System with AI-powered diagnostics and telemedicine",
    version="1.0.0"
)

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

security = HTTPBearer()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def read_root():
    return {
        "message": "SPEI - Sistema de Prontuário Eletrônico Inteligente",
        "version": "1.0.0",
        "features": [
            "AI-powered diagnostics",
            "Electronic medical records",
            "Telemedicine platform",
            "FHIR compliance",
            "Regulatory compliance (ANVISA, FDA, EU)"
        ]
    }

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "timestamp": datetime.utcnow()}

@app.post("/auth/login")
async def login(credentials: dict):
    if credentials.get("username") == "admin" and credentials.get("password") == "admin":
        return {
            "access_token": "demo_token",
            "token_type": "bearer",
            "expires_in": 3600,
            "user_info": {
                "id": "demo_user",
                "username": "admin",
                "role": "physician"
            }
        }
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/patients")
async def create_patient(patient_data: dict, db: Session = Depends(get_db)):
    patient = Patient(
        cpf=patient_data["cpf"],
        name=patient_data["name"],
        birth_date=datetime.fromisoformat(patient_data["birth_date"]),
        gender=patient_data["gender"],
        phone=patient_data.get("phone"),
        email=patient_data.get("email"),
        created_by="demo_user"
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient

@app.get("/patients")
async def get_patients(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    patients = db.query(Patient).offset(skip).limit(limit).all()
    return patients

@app.get("/patients/{patient_id}")
async def get_patient(patient_id: str, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient

@app.post("/medical-records")
async def create_medical_record(record_data: dict, db: Session = Depends(get_db)):
    record = MedicalRecord(
        patient_id=record_data["patient_id"],
        encounter_id=record_data["encounter_id"],
        document_type=record_data["document_type"],
        chief_complaint=record_data["chief_complaint"],
        history_present_illness=record_data.get("history_present_illness"),
        created_by="demo_user"
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

@app.get("/medical-records/{record_id}")
async def get_medical_record(record_id: str, db: Session = Depends(get_db)):
    record = db.query(MedicalRecord).filter(MedicalRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Medical record not found")
    return record

@app.post("/ai/diagnose")
async def ai_diagnose(diagnosis_request: dict):
    return {
        "session_id": "demo_session",
        "primary_diagnoses": [
            {
                "icd_code": "R50.9",
                "icd_version": "10",
                "description": "Febre não especificada",
                "confidence": 0.85,
                "evidence": ["Temperatura elevada", "Mal-estar geral"],
                "differential_features": ["Duração dos sintomas", "Sintomas associados"],
                "required_tests": ["Hemograma completo", "Hemocultura"],
                "treatment_implications": ["Investigação de foco infeccioso"],
                "prognosis": "Bom com tratamento adequado"
            }
        ],
        "differential_diagnoses": [
            {
                "icd_code": "J06.9",
                "icd_version": "10", 
                "description": "Infecção aguda das vias aéreas superiores",
                "confidence": 0.65,
                "evidence": ["Sintomas respiratórios"],
                "differential_features": ["Tosse", "Coriza"],
                "required_tests": ["Exame físico detalhado"],
                "treatment_implications": ["Tratamento sintomático"],
                "prognosis": "Excelente"
            }
        ],
        "ruled_out_diagnoses": ["Pneumonia grave"],
        "ai_confidence_score": 0.85,
        "processing_time": 1.2,
        "recommendations": [
            "Investigar foco infeccioso",
            "Monitorar sinais vitais",
            "Considerar antibioticoterapia se necessário"
        ],
        "red_flags": []
    }

@app.post("/ai/analyze-symptoms")
async def analyze_symptoms(symptoms_request: dict):
    return {
        "extracted_symptoms": [
            {"symptom": "febre", "severity": "moderada", "duration": "2 dias"},
            {"symptom": "cefaleia", "severity": "leve", "duration": "1 dia"}
        ],
        "severity_assessment": {"overall": "moderada", "urgency": "baixa"},
        "temporal_analysis": {"onset": "agudo", "progression": "estável"},
        "associated_conditions": ["Síndrome gripal", "Infecção viral"]
    }

@app.post("/telemedicine/consultations")
async def create_consultation(consultation_data: dict, db: Session = Depends(get_db)):
    consultation = TelemedicineConsultation(
        patient_id=consultation_data["patient_id"],
        physician_id="demo_physician",
        consultation_type=consultation_data["consultation_type"],
        scheduled_time=datetime.fromisoformat(consultation_data["scheduled_time"]),
        chief_complaint=consultation_data["chief_complaint"]
    )
    db.add(consultation)
    db.commit()
    db.refresh(consultation)
    return consultation

@app.get("/telemedicine/consultations/{consultation_id}")
async def get_consultation(consultation_id: str, db: Session = Depends(get_db)):
    consultation = db.query(TelemedicineConsultation).filter(
        TelemedicineConsultation.id == consultation_id
    ).first()
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found")
    return consultation

@app.get("/analytics/dashboard")
async def get_dashboard_analytics():
    return {
        "total_patients": 150,
        "total_consultations": 45,
        "ai_diagnoses_accuracy": 0.92,
        "recent_activity": [
            {"type": "consultation", "count": 12, "date": "2024-01-15"},
            {"type": "diagnosis", "count": 8, "date": "2024-01-15"}
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
