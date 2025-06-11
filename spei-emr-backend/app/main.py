"""FastAPI application entry point for the SPEI EMR backend."""

from __future__ import annotations

from pathlib import Path
from typing import List

import jwt
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import (HTTPAuthorizationCredentials, HTTPBearer)
from sqlalchemy.orm import Session

if __name__ == "__main__" and __package__ is None:
    # Allow running with `python app/main.py` by adjusting sys.path
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    __package__ = "app"
from datetime import datetime, timedelta
import bcrypt
import os
from .database import SessionLocal, engine
from .models import Base, User, Patient, MedicalRecord, Consultation
from .schemas import UserCreate, UserLogin, PatientCreate, PatientResponse, MedicalRecordCreate, MedicalRecordResponse, ConsultationCreate, ConsultationResponse, DiagnosticRequest, DiagnosticResponse

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="SPEI - Sistema de Prontuário Eletrônico Inteligente",
    description="Electronic Medical Record System with AI Diagnostics",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=24)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/auth/login")
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    if user_data.username == "admin" and user_data.password == "admin":
        access_token = create_access_token(data={"sub": user_data.username})
        return {"access_token": access_token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/patients", response_model=List[PatientResponse])
async def get_patients(current_user: str = Depends(verify_token), db: Session = Depends(get_db)):
    patients = db.query(Patient).all()
    return patients

@app.post("/patients", response_model=PatientResponse)
async def create_patient(patient: PatientCreate, current_user: str = Depends(verify_token), db: Session = Depends(get_db)):
    db_patient = Patient(**patient.dict())
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient

@app.get("/medical-records", response_model=List[MedicalRecordResponse])
async def get_medical_records(current_user: str = Depends(verify_token), db: Session = Depends(get_db)):
    records = db.query(MedicalRecord).all()
    return records

@app.post("/medical-records", response_model=MedicalRecordResponse)
async def create_medical_record(record: MedicalRecordCreate, current_user: str = Depends(verify_token), db: Session = Depends(get_db)):
    db_record = MedicalRecord(**record.dict())
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return db_record

@app.get("/consultations", response_model=List[ConsultationResponse])
async def get_consultations(current_user: str = Depends(verify_token), db: Session = Depends(get_db)):
    consultations = db.query(Consultation).all()
    return consultations

@app.post("/consultations", response_model=ConsultationResponse)
async def create_consultation(consultation: ConsultationCreate, current_user: str = Depends(verify_token), db: Session = Depends(get_db)):
    db_consultation = Consultation(**consultation.dict())
    db.add(db_consultation)
    db.commit()
    db.refresh(db_consultation)
    return db_consultation

@app.post("/ai/diagnose", response_model=DiagnosticResponse)
async def ai_diagnose(request: DiagnosticRequest, current_user: str = Depends(verify_token)):
    symptoms = request.symptoms.lower()
    
    diagnoses = []
    
    if any(symptom in symptoms for symptom in ["chest pain", "dor no peito", "angina"]):
        diagnoses.append({
            "condition": "Angina Pectoris",
            "confidence": 0.85,
            "type": "primary"
        })
        diagnoses.append({
            "condition": "Myocardial Infarction",
            "confidence": 0.65,
            "type": "differential"
        })
    
    if any(symptom in symptoms for symptom in ["shortness of breath", "dyspnea", "falta de ar"]):
        diagnoses.append({
            "condition": "Heart Failure",
            "confidence": 0.75,
            "type": "primary"
        })
        diagnoses.append({
            "condition": "Pulmonary Embolism",
            "confidence": 0.60,
            "type": "differential"
        })
    
    if any(symptom in symptoms for symptom in ["fever", "febre", "temperature"]):
        diagnoses.append({
            "condition": "Viral Infection",
            "confidence": 0.70,
            "type": "primary"
        })
        diagnoses.append({
            "condition": "Bacterial Infection",
            "confidence": 0.55,
            "type": "differential"
        })
    
    if not diagnoses:
        diagnoses.append({
            "condition": "Further Investigation Required",
            "confidence": 0.50,
            "type": "primary"
        })
    
    treatment_plan = "Consult with specialist for detailed evaluation and treatment planning."
    
    return DiagnosticResponse(
        diagnoses=diagnoses,
        treatment_plan=treatment_plan,
        confidence_score=max([d["confidence"] for d in diagnoses]) if diagnoses else 0.5,
        compliance_notice="This AI diagnostic assistance is for informational purposes only. Professional medical validation is required before any clinical decisions."
    )

@app.get("/")
async def root():
    return {"message": "SPEI - Sistema de Prontuário Eletrônico Inteligente API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
