from enum import Enum
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

class ClinicalUrgency(str, Enum):
    ROUTINE = "routine"
    URGENT = "urgent"
    EMERGENCY = "emergency"

class ECGAnalysisBase(BaseModel):
    patient_id: Optional[str] = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    urgency: ClinicalUrgency = ClinicalUrgency.ROUTINE

class ECGAnalysisResponse(ECGAnalysisBase):
    id: str
    results: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class ECGAnalysisUpdate(BaseModel):
    status: Optional[ProcessingStatus] = None
    results: Optional[Dict[str, Any]] = None
    urgency: Optional[ClinicalUrgency] = None
