from sqlalchemy import Column, Integer, String, DateTime, JSON, Enum as SQLEnum, Text
from sqlalchemy.orm import declarative_base
from datetime import datetime
import enum

Base = declarative_base()

class AnalysisStatus(enum.Enum):
    """Status da análise de ECG"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    
    def __str__(self):
        return self.value

class ECGAnalysis(Base):
    """Modelo de análise de ECG"""
    __tablename__ = "ecg_analyses"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_id = Column(Integer, nullable=False, index=True)
    status = Column(SQLEnum(AnalysisStatus), default=AnalysisStatus.PENDING, nullable=False)
    ecg_data = Column(JSON, nullable=False)
    results = Column(JSON, nullable=True)
    file_path = Column(String(500), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, onupdate=datetime.utcnow, nullable=True)
    
    def __repr__(self):
        return f"<ECGAnalysis(id={self.id}, patient_id={self.patient_id}, status={self.status})>"
    
    def to_dict(self):
        """Converte o modelo para dicionário"""
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "status": self.status.value if self.status else None,
            "ecg_data": self.ecg_data,
            "results": self.results,
            "file_path": self.file_path,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    def update_status(self, new_status: AnalysisStatus):
        """Atualiza o status da análise"""
        self.status = new_status
        self.updated_at = datetime.utcnow()
    
    def set_results(self, results: dict):
        """Define os resultados da análise"""
        self.results = results
        self.status = AnalysisStatus.COMPLETED
        self.updated_at = datetime.utcnow()
    
    def mark_as_failed(self, error_message: str = None):
        """Marca a análise como falhada"""
        self.status = AnalysisStatus.FAILED
        if error_message:
            self.results = {"error": error_message}
        self.updated_at = datetime.utcnow()

