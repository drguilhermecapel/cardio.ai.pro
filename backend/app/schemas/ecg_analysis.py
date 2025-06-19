from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

class ECGAnalysisBase(BaseModel):
    """Schema base para análise de ECG"""
    patient_id: int = Field(..., description="ID do paciente")
    ecg_data: Dict[str, Any] = Field(..., description="Dados do ECG em formato JSON")
    
class ECGAnalysisCreate(ECGAnalysisBase):
    """Schema para criação de análise de ECG"""
    file_path: Optional[str] = Field(None, description="Caminho do arquivo ECG")
    notes: Optional[str] = Field(None, description="Observações adicionais")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "patient_id": 12345,
                "ecg_data": {
                    "leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
                    "duration": 10.0,
                    "sample_rate": 500,
                    "amplitude_data": []
                },
                "file_path": "/uploads/ecg_12345.dat",
                "notes": "ECG de rotina"
            }
        }
    )
    
class ECGAnalysisUpdate(BaseModel):
    """Schema para atualização de análise de ECG"""
    status: Optional[str] = Field(None, description="Status da análise")
    results: Optional[Dict[str, Any]] = Field(None, description="Resultados da análise")
    notes: Optional[str] = Field(None, description="Observações atualizadas")
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "status": "completed",
                "results": {
                    "heart_rate": 72,
                    "rhythm": "sinus",
                    "abnormalities": [],
                    "confidence": 0.95
                },
                "notes": "Análise concluída - ECG normal"
            }
        }
    )

class ECGAnalysisResponse(ECGAnalysisBase):
    """Schema de resposta para análise de ECG"""
    id: int = Field(..., description="ID único da análise")
    status: str = Field(..., description="Status atual da análise")
    results: Optional[Dict[str, Any]] = Field(None, description="Resultados da análise")
    created_at: datetime = Field(..., description="Data/hora de criação")
    updated_at: Optional[datetime] = Field(None, description="Data/hora da última atualização")
    notes: Optional[str] = Field(None, description="Observações")
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "patient_id": 12345,
                "status": "completed",
                "ecg_data": {
                    "leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
                    "duration": 10.0,
                    "sample_rate": 500
                },
                "results": {
                    "heart_rate": 72,
                    "rhythm": "sinus",
                    "abnormalities": []
                },
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:35:00Z",
                "notes": "ECG normal"
            }
        }
    )

class ECGAnalysisList(BaseModel):
    """Schema para lista de análises de ECG"""
    analyses: List[ECGAnalysisResponse]
    total: int = Field(..., description="Total de análises")
    page: int = Field(1, description="Página atual")
    per_page: int = Field(10, description="Itens por página")
    
    model_config = ConfigDict(from_attributes=True)

