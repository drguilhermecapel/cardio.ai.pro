from sqlalchemy.orm import declarative_base

# Usar importação correta para SQLAlchemy 2.0
Base = declarative_base()

# Importar todos os modelos aqui para garantir que sejam registrados
from app.models.ecg_analysis import ECGAnalysis, AnalysisStatus

__all__ = ["Base", "ECGAnalysis", "AnalysisStatus"]

