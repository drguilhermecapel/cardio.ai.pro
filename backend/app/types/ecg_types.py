from __future__ import annotations

from typing import TypeVar, Protocol, Any
import pandas as pd
import numpy as np
from numpy.typing import NDArray

ECGDataFrame = TypeVar('ECGDataFrame', bound=pd.DataFrame)

class ECGSchema(Protocol):
    """Schema para validação de DataFrames de ECG"""
    timestamp: pd.Series[pd.Timestamp]
    lead_I: pd.Series[float]
    lead_II: pd.Series[float]
    lead_III: pd.Series[float]
    lead_aVR: pd.Series[float]
    lead_aVL: pd.Series[float]
    lead_aVF: pd.Series[float]
    lead_V1: pd.Series[float]
    lead_V2: pd.Series[float]
    lead_V3: pd.Series[float]
    lead_V4: pd.Series[float]
    lead_V5: pd.Series[float]
    lead_V6: pd.Series[float]
    
    heart_rate: pd.Series[int]
    rr_interval: pd.Series[float]
    qt_interval: pd.Series[float]

class ECGAnalysisResult(Protocol):
    """Type-safe result container para análise de ECG"""
    predictions: dict[str, float]
    features: dict[str, float]
    confidence: float
    metadata: dict[str, Any]
