"""
Configuração global para testes do CardioAI Pro.
Este arquivo configura mocks e fixtures comuns para todos os testes.
"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Configurar variáveis de ambiente para testes
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["SECRET_KEY"] = "test-secret-key-for-cardioai-pro"
os.environ["REDIS_URL"] = ""
os.environ["CELERY_BROKER_URL"] = ""

# Adicionar o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock para módulos que podem não estar instalados
sys.modules["redis"] = MagicMock()
sys.modules["celery"] = MagicMock()
sys.modules["minio"] = MagicMock()
sys.modules["pyedflib"] = MagicMock()
sys.modules["wfdb"] = MagicMock()


@pytest.fixture
def mock_ecg_signal():
    """Mock de sinal ECG para testes."""
    import numpy as np
    
    # Sinal ECG simulado de 10 segundos a 360 Hz
    duration = 10
    sampling_rate = 360
    t = np.linspace(0, duration, duration * sampling_rate)
    
    # Simular batimentos cardíacos
    heart_rate = 72
    ecg = np.zeros_like(t)
    
    # Adicionar picos R
    r_peaks = []
    for i in range(int(duration * heart_rate / 60)):
        peak_time = i * 60 / heart_rate
        peak_idx = int(peak_time * sampling_rate)
        if peak_idx < len(ecg):
            ecg[peak_idx] = 1.0
            r_peaks.append(peak_idx)
    
    return ecg, r_peaks, sampling_rate


@pytest.fixture
def mock_ml_service():
    """Mock do serviço de ML."""
    service = Mock()
    service.analyze_ecg = Mock(return_value={
        "predictions": {"NORMAL": 0.95, "AFIB": 0.05},
        "confidence": 0.95,
        "features": {"heart_rate": 72, "pr_interval": 160}
    })
    service.load_model = Mock(return_value=True)
    return service


@pytest.fixture
def mock_validation_service():
    """Mock do serviço de validação."""
    service = Mock()
    service.validate_analysis = Mock(return_value=True)
    service.create_validation = Mock(return_value={"id": 1, "status": "pending"})
    return service


@pytest_asyncio.fixture
async def test_db():
    """Cria sessão de banco de dados para testes."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    
    async_session_maker = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session_maker() as session:
        yield session
        await session.rollback()
        await session.close()


@pytest.fixture
def auth_headers():
    """Headers de autenticação para testes."""
    return {"Authorization": "Bearer test-token"}
