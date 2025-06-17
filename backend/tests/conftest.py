"""Test configuration and fixtures."""

import os
import sys
import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///test.db"
os.environ["SECRET_KEY"] = "test-secret-key"
os.environ["ALGORITHM"] = "HS256"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"
os.environ["MINIO_ENDPOINT"] = "localhost:9000"
os.environ["MINIO_ACCESS_KEY"] = "minioadmin"
os.environ["MINIO_SECRET_KEY"] = "minioadmin"

# Import after setting environment
from app.core.config import settings
from app.models import Base


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    import asyncio

    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def async_db_engine():
    """Create async database engine for tests."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:", echo=False, future=True
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest_asyncio.fixture
async def async_db_session(async_db_engine):
    """Create async database session for tests."""
    async_session = sessionmaker(
        async_db_engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        yield session


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis_mock = MagicMock()
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.set = AsyncMock(return_value=True)
    redis_mock.delete = AsyncMock(return_value=1)
    redis_mock.exists = AsyncMock(return_value=0)
    return redis_mock


@pytest.fixture
def mock_minio():
    """Mock MinIO client."""
    minio_mock = MagicMock()
    minio_mock.bucket_exists = Mock(return_value=True)
    minio_mock.put_object = Mock()
    minio_mock.get_object = Mock()
    minio_mock.remove_object = Mock()
    return minio_mock


@pytest.fixture
def mock_ml_service():
    """Mock ML model service."""
    ml_mock = MagicMock()
    ml_mock.predict_arrhythmia = AsyncMock(
        return_value={
            "prediction": "normal",
            "confidence": 0.95,
            "probabilities": {"normal": 0.95, "afib": 0.05},
        }
    )
    ml_mock.predict_pathology = AsyncMock(
        return_value={"predictions": [], "confidence": 0.90}
    )
    return ml_mock


@pytest.fixture
def mock_ecg_processor():
    """Mock ECG processor."""
    processor_mock = MagicMock()
    processor_mock.load_ecg_file = AsyncMock()
    processor_mock.preprocess_signal = AsyncMock()
    processor_mock.extract_features = AsyncMock(
        return_value={
            "heart_rate": 72,
            "pr_interval": 160,
            "qrs_duration": 90,
            "qt_interval": 400,
        }
    )
    return processor_mock


@pytest.fixture
def sample_ecg_data():
    """Sample ECG data for testing."""
    import numpy as np

    return {
        "signal": np.random.randn(5000, 12),
        "sampling_rate": 500,
        "duration": 10.0,
        "leads": [
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ],
    }


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "email": "test@example.com",
        "password": "testpass123",
        "full_name": "Test User",
        "role": "physician",
    }


@pytest.fixture
def sample_patient_data():
    """Sample patient data for testing."""
    return {
        "name": "John Doe",
        "birth_date": "1980-01-01",
        "gender": "M",
        "medical_record_number": "MRN123456",
        "contact_info": {"phone": "+1234567890", "email": "john.doe@example.com"},
    }


# Skip markers for different test categories
skip_db_tests = pytest.mark.skipif(
    os.getenv("SKIP_DB_TESTS", "").lower() == "true", reason="Database tests skipped"
)

skip_api_tests = pytest.mark.skipif(
    os.getenv("SKIP_API_TESTS", "").lower() == "true", reason="API tests skipped"
)

skip_slow_tests = pytest.mark.skipif(
    os.getenv("SKIP_SLOW_TESTS", "").lower() == "true", reason="Slow tests skipped"
)

skip_integration_tests = pytest.mark.skipif(
    os.getenv("SKIP_INTEGRATION_TESTS", "").lower() == "true",
    reason="Integration tests skipped",
)
