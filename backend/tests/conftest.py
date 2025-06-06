
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from app.models.base import Base
from app.models import *

import pytest
import asyncio
import sys
from unittest.mock import MagicMock, AsyncMock

sys.modules['celery'] = MagicMock()
sys.modules['redis'] = MagicMock()
sys.modules['wfdb'] = MagicMock()
sys.modules['pyedflib'] = MagicMock()

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture(scope="function")
def mock_db():
    """Create a mock database session."""
    return AsyncMock()

@pytest.fixture(scope="function") 
def sample_signal():
    """Create sample ECG signal for testing."""
    import numpy as np
    return np.sin(np.linspace(0, 10, 1000))

@pytest_asyncio.fixture(scope="function")
async def test_db():
    """Create test database session with proper table creation."""
    database_url = "sqlite+aiosqlite:///:memory:"
    
    engine = create_async_engine(
        database_url,
        echo=False,
        poolclass=NullPool,
        connect_args={"check_same_thread": False}
    )
    
    from sqlalchemy.ext.asyncio import async_sessionmaker
    async_session = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
        session = async_session(bind=conn)
        try:
            yield session
        finally:
            await session.rollback()
            await session.close()
    
    await engine.dispose()
