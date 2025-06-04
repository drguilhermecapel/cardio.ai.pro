"""Test configuration and fixtures."""

import asyncio
import os
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from unittest.mock import patch

os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///test_cardio.db"

from app.db.session import get_db
from app.main import app
from app.models.base import Base
from app.models import *





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


@pytest.fixture
def client():
    """Create test client with mocked database."""
    from unittest.mock import AsyncMock
    
    async def mock_get_db():
        mock_db = AsyncMock()
        yield mock_db
    
    app.dependency_overrides[get_db] = mock_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


@pytest.fixture
def notification_service(test_db):
    """Create notification service instance."""
    from app.services.notification_service import NotificationService
    return NotificationService(db=test_db)
