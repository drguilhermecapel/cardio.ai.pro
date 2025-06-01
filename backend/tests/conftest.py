"""Test configuration and fixtures."""

import asyncio
import os
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from app.db.session import get_db
from app.main import app
from app.models.base import Base
from app.models import *  # noqa: F403, F401


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    """Create test database engine."""
    
    if os.getenv("CI"):
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            pytest.fail("DATABASE_URL not set in CI environment")
        print(f"🧪 CI Test database URL: {database_url.split('@')[1]}")
        
        engine = create_async_engine(
            database_url,
            echo=True,
            poolclass=NullPool,
        )
    else:
        database_url = "sqlite+aiosqlite:///test_cardio.db"
        print(f"🧪 Local Test database URL: {database_url}")
        
        engine = create_async_engine(
            database_url,
            echo=True,
            poolclass=NullPool,
            connect_args={"check_same_thread": False}
        )
    
    async with engine.begin() as conn:
        print(f"🔧 Creating {len(Base.metadata.tables)} tables...")
        await conn.run_sync(Base.metadata.create_all)
        
        from sqlalchemy import text
        result = await conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
        tables = [row[0] for row in result.fetchall()]
        print(f"✅ Created tables: {tables}")
        
        if 'validations' not in tables:
            print("❌ ERROR: validations table not created!")
            raise RuntimeError("Failed to create validations table")
    
    yield engine
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()
    
    if os.path.exists("test_cardio.db"):
        os.remove("test_cardio.db")


@pytest_asyncio.fixture
async def test_db(test_engine):
    """Create test database session."""
    async_session = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.rollback()
            await session.close()


@pytest.fixture
def client(test_db):
    """Create test client."""
    def override_get_db():
        yield test_db
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


@pytest.fixture
def notification_service(test_db):
    """Create notification service instance."""
    from app.services.notification_service import NotificationService
    return NotificationService(db=test_db)
