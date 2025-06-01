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
    else:
        from app.db.session import get_engine
        base_engine = get_engine()
        base_url = str(base_engine.url)
        
        if base_url.endswith("/cardioai_test"):
            database_url = base_url
        elif "/cardioai_pro" in base_url:
            database_url = base_url.replace("/cardioai_pro", "/cardioai_pro_test")
        elif base_url.endswith("/postgres") and os.getenv("ENVIRONMENT") == "test":
            database_url = base_url
        else:
            url_parts = base_url.rsplit("/", 1)
            database_url = url_parts[0] + "/" + url_parts[1] + "_test"
        
        print(f"🧪 Local Test database URL: {database_url.split('@')[1]}")
    
    engine = create_async_engine(
        database_url,
        echo=False,
        poolclass=NullPool,
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest_asyncio.fixture
async def test_db(test_engine):
    """Create test database session."""
    async_session = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session


@pytest.fixture
def client(test_db):
    """Create test client."""
    def override_get_db():
        yield test_db
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()
