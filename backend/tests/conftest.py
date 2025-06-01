"""Test configuration and fixtures."""

import asyncio
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

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
    from app.db.session import get_engine
    
    base_engine = get_engine()
    base_url = str(base_engine.url)
    
    if base_url.endswith("/cardioai_test"):
        test_url = base_url  # Use CI test database directly
    elif "/cardioai_pro" in base_url:
        test_url = base_url.replace("/cardioai_pro", "/cardioai_pro_test")
    else:
        url_parts = base_url.rsplit("/", 1)
        test_url = url_parts[0] + "/" + url_parts[1] + "_test"
    
    engine = create_async_engine(test_url, echo=False)
    
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
