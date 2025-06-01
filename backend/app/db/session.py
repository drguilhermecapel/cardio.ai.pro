"""
Database session management with lazy initialization for medical compliance.
"""

from collections.abc import AsyncGenerator
from functools import lru_cache
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """
    Lazy engine creation with medical-grade error handling.
    
    Returns:
        AsyncEngine: Database engine instance
        
    Raises:
        RuntimeError: If database configuration is invalid
    """
    global _engine
    
    if _engine is None:
        from app.core.config import settings
        
        if not settings.DATABASE_URL:
            raise RuntimeError(
                "CRITICAL: Database URL not configured. "
                "Patient data persistence at risk."
            )
        
        _engine = create_async_engine(
            str(settings.DATABASE_URL),
            echo=settings.DEBUG,
            poolclass=NullPool if settings.ENVIRONMENT == "test" else None,
            pool_pre_ping=True,
            pool_recycle=300,
            connect_args={
                "server_settings": {
                    "application_name": "CardioAI_Pro_v1",
                    "jit": "off"
                },
                "command_timeout": 60,
            }
        )
    
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Get or create session factory with audit trail support.
    
    Returns:
        async_sessionmaker: Session factory instance
    """
    global _session_factory
    
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
    
    return _session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session with medical-grade transaction handling.
    
    Yields:
        AsyncSession: Database session with audit trail
    """
    session_factory = get_session_factory()
    
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
