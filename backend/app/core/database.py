"""
Configuração do banco de dados
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
import os

# URL do banco de dados
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./cardioai.db")

# Engine assíncrona
engine = create_async_engine(DATABASE_URL, echo=False, future=True)

# Session factory
async_session_maker = async_sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

# Base para modelos
Base = declarative_base()

# Dependência para obter sessão
async def get_db() -> AsyncSession:
    """Obtém sessão de banco de dados."""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()
