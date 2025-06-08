"""
Database initialization utilities for standalone CardioAI Pro.
"""

import asyncio
import logging
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.constants import UserRoles
from app.core.security import get_password_hash
from app.db.session import get_engine, get_session_factory
from app.models.base import Base
from app.models.user import User

logger = logging.getLogger(__name__)


async def init_db() -> None:
    """Initialize database with tables and default data."""
    try:
        logger.info("Initializing CardioAI Pro database...")

        engine = get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("Database tables created successfully")

        session_factory = get_session_factory()
        async with session_factory() as session:
            await create_admin_user(session)

        logger.info("Database initialization completed successfully")

    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise


async def create_admin_user(session: AsyncSession) -> User | None:
    """Create default admin user if it doesn't exist."""
    try:
        from sqlalchemy.future import select

        admin_email = "admin@cardioai.pro"
        stmt = select(User).where(User.email == admin_email)
        result = await session.execute(stmt)
        existing_user = result.scalar_one_or_none()

        if existing_user:
            logger.info("Admin user already exists")
            return existing_user

        admin_user = User()
        admin_user.username = "admin"
        admin_user.email = admin_email
        default_password = settings.FIRST_SUPERUSER_PASSWORD
        if default_password == "changeme123":
            import secrets
            import string
            alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
            default_password = ''.join(secrets.choice(alphabet) for _ in range(16))
            logger.warning(f"Generated secure admin password: {default_password}")
            logger.warning("IMPORTANT: Save this password and change it after first login!")

        admin_user.hashed_password = get_password_hash(default_password)
        admin_user.first_name = "CardioAI"
        admin_user.last_name = "Administrator"
        admin_user.role = UserRoles.ADMIN
        admin_user.is_active = True
        admin_user.is_superuser = True

        session.add(admin_user)
        await session.commit()
        await session.refresh(admin_user)

        logger.info("Created default admin user: %s", admin_user.email)
        return admin_user

    except Exception as e:
        logger.error("Failed to create admin user: %s", str(e))
        await session.rollback()
        return None


async def check_database_exists() -> bool:
    """Check if database file exists and is accessible."""
    try:
        db_url = str(settings.DATABASE_URL)
        if "sqlite" in db_url:
            db_path_str = db_url.replace("sqlite+aiosqlite:///", "")
            if not db_path_str.startswith("/"):
                db_path = Path.cwd() / db_path_str
            else:
                db_path = Path(db_path_str)

            return db_path.exists() and db_path.is_file()

        return True

    except Exception as e:
        logger.error(f"Error checking database existence: {str(e)}")
        return False


async def ensure_database_ready() -> None:
    """Ensure database is ready for use, initialize if needed."""
    try:
        db_exists = await check_database_exists()

        if not db_exists:
            logger.info("Database not found, initializing...")
            await init_db()
        else:
            logger.info("Database found, verifying structure...")
            engine = get_engine()
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            session_factory = get_session_factory()
            async with session_factory() as session:
                await create_admin_user(session)

        logger.info("Database is ready for use")

    except Exception as e:
        logger.error(f"Database preparation failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(init_db())
