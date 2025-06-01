"""
Database initialization utilities.
"""

import asyncio
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.constants import UserRoles
from app.core.security import get_password_hash
from app.db.session import async_sessionmaker
from app.models.user import User

logger = logging.getLogger(__name__)


async def init_db() -> None:
    """Initialize database with default data."""
    async with async_sessionmaker() as session:
        await create_admin_user(session)


async def create_admin_user(session: AsyncSession) -> User | None:
    """Create default admin user if it doesn't exist."""
    try:
        from sqlalchemy.future import select
        stmt = select(User).where(User.username == settings.FIRST_SUPERUSER)
        result = await session.execute(stmt)
        existing_user = result.scalar_one_or_none()

        if existing_user:
            logger.info("Admin user already exists")
            return existing_user

        admin_user = User(
            username=settings.FIRST_SUPERUSER,
            email=settings.FIRST_SUPERUSER_EMAIL,
            hashed_password=get_password_hash(settings.FIRST_SUPERUSER_PASSWORD),
            first_name="Admin",
            last_name="User",
            role=UserRoles.ADMIN,
            is_active=True,
            is_superuser=True,
        )

        session.add(admin_user)
        await session.commit()
        await session.refresh(admin_user)

        logger.info("Created admin user: %s", admin_user.username)
        return admin_user

    except Exception as e:
        logger.error("Failed to create admin user: %s", str(e))
        await session.rollback()
        return None


if __name__ == "__main__":
    asyncio.run(init_db())
