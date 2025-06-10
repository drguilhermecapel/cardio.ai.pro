import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.core.security import get_password_hash
from app.models.user import User
from app.models.base import Base

async def force_create_admin():
    engine = create_async_engine(settings.DATABASE_URL, echo=True)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async with async_session() as session:
        from sqlalchemy import select
        result = await session.execute(select(User).where(User.username == "admin@cardioai.pro"))
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            existing_user.hashed_password = get_password_hash("Admin123!@#")
            existing_user.is_active = True
            existing_user.is_superuser = True
            await session.commit()
            print(f"✅ Updated existing admin user: {existing_user.username}")
        else:
            admin_user = User(
                username="admin@cardioai.pro",
                email="admin@cardioai.pro",
                hashed_password=get_password_hash("Admin123!@#"),
                first_name="Admin",
                last_name="User",
                phone="",
                role="admin",
                license_number="",
                specialty="",
                institution="CardioAI Pro",
                experience_years=0,
                is_active=True,
                is_verified=True,
                is_superuser=True
            )
            session.add(admin_user)
            await session.commit()
            print(f"✅ Created new admin user: {admin_user.username}")

if __name__ == "__main__":
    asyncio.run(force_create_admin())
