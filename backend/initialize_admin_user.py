import asyncio
from app.db.session import get_db
from app.services.user_service import UserService
from app.schemas.user import UserCreate
from app.core.config import settings

async def initialize_admin_user():
    async for db in get_db():
        user_service = UserService(db)
        
        existing_user = await user_service.get_user_by_username(settings.FIRST_SUPERUSER)
        if existing_user:
            print(f'✅ Admin user already exists: {existing_user.username}')
            return
        
        admin_data = UserCreate(
            username=settings.FIRST_SUPERUSER,
            email=settings.FIRST_SUPERUSER_EMAIL,
            password="Admin123!@#",
            first_name="Admin",
            last_name="User",
            phone="",
            role="admin",
            license_number="",
            specialty="",
            institution="CardioAI Pro",
            experience_years=0
        )
        
        try:
            admin_user = await user_service.create_user(admin_data)
            print(f'✅ Admin user created successfully: {admin_user.username}')
        except Exception as e:
            print(f'❌ Error creating admin user: {e}')
        
        break

if __name__ == "__main__":
    asyncio.run(initialize_admin_user())
