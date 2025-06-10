import asyncio
from app.db.session import get_db
from app.services.user_service import UserService
from app.core.config import settings

async def check_admin_user():
    async for db in get_db():
        user_service = UserService(db)
        user = await user_service.get_user_by_username('admin@cardioai.pro')
        if user:
            print(f'✅ Admin user exists: {user.username}, active: {user.is_active}')
        else:
            print('❌ Admin user not found in database')
        break

if __name__ == "__main__":
    asyncio.run(check_admin_user())
