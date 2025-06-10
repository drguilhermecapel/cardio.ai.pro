import asyncio
from app.db.session import get_db
from app.services.user_service import UserService
from app.core.security import get_password_hash
from app.core.config import settings

async def update_admin_password():
    async for db in get_db():
        user_service = UserService(db)
        
        admin_user = await user_service.get_user_by_username(settings.FIRST_SUPERUSER)
        if not admin_user:
            print('❌ Admin user not found')
            return
        
        new_password_hash = get_password_hash("Admin123!@#")
        
        try:
            await user_service.update_user(admin_user.id, {"hashed_password": new_password_hash})
            print(f'✅ Admin password updated successfully for user: {admin_user.username}')
        except Exception as e:
            print(f'❌ Error updating admin password: {e}')
        
        break

if __name__ == "__main__":
    asyncio.run(update_admin_password())
