import asyncio
from app.db.session import get_db
from app.services.user_service import UserService
from app.core.security import verify_password, get_password_hash
from app.core.config import settings

async def verify_and_fix_admin_password():
    """Verify and fix the admin user password if needed."""
    print("🔍 Starting admin password verification...")
    
    async for db in get_db():
        user_service = UserService(db)
        
        admin_user = await user_service.get_user_by_username(settings.FIRST_SUPERUSER)
        if not admin_user:
            print(f"❌ Admin user '{settings.FIRST_SUPERUSER}' not found in database")
            return
        
        print(f"✅ Admin user found: {admin_user.username}")
        print(f"   Email: {admin_user.email}")
        print(f"   Active: {admin_user.is_active}")
        print(f"   Superuser: {admin_user.is_superuser}")
        
        expected_password = settings.FIRST_SUPERUSER_PASSWORD
        print(f"🔐 Testing password verification with: '{expected_password}'")
        
        is_valid = verify_password(expected_password, admin_user.hashed_password)
        print(f"   Password verification result: {is_valid}")
        
        if not is_valid:
            print("🔧 Password verification failed. Updating password hash...")
            
            new_hash = get_password_hash(expected_password)
            print(f"   New hash generated: {new_hash[:50]}...")
            
            try:
                await user_service.update_user(admin_user.id, {"hashed_password": new_hash})
                print("✅ Password hash updated successfully")
                
                updated_user = await user_service.get_user_by_username(settings.FIRST_SUPERUSER)
                if updated_user and verify_password(expected_password, updated_user.hashed_password):
                    print("✅ Password verification now works correctly")
                else:
                    print("❌ Password verification still failing after update")
                    
            except Exception as e:
                print(f"❌ Error updating password: {e}")
        else:
            print("✅ Password verification is working correctly")
        
        print("🔐 Testing full authentication flow...")
        auth_result = await user_service.authenticate_user(settings.FIRST_SUPERUSER, expected_password)
        if auth_result:
            print("✅ Full authentication flow works correctly")
        else:
            print("❌ Full authentication flow is still failing")
        
        break

if __name__ == "__main__":
    asyncio.run(verify_and_fix_admin_password())
