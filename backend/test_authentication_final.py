#!/usr/bin/env python3
"""
Final authentication test to verify the admin user authentication system works correctly.
"""
import asyncio
from app.services.user_service import UserService
from app.db.session import get_db
from app.core.config import settings

async def test_authentication_system():
    """Test the complete authentication system."""
    print("🔍 Testing CardioAI Pro Authentication System")
    print("=" * 50)
    
    async for db in get_db():
        user_service = UserService(db)
        
        print("1. Checking admin user exists...")
        admin_user = await user_service.get_user_by_username(settings.FIRST_SUPERUSER)
        if admin_user:
            print(f"   ✅ Admin user found: {admin_user.username}")
            print(f"   📧 Email: {admin_user.email}")
            print(f"   🔓 Active: {admin_user.is_active}")
            print(f"   👑 Superuser: {admin_user.is_superuser}")
        else:
            print(f"   ❌ Admin user '{settings.FIRST_SUPERUSER}' not found")
            return False
        
        print("\n2. Testing authentication with correct credentials...")
        auth_result = await user_service.authenticate_user(
            settings.FIRST_SUPERUSER, 
            settings.FIRST_SUPERUSER_PASSWORD
        )
        if auth_result:
            print("   ✅ Authentication successful")
            print(f"   👤 Authenticated user: {auth_result.username}")
        else:
            print("   ❌ Authentication failed with correct credentials")
            return False
        
        print("\n3. Testing authentication with wrong password...")
        wrong_auth = await user_service.authenticate_user(
            settings.FIRST_SUPERUSER, 
            "wrong_password"
        )
        if not wrong_auth:
            print("   ✅ Authentication correctly rejected wrong password")
        else:
            print("   ❌ Authentication incorrectly accepted wrong password")
            return False
        
        print("\n4. Checking configuration...")
        print(f"   Username: {settings.FIRST_SUPERUSER}")
        print(f"   Email: {settings.FIRST_SUPERUSER_EMAIL}")
        print(f"   Password configured: {'Yes' if settings.FIRST_SUPERUSER_PASSWORD else 'No'}")
        
        print("\n" + "=" * 50)
        print("🎉 All authentication tests PASSED!")
        print("✅ Authentication system is working correctly")
        return True
        
        break

if __name__ == "__main__":
    success = asyncio.run(test_authentication_system())
    exit(0 if success else 1)
