from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, status, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.core.exceptions import (
    NotFoundException, 
    ValidationException, 
    UnauthorizedException
)

from app.schemas.user import UserCreate, User, Token
from app.services.user_service import UserService
from app.repositories.user_repository import UserRepository

router = APIRouter(tags=["authentication"])

@router.post("/register", response_model=User)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user"""
    try:
        user_repo = UserRepository(db)
        user_service = UserService(db, user_repo)
        user = await user_service.create_user(user_data)
        return user
    except ValidationException as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/login", response_model=Token)
async def login(
    username: str,
    password: str,
    db: AsyncSession = Depends(get_db)
):
    """Login and get access token"""
    try:
        user_repo = UserRepository(db)
        user_service = UserService(db, user_repo)
        user = await user_service.authenticate_user(username, password)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials"
            )
        # Generate token (simplified for now)
        return {
            "access_token": f"token_for_{user.id}",
            "token_type": "bearer"
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed")

@router.get("/me", response_model=User)
async def get_current_user_info(
    current_user = Depends(get_current_user)
):
    """Get current user information"""
    return current_user

@router.post("/logout")
async def logout(
    current_user = Depends(get_current_user)
):
    """Logout current user"""
    # In a real implementation, you would invalidate the token
    return {"message": "Logged out successfully"}

@router.post("/refresh", response_model=Token)
async def refresh_token(
    current_user: User = Depends(get_current_user)
):
    """Refresh access token"""
    return {
        "access_token": f"refreshed_token_for_{current_user.id}",
        "token_type": "bearer"
    }
