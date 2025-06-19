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

from app.schemas.user import User, UserUpdate
from app.services.user_service import UserService
from app.repositories.user_repository import UserRepository
from app.models.user import UserRoles

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/", response_model=List[User])
async def get_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get list of users (admin only)"""
    if current_user.role != UserRoles.ADMIN:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    user_repo = UserRepository(db)
    users = await user_repo.get_users(skip=skip, limit=limit)
    return users

@router.get("/{user_id}", response_model=User)
async def get_user(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user by ID"""
    user_repo = UserRepository(db)
    user = await user_repo.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.put("/{user_id}", response_model=User)
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update user information"""
    if current_user.id != user_id and current_user.role != UserRoles.ADMIN:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    user_repo = UserRepository(db)
    user_service = UserService(db, user_repo)
    user = await user_service.update_user(user_id, user_update)
    return user

@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete user (admin only)"""
    if current_user.role != UserRoles.ADMIN:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    user_repo = UserRepository(db)
    await user_repo.delete_user(user_id)
    return {"message": "User deleted successfully"}
