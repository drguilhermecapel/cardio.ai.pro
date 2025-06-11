"""
User management endpoints.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import UserRoles
from app.db.session import get_db
from app.models.user import User
from app.schemas.user import PasswordChange, UserList, UserUpdate
from app.schemas.user import User as UserSchema
from app.services.user_service import UserService

router = APIRouter()

@router.get("/me", response_model=UserSchema)
async def get_current_user_info(
    current_user: User = Depends(UserService.get_current_user),
) -> Any:
    """Get current user information."""
    return current_user

@router.put("/me", response_model=UserSchema)
async def update_current_user(
    user_update: UserUpdate,
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Update current user information."""
    user_service = UserService(db)

    update_data = user_update.dict(exclude_unset=True)

    updated_user = await user_service.repository.update_user(current_user.id, update_data)
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )

    return updated_user

@router.post("/me/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Change current user password."""
    user_service = UserService(db)

    from app.core.security import get_password_hash, verify_password
    if not verify_password(password_data.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password"
        )

    new_hashed_password = get_password_hash(password_data.new_password)
    await user_service.repository.update_user(
        current_user.id,
        {"hashed_password": new_hashed_password}
    )

    return {"message": "Password changed successfully"}

@router.get("/", response_model=UserList)
async def list_users(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """List users (admin only)."""
    if current_user.role != UserRoles.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    user_service = UserService(db)
    users = await user_service.repository.get_users(limit, offset)

    users_schemas = [UserSchema.from_orm(u) for u in users]
    return UserList(
        users=users_schemas,
        total=len(users),  # Simplified - in production, get actual count
        page=offset // limit + 1,
        size=limit,
    )

@router.get("/{user_id}", response_model=UserSchema)
async def get_user(
    user_id: int,
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get user by ID (admin only or own profile)."""
    if current_user.role != UserRoles.ADMIN and current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    user_service = UserService(db)
    user = await user_service.repository.get_user_by_id(user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return user
