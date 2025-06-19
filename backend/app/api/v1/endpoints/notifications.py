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

from app.schemas.notification import Notification
from app.services.notification_service import NotificationService
from app.repositories.notification_repository import NotificationRepository

router = APIRouter(prefix="/notifications", tags=["notifications"])

@router.get("/", response_model=List[Notification])
async def get_notifications(
    unread_only: bool = Query(False),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user notifications"""
    notification_repo = NotificationRepository(db)
    notification_service = NotificationService(db, notification_repo)
    
    notifications = await notification_service.get_user_notifications(
        current_user.id,
        unread_only=unread_only,
        skip=skip,
        limit=limit
    )
    return notifications

@router.post("/{notification_id}/read")
async def mark_notification_read(
    notification_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Mark notification as read"""
    notification_repo = NotificationRepository(db)
    notification_service = NotificationService(db, notification_repo)
    
    await notification_service.mark_notification_read(notification_id, current_user.id)
    return {"message": "Notification marked as read"}

@router.post("/read-all")
async def mark_all_read(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Mark all notifications as read"""
    notification_repo = NotificationRepository(db)
    notification_service = NotificationService(db, notification_repo)
    
    await notification_service.mark_all_read(current_user.id)
    return {"message": "All notifications marked as read"}

@router.get("/unread-count")
async def get_unread_count(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get count of unread notifications"""
    notification_repo = NotificationRepository(db)
    notification_service = NotificationService(db, notification_repo)
    
    count = await notification_service.get_unread_count(current_user.id)
    return {"count": count}
