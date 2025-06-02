"""
Notification endpoints.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.user import User
from app.schemas.notification import Notification, NotificationList
from app.services.notification_service import NotificationService
from app.services.user_service import UserService

router = APIRouter()


@router.get("/", response_model=NotificationList)
async def get_notifications(
    limit: int = 50,
    offset: int = 0,
    unread_only: bool = False,
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get user notifications."""
    notification_service = NotificationService(db)

    notifications = await notification_service.get_user_notifications(
        current_user.id, limit, offset, unread_only
    )

    notifications_schemas = [Notification.from_orm(n) for n in notifications]
    return NotificationList(
        notifications=notifications_schemas,
        total=len(notifications),  # Simplified
        page=offset // limit + 1,
        size=limit,
    )


@router.post("/{notification_id}/read")
async def mark_notification_read(
    notification_id: int,
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Mark notification as read."""
    notification_service = NotificationService(db)

    success = await notification_service.mark_notification_read(
        notification_id, current_user.id
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification not found"
        )

    return {"message": "Notification marked as read"}


@router.post("/mark-all-read")
async def mark_all_read(
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Mark all notifications as read."""
    notification_service = NotificationService(db)

    count = await notification_service.mark_all_read(current_user.id)

    return {"message": f"Marked {count} notifications as read"}


@router.get("/unread-count")
async def get_unread_count(
    current_user: User = Depends(UserService.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get unread notification count."""
    notification_service = NotificationService(db)

    count = await notification_service.get_unread_count(current_user.id)

    return {"unread_count": count}
