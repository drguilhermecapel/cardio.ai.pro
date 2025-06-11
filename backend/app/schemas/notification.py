"""
Notification schemas.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel

from app.core.constants import NotificationPriority, NotificationType


class NotificationBase(BaseModel):
    """Base notification schema."""
    title: str
    message: str
    notification_type: NotificationType
    priority: NotificationPriority

class NotificationCreate(NotificationBase):
    """Notification creation schema."""
    user_id: int
    channels: list[str] = []
    related_resource_type: str | None = None
    related_resource_id: int | None = None
    metadata: dict[str, Any] | None = None
    expires_at: datetime | None = None

class NotificationUpdate(BaseModel):
    """Notification update schema."""
    title: str | None = None
    message: str | None = None
    notification_type: NotificationType | None = None
    priority: NotificationPriority | None = None
    is_read: bool | None = None
    metadata: dict[str, Any] | None = None

class NotificationInDB(NotificationBase):
    """Notification in database schema."""
    id: int
    user_id: int
    channels: list[str]
    related_resource_type: str | None
    related_resource_id: int | None
    is_read: bool
    is_sent: bool
    sent_at: datetime | None
    read_at: datetime | None
    delivery_attempts: int
    last_delivery_attempt: datetime | None
    delivery_status: str | None
    metadata: dict[str, Any] | None
    expires_at: datetime | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class Notification(NotificationInDB):
    """Notification response schema."""
    pass

class NotificationList(BaseModel):
    """Notification list response schema."""
    notifications: list[Notification]
    total: int
    page: int
    size: int
