"""Test notification service."""

import pytest
from unittest.mock import AsyncMock, Mock

from app.services.notification_service import NotificationService
from app.models.notification import Notification
from app.core.constants import NotificationPriority, NotificationType, ClinicalUrgency


@pytest.fixture
def notification_service(test_db):
    """Create notification service instance."""
    return NotificationService(db=test_db)


@pytest.fixture
def sample_notification():
    """Sample notification data."""
    notification = Notification()
    notification.user_id = 1
    notification.title = "Test Notification"
    notification.message = "Test message"
    notification.notification_type = NotificationType.ANALYSIS_COMPLETE
    notification.priority = NotificationPriority.NORMAL
    notification.channels = ["in_app", "email"]
    return notification


@pytest.mark.asyncio
async def test_send_validation_assignment(notification_service):
    """Test sending validation assignment notification."""
    await notification_service.send_validation_assignment(
        validator_id=1, 
        analysis_id=123, 
        urgency=ClinicalUrgency.HIGH
    )


@pytest.mark.asyncio
async def test_send_urgent_validation_alert(notification_service):
    """Test sending urgent validation alert."""
    await notification_service.send_urgent_validation_alert(
        validator_id=1, 
        analysis_id=123
    )


@pytest.mark.asyncio
async def test_send_validation_complete(notification_service):
    """Test sending validation complete notification."""
    await notification_service.send_validation_complete(
        user_id=1,
        analysis_id=123,
        status="approved"
    )


@pytest.mark.asyncio
async def test_send_analysis_complete(notification_service):
    """Test sending analysis complete notification."""
    await notification_service.send_analysis_complete(
        user_id=1,
        analysis_id=123,
        has_critical_findings=False
    )


@pytest.mark.asyncio
async def test_send_quality_alert(notification_service):
    """Test sending quality alert notification."""
    await notification_service.send_quality_alert(
        user_id=1,
        analysis_id=123,
        quality_issues=["noise", "artifacts"]
    )


@pytest.mark.asyncio
async def test_send_system_alert(notification_service):
    """Test sending system alert notification."""
    await notification_service.send_system_alert(
        title="System Alert",
        message="System maintenance scheduled",
        priority=NotificationPriority.HIGH
    )


@pytest.mark.asyncio
async def test_get_user_notifications(notification_service):
    """Test getting user notifications."""
    notifications = await notification_service.get_user_notifications(user_id=1)
    assert isinstance(notifications, list)


@pytest.mark.asyncio
async def test_mark_notification_read(notification_service):
    """Test marking notification as read."""
    result = await notification_service.mark_notification_read(
        notification_id=1, 
        user_id=1
    )
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_mark_all_read(notification_service):
    """Test marking all notifications as read."""
    count = await notification_service.mark_all_read(user_id=1)
    assert isinstance(count, int)


@pytest.mark.asyncio
async def test_get_unread_count(notification_service):
    """Test getting unread notification count."""
    count = await notification_service.get_unread_count(user_id=1)
    assert isinstance(count, int)
