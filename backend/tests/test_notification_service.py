"""Test Notification Service."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from app.services.notification_service import NotificationService
from app.models.notification import Notification
from app.schemas.notification import NotificationCreate, NotificationUpdate


@pytest.fixture
def notification_service(test_db):
    """Create notification service instance."""
    return NotificationService(db=test_db)


@pytest.fixture
def sample_notification_data():
    """Sample notification data."""
    return NotificationCreate(
        user_id=1,
        title="ECG Analysis Complete",
        message="Your ECG analysis has been completed and is ready for review.",
        notification_type="analysis_complete",
        priority="medium",
        metadata={"analysis_id": 123, "patient_id": 456}
    )


@pytest.mark.asyncio
async def test_create_notification_success(notification_service, sample_notification_data):
    """Test successful notification creation."""
    result = await notification_service.create_notification(sample_notification_data)
    
    assert result is not None
    assert result.user_id == sample_notification_data.user_id
    assert result.title == sample_notification_data.title
    assert result.message == sample_notification_data.message
    assert result.notification_type == sample_notification_data.notification_type
    assert result.status == "pending"


@pytest.mark.asyncio
async def test_get_notification_by_id(notification_service, test_db):
    """Test retrieving notification by ID."""
    notification = Notification(
        user_id=1,
        title="Test Notification",
        message="Test message",
        notification_type="info",
        priority="low",
        status="pending"
    )
    test_db.add(notification)
    await test_db.commit()
    await test_db.refresh(notification)
    
    result = await notification_service.get_notification_by_id(notification.id)
    
    assert result is not None
    assert result.id == notification.id
    assert result.title == "Test Notification"


@pytest.mark.asyncio
async def test_get_notification_by_id_not_found(notification_service):
    """Test retrieving non-existent notification."""
    result = await notification_service.get_notification_by_id(99999)
    assert result is None


@pytest.mark.asyncio
async def test_get_notifications_by_user(notification_service, test_db):
    """Test retrieving notifications by user ID."""
    user_id = 1
    
    for i in range(3):
        notification = Notification(
            user_id=user_id,
            title=f"Notification {i}",
            message=f"Message {i}",
            notification_type="info",
            priority="medium",
            status="pending"
        )
        test_db.add(notification)
    
    await test_db.commit()
    
    results = await notification_service.get_notifications_by_user(user_id)
    
    assert len(results) == 3
    assert all(n.user_id == user_id for n in results)


@pytest.mark.asyncio
async def test_get_unread_notifications(notification_service, test_db):
    """Test retrieving unread notifications."""
    user_id = 1
    
    for i in range(5):
        status = "pending" if i < 3 else "read"
        notification = Notification(
            user_id=user_id,
            title=f"Notification {i}",
            message=f"Message {i}",
            notification_type="info",
            priority="medium",
            status=status
        )
        test_db.add(notification)
    
    await test_db.commit()
    
    results = await notification_service.get_unread_notifications(user_id)
    
    assert len(results) == 3
    assert all(n.status == "pending" for n in results)


@pytest.mark.asyncio
async def test_mark_notification_as_read(notification_service, test_db):
    """Test marking notification as read."""
    notification = Notification(
        user_id=1,
        title="Test Notification",
        message="Test message",
        notification_type="info",
        priority="medium",
        status="pending"
    )
    test_db.add(notification)
    await test_db.commit()
    await test_db.refresh(notification)
    
    result = await notification_service.mark_as_read(notification.id)
    
    assert result is not None
    assert result.status == "read"
    assert result.read_at is not None


@pytest.mark.asyncio
async def test_mark_all_as_read(notification_service, test_db):
    """Test marking all notifications as read for a user."""
    user_id = 1
    
    for i in range(3):
        notification = Notification(
            user_id=user_id,
            title=f"Notification {i}",
            message=f"Message {i}",
            notification_type="info",
            priority="medium",
            status="pending"
        )
        test_db.add(notification)
    
    await test_db.commit()
    
    count = await notification_service.mark_all_as_read(user_id)
    
    assert count == 3
    
    unread = await notification_service.get_unread_notifications(user_id)
    assert len(unread) == 0


@pytest.mark.asyncio
async def test_delete_notification(notification_service, test_db):
    """Test deleting notification."""
    notification = Notification(
        user_id=1,
        title="Test Notification",
        message="Test message",
        notification_type="info",
        priority="medium",
        status="pending"
    )
    test_db.add(notification)
    await test_db.commit()
    await test_db.refresh(notification)
    
    success = await notification_service.delete_notification(notification.id)
    assert success is True
    
    deleted = await notification_service.get_notification_by_id(notification.id)
    assert deleted is None


@pytest.mark.asyncio
async def test_send_real_time_notification(notification_service):
    """Test sending real-time notification via WebSocket."""
    with patch('app.services.notification_service.websocket_manager') as mock_ws:
        mock_ws.send_personal_message = AsyncMock()
        
        notification_data = {
            "user_id": 1,
            "title": "Real-time Alert",
            "message": "Urgent notification",
            "type": "alert"
        }
        
        await notification_service.send_real_time_notification(notification_data)
        
        mock_ws.send_personal_message.assert_called_once()


@pytest.mark.asyncio
async def test_send_email_notification(notification_service):
    """Test sending email notification."""
    with patch('app.services.notification_service.email_service') as mock_email:
        mock_email.send_email = AsyncMock(return_value=True)
        
        notification_data = {
            "user_id": 1,
            "email": "test@example.com",
            "subject": "ECG Analysis Complete",
            "body": "Your analysis is ready for review."
        }
        
        result = await notification_service.send_email_notification(notification_data)
        
        assert result is True
        mock_email.send_email.assert_called_once()


@pytest.mark.asyncio
async def test_send_sms_notification(notification_service):
    """Test sending SMS notification."""
    with patch('app.services.notification_service.sms_service') as mock_sms:
        mock_sms.send_sms = AsyncMock(return_value=True)
        
        notification_data = {
            "user_id": 1,
            "phone": "+1234567890",
            "message": "Critical ECG alert"
        }
        
        result = await notification_service.send_sms_notification(notification_data)
        
        assert result is True
        mock_sms.send_sms.assert_called_once()


@pytest.mark.asyncio
async def test_get_notification_preferences(notification_service, test_db):
    """Test retrieving user notification preferences."""
    user_id = 1
    
    preferences = await notification_service.get_notification_preferences(user_id)
    
    assert preferences is not None
    assert "email_enabled" in preferences
    assert "sms_enabled" in preferences
    assert "push_enabled" in preferences


@pytest.mark.asyncio
async def test_update_notification_preferences(notification_service):
    """Test updating user notification preferences."""
    user_id = 1
    new_preferences = {
        "email_enabled": True,
        "sms_enabled": False,
        "push_enabled": True,
        "quiet_hours": {"start": "22:00", "end": "08:00"}
    }
    
    result = await notification_service.update_notification_preferences(user_id, new_preferences)
    
    assert result is not None
    assert result["email_enabled"] is True
    assert result["sms_enabled"] is False


@pytest.mark.asyncio
async def test_schedule_notification(notification_service):
    """Test scheduling future notification."""
    scheduled_time = datetime.utcnow() + timedelta(hours=1)
    
    notification_data = NotificationCreate(
        user_id=1,
        title="Scheduled Reminder",
        message="Don't forget to review the ECG analysis",
        notification_type="reminder",
        priority="low",
        scheduled_at=scheduled_time
    )
    
    result = await notification_service.schedule_notification(notification_data)
    
    assert result is not None
    assert result.scheduled_at == scheduled_time
    assert result.status == "scheduled"


@pytest.mark.asyncio
async def test_process_scheduled_notifications(notification_service, test_db):
    """Test processing scheduled notifications."""
    past_time = datetime.utcnow() - timedelta(minutes=5)
    future_time = datetime.utcnow() + timedelta(hours=1)
    
    past_notification = Notification(
        user_id=1,
        title="Past Notification",
        message="Should be sent",
        notification_type="reminder",
        priority="medium",
        status="scheduled",
        scheduled_at=past_time
    )
    
    future_notification = Notification(
        user_id=1,
        title="Future Notification",
        message="Should not be sent yet",
        notification_type="reminder",
        priority="medium",
        status="scheduled",
        scheduled_at=future_time
    )
    
    test_db.add(past_notification)
    test_db.add(future_notification)
    await test_db.commit()
    
    with patch.object(notification_service, 'send_real_time_notification') as mock_send:
        mock_send.return_value = AsyncMock()
        
        count = await notification_service.process_scheduled_notifications()
        
        assert count == 1


@pytest.mark.asyncio
async def test_get_notification_statistics(notification_service, test_db):
    """Test retrieving notification statistics."""
    user_id = 1
    
    statuses = ["pending", "read", "sent", "pending", "read"]
    for i, status in enumerate(statuses):
        notification = Notification(
            user_id=user_id,
            title=f"Notification {i}",
            message=f"Message {i}",
            notification_type="info",
            priority="medium",
            status=status
        )
        test_db.add(notification)
    
    await test_db.commit()
    
    stats = await notification_service.get_notification_statistics(user_id)
    
    assert "total" in stats
    assert "unread" in stats
    assert "read" in stats
    assert stats["total"] == 5
    assert stats["unread"] == 2
    assert stats["read"] == 2


@pytest.mark.asyncio
async def test_bulk_delete_notifications(notification_service, test_db):
    """Test bulk deletion of notifications."""
    user_id = 1
    notification_ids = []
    
    for i in range(3):
        notification = Notification(
            user_id=user_id,
            title=f"Notification {i}",
            message=f"Message {i}",
            notification_type="info",
            priority="medium",
            status="read"
        )
        test_db.add(notification)
        await test_db.flush()
        notification_ids.append(notification.id)
    
    await test_db.commit()
    
    count = await notification_service.bulk_delete_notifications(notification_ids)
    
    assert count == 3
    
    for notification_id in notification_ids:
        deleted = await notification_service.get_notification_by_id(notification_id)
        assert deleted is None


@pytest.mark.asyncio
async def test_notification_delivery_retry(notification_service):
    """Test notification delivery retry mechanism."""
    with patch('app.services.notification_service.email_service') as mock_email:
        mock_email.send_email = AsyncMock(side_effect=[False, False, True])
        
        notification_data = {
            "user_id": 1,
            "email": "test@example.com",
            "subject": "Test",
            "body": "Test message"
        }
        
        result = await notification_service.send_with_retry(
            notification_service.send_email_notification,
            notification_data,
            max_retries=3
        )
        
        assert result is True
        assert mock_email.send_email.call_count == 3


@pytest.mark.asyncio
async def test_notification_template_rendering(notification_service):
    """Test notification template rendering."""
    template_data = {
        "patient_name": "John Doe",
        "analysis_type": "ECG",
        "completion_time": "2025-06-01 14:30:00"
    }
    
    rendered = await notification_service.render_template(
        "analysis_complete",
        template_data
    )
    
    assert "John Doe" in rendered["title"]
    assert "ECG" in rendered["message"]
    assert "2025-06-01 14:30:00" in rendered["message"]


@pytest.mark.asyncio
async def test_notification_rate_limiting(notification_service):
    """Test notification rate limiting."""
    user_id = 1
    
    with patch.object(notification_service, '_check_rate_limit') as mock_rate_limit:
        mock_rate_limit.return_value = False
        
        notification_data = NotificationCreate(
            user_id=user_id,
            title="Rate Limited",
            message="This should be rate limited",
            notification_type="info",
            priority="low"
        )
        
        with pytest.raises(Exception, match="Rate limit exceeded"):
            await notification_service.create_notification(notification_data)


@pytest.mark.asyncio
async def test_notification_filtering(notification_service, test_db):
    """Test notification filtering by type and priority."""
    user_id = 1
    
    notifications_data = [
        ("alert", "high"),
        ("info", "low"),
        ("alert", "medium"),
        ("reminder", "high")
    ]
    
    for notification_type, priority in notifications_data:
        notification = Notification(
            user_id=user_id,
            title=f"{notification_type} notification",
            message="Test message",
            notification_type=notification_type,
            priority=priority,
            status="pending"
        )
        test_db.add(notification)
    
    await test_db.commit()
    
    alerts = await notification_service.get_notifications_by_type(user_id, "alert")
    assert len(alerts) == 2
    
    high_priority = await notification_service.get_notifications_by_priority(user_id, "high")
    assert len(high_priority) == 2
