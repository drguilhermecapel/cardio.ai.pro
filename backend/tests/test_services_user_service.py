import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.core.config import settings

if settings.STANDALONE_MODE:
    pytest.skip("User service tests skipped in standalone mode", allow_module_level=True)

from app.services.user_service import UserService
from app.schemas.user import UserCreate, UserUpdate


@pytest.mark.asyncio
async def test_user_service_initialization():
    """Test user service initialization."""
    mock_session = AsyncMock()
    service = UserService(mock_session)
    
    assert service.session == mock_session


@pytest.mark.asyncio
async def test_create_user():
    """Test user creation."""
    mock_session = AsyncMock()
    service = UserService(mock_session)
    
    user_data = UserCreate(
        email="test@example.com",
        password="password123",
        full_name="Test User"
    )
    
    with patch.object(service, 'user_repo') as mock_repo:
        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.email = "test@example.com"
        mock_repo.create.return_value = mock_user
        
        result = await service.create_user(user_data)
        
        mock_repo.create.assert_called_once()
        assert result.id == 1
        assert result.email == "test@example.com"


@pytest.mark.asyncio
async def test_get_user_by_id():
    """Test getting user by ID."""
    mock_session = AsyncMock()
    service = UserService(mock_session)
    
    with patch.object(service, 'user_repo') as mock_repo:
        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.email = "test@example.com"
        mock_repo.get.return_value = mock_user
        
        result = await service.get_user(1)
        
        mock_repo.get.assert_called_once_with(1)
        assert result.id == 1


@pytest.mark.asyncio
async def test_get_user_by_email():
    """Test getting user by email."""
    mock_session = AsyncMock()
    service = UserService(mock_session)
    
    with patch.object(service, 'user_repo') as mock_repo:
        mock_user = MagicMock()
        mock_user.email = "test@example.com"
        mock_repo.get_by_email.return_value = mock_user
        
        result = await service.get_user_by_email("test@example.com")
        
        mock_repo.get_by_email.assert_called_once_with("test@example.com")
        assert result.email == "test@example.com"


@pytest.mark.asyncio
async def test_update_user():
    """Test user update."""
    mock_session = AsyncMock()
    service = UserService(mock_session)
    
    user_update = UserUpdate(full_name="Updated Name")
    
    with patch.object(service, 'user_repo') as mock_repo:
        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.full_name = "Updated Name"
        mock_repo.update.return_value = mock_user
        
        result = await service.update_user(1, user_update)
        
        mock_repo.update.assert_called_once_with(1, user_update)
        assert result.full_name == "Updated Name"


@pytest.mark.asyncio
async def test_delete_user():
    """Test user deletion."""
    mock_session = AsyncMock()
    service = UserService(mock_session)
    
    with patch.object(service, 'user_repo') as mock_repo:
        mock_repo.delete.return_value = True
        
        result = await service.delete_user(1)
        
        mock_repo.delete.assert_called_once_with(1)
        assert result is True


@pytest.mark.asyncio
async def test_authenticate_user():
    """Test user authentication."""
    mock_session = AsyncMock()
    service = UserService(mock_session)
    
    with patch.object(service, 'user_repo') as mock_repo:
        with patch('app.services.user_service.verify_password') as mock_verify:
            mock_user = MagicMock()
            mock_user.email = "test@example.com"
            mock_user.hashed_password = "hashed_password"
            mock_repo.get_by_email.return_value = mock_user
            mock_verify.return_value = True
            
            result = await service.authenticate_user("test@example.com", "password123")
            
            mock_repo.get_by_email.assert_called_once_with("test@example.com")
            mock_verify.assert_called_once_with("password123", "hashed_password")
            assert result == mock_user


@pytest.mark.asyncio
async def test_authenticate_user_invalid_password():
    """Test user authentication with invalid password."""
    mock_session = AsyncMock()
    service = UserService(mock_session)
    
    with patch.object(service, 'user_repo') as mock_repo:
        with patch('app.services.user_service.verify_password') as mock_verify:
            mock_user = MagicMock()
            mock_user.email = "test@example.com"
            mock_user.hashed_password = "hashed_password"
            mock_repo.get_by_email.return_value = mock_user
            mock_verify.return_value = False
            
            result = await service.authenticate_user("test@example.com", "wrong_password")
            
            assert result is None


@pytest.mark.asyncio
async def test_authenticate_user_not_found():
    """Test user authentication with non-existent user."""
    mock_session = AsyncMock()
    service = UserService(mock_session)
    
    with patch.object(service, 'user_repo') as mock_repo:
        mock_repo.get_by_email.return_value = None
        
        result = await service.authenticate_user("nonexistent@example.com", "password123")
        
        assert result is None


@pytest.mark.asyncio
async def test_list_users():
    """Test listing users."""
    mock_session = AsyncMock()
    service = UserService(mock_session)
    
    with patch.object(service, 'user_repo') as mock_repo:
        mock_users = [MagicMock(), MagicMock()]
        mock_repo.list.return_value = mock_users
        
        result = await service.list_users(skip=0, limit=10)
        
        mock_repo.list.assert_called_once_with(skip=0, limit=10)
        assert len(result) == 2


@pytest.mark.asyncio
async def test_user_exists():
    """Test checking if user exists."""
    mock_session = AsyncMock()
    service = UserService(mock_session)
    
    with patch.object(service, 'user_repo') as mock_repo:
        mock_repo.get_by_email.return_value = MagicMock()
        
        result = await service.user_exists("test@example.com")
        
        mock_repo.get_by_email.assert_called_once_with("test@example.com")
        assert result is True


@pytest.mark.asyncio
async def test_user_not_exists():
    """Test checking if user does not exist."""
    mock_session = AsyncMock()
    service = UserService(mock_session)
    
    with patch.object(service, 'user_repo') as mock_repo:
        mock_repo.get_by_email.return_value = None
        
        result = await service.user_exists("nonexistent@example.com")
        
        assert result is False
