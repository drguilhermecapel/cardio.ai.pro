"""
User schemas.
"""

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, field_validator

from app.core.constants import UserRoles

class UserBase(BaseModel):
    """Base user schema."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    phone: str | None = Field(None, max_length=20)
    role: UserRoles = UserRoles.VIEWER
    license_number: str | None = Field(None, max_length=50)
    specialty: str | None = Field(None, max_length=100)
    institution: str | None = Field(None, max_length=200)
    experience_years: int | None = Field(None, ge=0, le=70)

class UserCreate(UserBase):
    """User creation schema."""
    password: str = Field(..., min_length=8, max_length=100)

    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in v):
            raise ValueError('Password must contain at least one special character')
        return v

class UserUpdate(BaseModel):
    """User update schema."""
    email: EmailStr | None = None
    first_name: str | None = Field(None, min_length=1, max_length=100)
    last_name: str | None = Field(None, min_length=1, max_length=100)
    phone: str | None = Field(None, max_length=20)
    license_number: str | None = Field(None, max_length=50)
    specialty: str | None = Field(None, max_length=100)
    institution: str | None = Field(None, max_length=200)
    experience_years: int | None = Field(None, ge=0, le=70)

class UserInDB(UserBase):
    """User in database schema."""
    id: int
    is_active: bool
    is_verified: bool
    is_superuser: bool
    last_login: datetime | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class User(UserInDB):
    """User response schema."""
    pass

class UserList(BaseModel):
    """User list response schema."""
    users: list[User]
    total: int
    page: int
    size: int

class PasswordChange(BaseModel):
    """Password change schema."""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)

    @field_validator('new_password')
    @classmethod
    def validate_new_password(cls, v: str) -> str:
        """Validate new password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in v):
            raise ValueError('Password must contain at least one special character')
        return v

class PasswordReset(BaseModel):
    """Password reset schema."""
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    """Password reset confirmation schema."""
    token: str
    new_password: str = Field(..., min_length=8, max_length=100)

class Token(BaseModel):
    """Token schema."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class TokenRefresh(BaseModel):
    """Token refresh schema."""
    refresh_token: str

class APIKeyCreate(BaseModel):
    """API key creation schema."""
    name: str = Field(..., min_length=1, max_length=100)
    scopes: list[str] = Field(..., min_length=1)
    expires_at: datetime | None = None

class APIKeyResponse(BaseModel):
    """API key response schema."""
    id: int
    name: str
    key: str  # Only returned on creation
    scopes: list[str]
    expires_at: datetime | None
    created_at: datetime

    class Config:
        from_attributes = True

class APIKeyList(BaseModel):
    """API key list schema."""
    id: int
    name: str
    scopes: list[str]
    is_active: bool
    expires_at: datetime | None
    last_used: datetime | None
    usage_count: int
    created_at: datetime

    class Config:
        from_attributes = True
