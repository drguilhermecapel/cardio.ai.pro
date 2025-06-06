"""
User schemas.
"""

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, field_validator

from app.core.constants import UserRoles
from app.services.i18n_service import i18n_service


class UserBase(BaseModel):
    """Base user schema."""
    username: str = Field(..., min_length=3, max_length=50, description=i18n_service.translate("schemas.user.username.description"))
    email: EmailStr = Field(..., description=i18n_service.translate("schemas.user.email.description"))
    first_name: str = Field(..., min_length=1, max_length=100, description=i18n_service.translate("schemas.user.first_name.description"))
    last_name: str = Field(..., min_length=1, max_length=100, description=i18n_service.translate("schemas.user.last_name.description"))
    phone: str | None = Field(None, max_length=20, description=i18n_service.translate("schemas.user.phone.description"))
    role: UserRoles = Field(UserRoles.VIEWER, description=i18n_service.translate("schemas.user.role.description"))
    license_number: str | None = Field(None, max_length=50, description=i18n_service.translate("schemas.user.license_number.description"))
    specialty: str | None = Field(None, max_length=100, description=i18n_service.translate("schemas.user.specialty.description"))
    institution: str | None = Field(None, max_length=200, description=i18n_service.translate("schemas.user.institution.description"))
    experience_years: int | None = Field(None, ge=0, le=70, description=i18n_service.translate("schemas.user.experience_years.description"))


class UserCreate(UserBase):
    """User creation schema."""
    password: str = Field(..., min_length=8, max_length=100, description=i18n_service.translate("schemas.user.password.description"))

    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError(i18n_service.translate("schemas.user.password.error.too_short"))
        if not any(c.isupper() for c in v):
            raise ValueError(i18n_service.translate("schemas.user.password.error.no_uppercase"))
        if not any(c.islower() for c in v):
            raise ValueError(i18n_service.translate("schemas.user.password.error.no_lowercase"))
        if not any(c.isdigit() for c in v):
            raise ValueError(i18n_service.translate("schemas.user.password.error.no_digit"))
        if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in v):
            raise ValueError(i18n_service.translate("schemas.user.password.error.no_special"))
        return v


class UserUpdate(BaseModel):
    """User update schema."""
    email: EmailStr | None = Field(None, description=i18n_service.translate("schemas.user.email.description"))
    first_name: str | None = Field(None, min_length=1, max_length=100, description=i18n_service.translate("schemas.user.first_name.description"))
    last_name: str | None = Field(None, min_length=1, max_length=100, description=i18n_service.translate("schemas.user.last_name.description"))
    phone: str | None = Field(None, max_length=20, description=i18n_service.translate("schemas.user.phone.description"))
    license_number: str | None = Field(None, max_length=50, description=i18n_service.translate("schemas.user.license_number.description"))
    specialty: str | None = Field(None, max_length=100, description=i18n_service.translate("schemas.user.specialty.description"))
    institution: str | None = Field(None, max_length=200, description=i18n_service.translate("schemas.user.institution.description"))
    experience_years: int | None = Field(None, ge=0, le=70, description=i18n_service.translate("schemas.user.experience_years.description"))


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
    current_password: str = Field(..., description=i18n_service.translate("schemas.user.current_password.description"))
    new_password: str = Field(..., min_length=8, max_length=100, description=i18n_service.translate("schemas.user.new_password.description"))

    @field_validator('new_password')
    @classmethod
    def validate_new_password(cls, v: str) -> str:
        """Validate new password strength."""
        if len(v) < 8:
            raise ValueError(i18n_service.translate("schemas.user.password.error.too_short"))
        if not any(c.isupper() for c in v):
            raise ValueError(i18n_service.translate("schemas.user.password.error.no_uppercase"))
        if not any(c.islower() for c in v):
            raise ValueError(i18n_service.translate("schemas.user.password.error.no_lowercase"))
        if not any(c.isdigit() for c in v):
            raise ValueError(i18n_service.translate("schemas.user.password.error.no_digit"))
        if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in v):
            raise ValueError(i18n_service.translate("schemas.user.password.error.no_special"))
        return v


class PasswordReset(BaseModel):
    """Password reset schema."""
    email: EmailStr = Field(..., description=i18n_service.translate("schemas.user.email.description"))


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation schema."""
    token: str = Field(..., description=i18n_service.translate("schemas.user.reset_token.description"))
    new_password: str = Field(..., min_length=8, max_length=100, description=i18n_service.translate("schemas.user.new_password.description"))


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
    name: str = Field(..., min_length=1, max_length=100, description=i18n_service.translate("schemas.user.api_key_name.description"))
    scopes: list[str] = Field(..., min_length=1, description=i18n_service.translate("schemas.user.api_key_scopes.description"))
    expires_at: datetime | None = Field(None, description=i18n_service.translate("schemas.user.api_key_expires.description"))


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
