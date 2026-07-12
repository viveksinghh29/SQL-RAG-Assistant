"""Pydantic schemas for User. Endpoints (Phase 11) never return the ORM
model or `hashed_password` directly — only `UserRead`."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.auth.roles import Role


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: str = Field(min_length=1, max_length=255)
    role: Role = Role.EMPLOYEE


class UserUpdate(BaseModel):
    full_name: str | None = Field(default=None, min_length=1, max_length=255)
    role: Role | None = None
    is_active: bool | None = None


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    full_name: str
    role: Role
    is_active: bool
    created_at: datetime
