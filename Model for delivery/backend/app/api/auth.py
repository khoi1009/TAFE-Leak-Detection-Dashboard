# backend/app/api/auth.py
"""
Authentication API endpoints - Pydantic v2 Compatible.
"""
from datetime import datetime, timedelta
from typing import Optional, Annotated
from fastapi import APIRouter, Depends, HTTPException, status, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, EmailStr

from app.core.database import get_db
from app.core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    get_current_user,
    verify_token,
)
from app.core.config import settings
from app.models.user import User

router = APIRouter(prefix="/auth", tags=["Authentication"])


# Pydantic schemas (v2 compatible)
class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    full_name: Optional[str]
    role: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenRefresh(BaseModel):
    refresh_token: str


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserResponse


@router.post(
    "/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED
)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register a new user."""
    # Check if email exists
    result = await db.execute(select(User).where(User.email == user_data.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")

    # Check if username exists
    result = await db.execute(select(User).where(User.username == user_data.username))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Username already taken")

    # Create user
    user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=get_password_hash(user_data.password),
        full_name=user_data.full_name,
        role="viewer",
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@router.post("/login", response_model=LoginResponse)
async def login(
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
    db: AsyncSession = Depends(get_db),
):
    """Login and get access token."""
    # Find user by username or email
    result = await db.execute(
        select(User).where((User.username == username) | (User.email == username))
    )
    user = result.scalar_one_or_none()

    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    # Update last login
    user.last_login = datetime.utcnow()
    await db.commit()

    # Create tokens
    access_token = create_access_token(data={"sub": user.username})
    refresh_token = create_refresh_token(data={"sub": user.username})

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        user=UserResponse.model_validate(user),
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(token_data: TokenRefresh, db: AsyncSession = Depends(get_db)):
    """Refresh access token using refresh token."""
    payload = verify_token(token_data.refresh_token, expected_type="refresh")
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    username = payload.get("sub")
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise HTTPException(status_code=400, detail="User not found or inactive")

    access_token = create_access_token(data={"sub": user.username})
    refresh_token_new = create_refresh_token(data={"sub": user.username})

    return Token(
        access_token=access_token,
        refresh_token=refresh_token_new,
        token_type="bearer",
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user info."""
    return current_user


@router.post("/logout")
async def logout():
    """Logout (client should discard tokens)."""
    return {"message": "Successfully logged out"}
