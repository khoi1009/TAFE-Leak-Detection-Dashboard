# backend/app/models/user.py
"""
User model for authentication.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Enum
from sqlalchemy.orm import relationship
import enum

from app.core.database import Base


class UserRole(str, enum.Enum):
    """User roles for access control."""

    admin = "admin"
    manager = "manager"
    operator = "operator"
    viewer = "viewer"


class User(Base):
    """User model for authentication and access control."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=True)
    role = Column(String(20), default=UserRole.viewer.value)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Relationships
    incidents = relationship(
        "Incident", back_populates="assigned_user", foreign_keys="Incident.assigned_to"
    )

    def __repr__(self):
        return f"<User {self.username}>"
