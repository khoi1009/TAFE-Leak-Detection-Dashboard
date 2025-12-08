# backend/app/models/incident.py
"""
Incident model for leak detection events.
"""
from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Text,
    Boolean,
)
from sqlalchemy.orm import relationship

from app.core.database import Base


class Incident(Base):
    """Incident model for tracking leak detection events."""

    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, index=True)
    property_id = Column(String(50), index=True, nullable=False)
    school_name = Column(String(200), nullable=True)

    # Detection details
    detection_date = Column(DateTime, default=datetime.utcnow)
    confidence = Column(Float, default=0.0)
    signal_type = Column(String(50), nullable=True)  # MNF, RESIDUAL, CUSUM, etc.

    # Location
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    address = Column(String(500), nullable=True)

    # Status tracking
    status = Column(
        String(50), default="new"
    )  # new, acknowledged, watching, escalated, resolved, ignored
    priority = Column(String(20), default="medium")  # low, medium, high, critical

    # Assignment
    assigned_to = Column(Integer, ForeignKey("users.id"), nullable=True)
    assigned_at = Column(DateTime, nullable=True)

    # Resolution
    resolved_at = Column(DateTime, nullable=True)
    resolution_notes = Column(Text, nullable=True)
    estimated_cost = Column(Float, nullable=True)
    actual_cost = Column(Float, nullable=True)
    water_saved = Column(Float, nullable=True)  # liters saved

    # Metadata
    is_false_alarm = Column(Boolean, default=False)
    false_alarm_pattern = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    assigned_user = relationship(
        "User", back_populates="incidents", foreign_keys=[assigned_to]
    )

    def __repr__(self):
        return f"<Incident {self.id} - {self.property_id}>"
