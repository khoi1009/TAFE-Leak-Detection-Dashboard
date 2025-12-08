# backend/app/api/incidents.py
"""
Incidents API endpoints.
"""
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pydantic import BaseModel

from app.core.database import get_db
from app.core.security import get_current_active_user
from app.models.user import User
from app.models.incident import Incident

router = APIRouter(prefix="/incidents", tags=["Incidents"])


# Pydantic schemas
class IncidentCreate(BaseModel):
    property_id: str
    school_name: Optional[str] = None
    confidence: float = 0.0
    signal_type: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    address: Optional[str] = None
    priority: str = "medium"


class IncidentUpdate(BaseModel):
    status: Optional[str] = None
    priority: Optional[str] = None
    assigned_to: Optional[int] = None
    resolution_notes: Optional[str] = None
    estimated_cost: Optional[float] = None
    actual_cost: Optional[float] = None
    water_saved: Optional[float] = None
    is_false_alarm: Optional[bool] = None
    false_alarm_pattern: Optional[str] = None


class IncidentResponse(BaseModel):
    id: int
    property_id: str
    school_name: Optional[str]
    detection_date: datetime
    confidence: float
    signal_type: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    address: Optional[str]
    status: str
    priority: str
    assigned_to: Optional[int]
    resolved_at: Optional[datetime]
    resolution_notes: Optional[str]
    is_false_alarm: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class IncidentStats(BaseModel):
    total: int
    new: int
    acknowledged: int
    watching: int
    escalated: int
    resolved: int
    ignored: int


@router.get("/", response_model=List[IncidentResponse])
async def list_incidents(
    status: Optional[str] = None,
    priority: Optional[str] = None,
    limit: int = Query(default=100, le=500),
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List all incidents with optional filtering."""
    query = select(Incident)

    if status:
        query = query.where(Incident.status == status)
    if priority:
        query = query.where(Incident.priority == priority)

    query = query.order_by(Incident.detection_date.desc()).limit(limit).offset(offset)
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/stats", response_model=IncidentStats)
async def get_incident_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get incident statistics."""
    result = await db.execute(select(func.count(Incident.id)))
    total = result.scalar() or 0

    stats = {"total": total}
    for status in [
        "new",
        "acknowledged",
        "watching",
        "escalated",
        "resolved",
        "ignored",
    ]:
        result = await db.execute(
            select(func.count(Incident.id)).where(Incident.status == status)
        )
        stats[status] = result.scalar() or 0

    return IncidentStats(**stats)


@router.get("/{incident_id}", response_model=IncidentResponse)
async def get_incident(
    incident_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get a specific incident."""
    result = await db.execute(select(Incident).where(Incident.id == incident_id))
    incident = result.scalar_one_or_none()

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    return incident


@router.post("/", response_model=IncidentResponse, status_code=201)
async def create_incident(
    incident_data: IncidentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Create a new incident."""
    incident = Incident(**incident_data.model_dump())
    db.add(incident)
    await db.commit()
    await db.refresh(incident)
    return incident


@router.patch("/{incident_id}", response_model=IncidentResponse)
async def update_incident(
    incident_id: int,
    incident_data: IncidentUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Update an incident."""
    result = await db.execute(select(Incident).where(Incident.id == incident_id))
    incident = result.scalar_one_or_none()

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    # Update fields
    update_data = incident_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(incident, field, value)

    # Set resolved_at if status changed to resolved
    if incident_data.status == "resolved" and not incident.resolved_at:
        incident.resolved_at = datetime.utcnow()

    # Set assigned_at if assigned_to changed
    if incident_data.assigned_to and not incident.assigned_at:
        incident.assigned_at = datetime.utcnow()

    await db.commit()
    await db.refresh(incident)
    return incident
