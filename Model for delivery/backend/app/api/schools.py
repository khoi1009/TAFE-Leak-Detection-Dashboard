# backend/app/api/schools.py
"""
Schools GIS API endpoints.
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import json
import os

from app.core.security import get_current_active_user
from app.models.user import User

router = APIRouter(prefix="/schools", tags=["Schools"])


class SchoolGIS(BaseModel):
    """School GIS data model."""

    name: str
    latitude: float
    longitude: float
    address: Optional[str] = None
    suburb: Optional[str] = None
    postcode: Optional[str] = None
    school_type: Optional[str] = None
    status: str = "normal"  # normal, warning, leak, critical


class SchoolListResponse(BaseModel):
    total: int
    schools: List[SchoolGIS]


# Cache for school data
_schools_cache = None


def load_schools_data():
    """Load schools GIS data from file."""
    global _schools_cache
    if _schools_cache is not None:
        return _schools_cache

    # Try to load from demo data
    gis_paths = [
        "../frontend/data/demo_schools_gis.json",
        "../../frontend/data/demo_schools_gis.json",
        "./demo_schools_gis.json",
    ]

    for path in gis_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                _schools_cache = json.load(f)
            return _schools_cache

    # Return empty if not found
    _schools_cache = []
    return _schools_cache


@router.get("/", response_model=SchoolListResponse)
async def list_schools(
    limit: int = 100,
    offset: int = 0,
    current_user: User = Depends(get_current_active_user),
):
    """List all schools with GIS data."""
    schools = load_schools_data()
    total = len(schools)

    # Apply pagination
    paginated = schools[offset : offset + limit]

    return SchoolListResponse(total=total, schools=[SchoolGIS(**s) for s in paginated])


@router.get("/search")
async def search_schools(
    query: str,
    limit: int = 20,
    current_user: User = Depends(get_current_active_user),
):
    """Search schools by name."""
    schools = load_schools_data()
    query_lower = query.lower()

    matches = [
        SchoolGIS(**s) for s in schools if query_lower in s.get("name", "").lower()
    ][:limit]

    return {"total": len(matches), "schools": matches}


@router.get("/alerts")
async def get_school_alerts(
    current_user: User = Depends(get_current_active_user),
):
    """Get schools with active leak alerts."""
    schools = load_schools_data()

    # Filter schools with alerts (status != normal)
    alerts = [SchoolGIS(**s) for s in schools if s.get("status", "normal") != "normal"]

    return {"total": len(alerts), "schools": alerts}
