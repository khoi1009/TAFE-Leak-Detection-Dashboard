# backend/app/main.py
"""
TAFE Leak Detection - FastAPI Backend
Main application entry point.
"""
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from app.api import api_router
from app.core.config import settings
from app.core.database import init_db, async_session
from app.core.security import get_password_hash
from app.models.user import User


async def create_default_users():
    """Create default admin and operator users if they don't exist."""
    async with async_session() as db:
        # Check if admin exists
        result = await db.execute(select(User).where(User.username == "admin"))
        if not result.scalar_one_or_none():
            admin = User(
                email="admin@tafe.nsw.edu.au",
                username="admin",
                hashed_password=get_password_hash("admin123"),
                full_name="System Administrator",
                role="admin",
                is_active=True,
            )
            db.add(admin)

        # Check if operator exists
        result = await db.execute(select(User).where(User.username == "operator"))
        if not result.scalar_one_or_none():
            operator = User(
                email="operator@tafe.nsw.edu.au",
                username="operator",
                hashed_password=get_password_hash("operator123"),
                full_name="Leak Detection Operator",
                role="operator",
                is_active=True,
            )
            db.add(operator)

        await db.commit()
        print("✅ Default users created (admin/admin123, operator/operator123)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("🚀 Starting TAFE Leak Detection API...")
    await init_db()
    await create_default_users()
    print(f"✅ API ready at http://{settings.HOST}:{settings.PORT}")
    print(f"📚 API docs at http://{settings.HOST}:{settings.PORT}/docs")
    yield
    # Shutdown
    print("👋 Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="API for TAFE NSW Leak Detection Dashboard",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
