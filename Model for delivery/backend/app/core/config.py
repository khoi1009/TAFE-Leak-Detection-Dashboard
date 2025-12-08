# backend/app/core/config.py
"""
Application Configuration
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    APP_NAME: str = "TAFE Leak Detection API"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = True

    # Server
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # Security
    SECRET_KEY: str = "tafe-leak-detection-secret-key-2025-production-ready"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./leak_detection.db"

    # CORS
    CORS_ORIGINS: list = [
        "http://localhost:8050",
        "http://127.0.0.1:8050",
        "http://localhost:8051",
        "http://127.0.0.1:8051",
    ]

    # Dashboard
    DASHBOARD_URL: str = "http://localhost:8050"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
