# app.py
# -*- coding: utf-8 -*-
"""
TAFE Leak Detection - Main Dashboard Application
Complete Edition with GIS Map Integration

Runs on port 8051 (login portal on 8050, API on 8000)
Supports both authenticated and demo modes.
"""
# %%
import os
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
from urllib.parse import parse_qs

# Import modular components
from layout import create_layout
from callbacks import register_callbacks
from config import APP_TITLE

# Configuration
DEBUG = os.environ.get("DEBUG", "true").lower() == "true"
PORT = int(os.environ.get("DASHBOARD_PORT", 8051))
API_URL = os.environ.get("API_URL", "http://localhost:8000/api/v1")
LOGIN_URL = os.environ.get("LOGIN_URL", "http://localhost:8050")

# ============================================
# RESPONSIVE META TAGS (UX guideline #68)
# ============================================
meta_tags = [
    {
        "name": "viewport",
        "content": "width=device-width, initial-scale=1.0, shrink-to-fit=no",
    },
    {"name": "theme-color", "content": "#151515"},
    {"name": "description", "content": "TAFE NSW Water Leak Detection Dashboard"},
    {"name": "mobile-web-app-capable", "content": "yes"},
]

# Initialize Dash app with responsive configuration
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css",
    ],
    suppress_callback_exceptions=True,
    meta_tags=meta_tags,
    assets_folder="assets",
    compress=True,
)
app.title = APP_TITLE
server = app.server
server.secret_key = os.environ.get("SECRET_KEY", "tafe-leak-detection-dashboard-2025")

# Set layout with auth wrapper
app.layout = html.Div(
    [
        dcc.Store(id="auth-token-store", storage_type="session"),
        dcc.Store(id="user-info-store", storage_type="session"),
        dcc.Location(id="dashboard-url", refresh=False),
        html.Div(id="dashboard-content"),
    ]
)


@app.callback(
    Output("dashboard-content", "children"),
    Output("auth-token-store", "data"),
    Output("user-info-store", "data"),
    Input("dashboard-url", "href"),
    State("auth-token-store", "data"),
)
def check_authentication(href, stored_token):
    """Check authentication and render dashboard or redirect to login."""
    import requests
    from urllib.parse import urlparse, parse_qs

    # Parse URL for token
    token = None
    demo_mode = False

    if href:
        parsed = urlparse(href)
        query_params = parse_qs(parsed.query)
        token = query_params.get("token", [None])[0]
        demo_mode = query_params.get("demo", ["false"])[0].lower() == "true"

    # Use stored token if no URL token
    if not token and stored_token:
        token = stored_token

    # Demo mode - allow access without authentication
    if demo_mode or os.environ.get("DEMO_MODE", "false").lower() == "true":
        return (
            create_layout(),
            None,
            {"username": "demo", "role": "viewer", "demo": True},
        )

    # If we have a token, verify it
    if token:
        try:
            response = requests.get(
                f"{API_URL}/auth/me",
                headers={"Authorization": f"Bearer {token}"},
                timeout=5,
            )
            if response.status_code == 200:
                user_info = response.json()
                return create_layout(), token, user_info
        except:
            pass

    # No valid auth - but still allow dashboard access for now
    # In production, you might redirect to login
    return (
        create_layout(),
        None,
        {"username": "guest", "role": "viewer", "authenticated": False},
    )


# Register all callbacks
register_callbacks(app)

# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("🌊 TAFE Leak Detection Dashboard - Complete Edition")
    print("=" * 60)
    print(f"📊 Dashboard:    http://127.0.0.1:{PORT}")
    print(f"🔐 Login Portal: {LOGIN_URL}")
    print(f"🔗 API Backend:  {API_URL}")
    print("-" * 60)
    print("Demo Credentials: admin/admin123 or operator/operator123")
    print("=" * 60)
    # Note: use_reloader=False is required because the Excel file loading
    # conflicts with Werkzeug's auto-reloader on Windows
    app.run(debug=False, host="127.0.0.1", port=PORT, use_reloader=False)

# %%
