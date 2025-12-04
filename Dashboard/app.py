# app.py
# -*- coding: utf-8 -*-
"""
Main Dash application entry point.
Modularized version of leak_detection_dashboard_realtime_simulation_all_schools.py

Responsive Design: Mobile-first approach supporting 320px to 1440px+
Based on UI UX Pro Max guidelines for nationwide deployment.
"""

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

# Import modular components
from layout import create_layout
from callbacks import register_callbacks
from config import APP_TITLE

# ============================================
# RESPONSIVE META TAGS (UX guideline #68)
# ============================================
meta_tags = [
    # Viewport for mobile devices - CRITICAL for responsiveness
    {
        "name": "viewport",
        "content": "width=device-width, initial-scale=1.0, shrink-to-fit=no",
    },
    # Theme color for mobile browsers
    {"name": "theme-color", "content": "#151515"},
    # App description
    {
        "name": "description",
        "content": "TAFE NSW Water Leak Detection Dashboard - Monitor and manage water leaks across all TAFE properties",
    },
    # Mobile web app capable
    {"name": "mobile-web-app-capable", "content": "yes"},
    # Apple mobile web app
    {"name": "apple-mobile-web-app-capable", "content": "yes"},
    {"name": "apple-mobile-web-app-status-bar-style", "content": "black-translucent"},
]

# ============================================
# Initialize Dash app with responsive configuration
# ============================================
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    meta_tags=meta_tags,
    # Assets folder for custom CSS (responsive.css)
    assets_folder="assets",
    # Compress responses for faster mobile loading
    compress=True,
)

app.title = APP_TITLE
server = app.server

# Set layout with skip link for accessibility (UX guideline #45)
app.layout = html.Div(
    [
        # Skip link for keyboard navigation
        html.A(
            "Skip to main content",
            href="#main-content",
            className="skip-link sr-only-focusable",
            style={
                "position": "absolute",
                "top": "-40px",
                "left": "0",
                "background": "#3B82F6",
                "color": "white",
                "padding": "8px 16px",
                "zIndex": "9999",
                "transition": "top 0.2s ease",
            },
        ),
        # Main content wrapper
        html.Main(
            id="main-content",
            children=create_layout(),
            role="main",
        ),
    ]
)

# Register all callbacks
register_callbacks(app)

# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    app.run_server(debug=True, host="127.0.0.1", port=8050)
