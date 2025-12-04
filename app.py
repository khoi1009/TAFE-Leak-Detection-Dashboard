# app.py
# -*- coding: utf-8 -*-
"""
Main Dash application entry point.
Modularized version of leak_detection_dashboard_realtime_simulation_all_schools.py

Responsive Design: Mobile-first approach supporting 320px to 1440px+
Based on UI UX Pro Max guidelines for nationwide deployment.
"""
# %%
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
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    meta_tags=meta_tags,
    assets_folder="assets",  # Load CSS from assets folder
    compress=True,
)
app.title = APP_TITLE
server = app.server

# Set layout
app.layout = create_layout()

# Register all callbacks
register_callbacks(app)

# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
