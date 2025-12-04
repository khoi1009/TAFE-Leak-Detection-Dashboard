# layout.py
# -*- coding: utf-8 -*-
"""
Dashboard layout definition - controls, tabs, and modals.

UI UX PRO MAX DESIGN SYSTEM:
- Style: Real-Time Monitoring Dashboard (#31) + Data-Dense Dashboard (#28)
- Colors: Analytics Dashboard palette (#8)
- Typography: Dashboard Data (Fira Sans + Fira Code) (#42)
- Mobile-first responsive design with touch-friendly targets

RESPONSIVE DESIGN: Mobile-first approach
- Uses xs, sm, md, lg, xl breakpoints for all layouts
- Follows UI UX Pro Max guidelines for touch targets (44px) and spacing
- Tested for: 320px, 375px, 414px, 768px, 1024px, 1440px+
"""

from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import plotly.io as pio

from config import APP_TITLE, cfg
from data import ALL_SITES, DATE_INDEX, DATE_MARKS, DEFAULT_CUTOFF_IDX
from utils import fig_placeholder, safe_read_actions, COLORS

# Set dark theme globally
pio.templates.default = "plotly_dark"


# ============================================
# DESIGN SYSTEM STYLES
# ============================================

# Common styles for cards
CARD_STYLE = {
    "background": "linear-gradient(135deg, #1C1C1F 0%, #18181B 100%)",
    "border": "1px solid rgba(255, 255, 255, 0.08)",
    "borderRadius": "16px",
}

# Touch-friendly button style
BUTTON_STYLE = {
    "minHeight": "44px",
    "minWidth": "100px",
    "borderRadius": "12px",
    "fontWeight": "500",
    "fontSize": "0.9rem",
}

# Input style
INPUT_STYLE = {
    "minHeight": "44px",
    "borderRadius": "12px",
    "backgroundColor": "#111113",
    "border": "1px solid rgba(255, 255, 255, 0.08)",
}


def create_controls():
    """
    Create the controls card at the top of the dashboard.

    RESPONSIVE BEHAVIOR:
    - Mobile (< 576px): All controls stack vertically, full-width buttons
    - Tablet (576px-991px): 2-column layout for controls
    - Desktop (992px+): 3-column layout, inline buttons

    UI UX Pro Max: Control Panel with glassmorphism effect
    """
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        # Header with icon - Professional style
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span(
                                            "🚰",
                                            style={
                                                "fontSize": "1.75rem",
                                                "marginRight": "12px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.H4(
                                                    "Controls",
                                                    className="mb-0",
                                                    style={
                                                        "fontSize": "clamp(1.125rem, 3vw, 1.375rem)",
                                                        "fontWeight": "600",
                                                        "color": "#F4F4F5",
                                                    },
                                                ),
                                                html.Small(
                                                    "Select site → Set dates → Run analysis",
                                                    style={
                                                        "color": "#71717A",
                                                        "fontSize": "0.85rem",
                                                    },
                                                ),
                                            ]
                                        ),
                                    ],
                                    style={
                                        "display": "flex",
                                        "alignItems": "center",
                                    },
                                ),
                            ],
                            className="mb-4",
                        ),
                        # Controls Row - RESPONSIVE with all breakpoints
                        dbc.Row(
                            [
                                # Site Dropdown - Full width on mobile
                                dbc.Col(
                                    [
                                        html.Label(
                                            "Site / Property",
                                            style={
                                                "fontSize": "0.8rem",
                                                "fontWeight": "500",
                                                "color": "#A1A1AA",
                                                "textTransform": "uppercase",
                                                "letterSpacing": "0.025em",
                                                "marginBottom": "8px",
                                                "display": "block",
                                            },
                                        ),
                                        dcc.Dropdown(
                                            id="site-dd",
                                            options=[
                                                {
                                                    "label": "📊 All Sites (Portfolio)",
                                                    "value": "ALL_SITES",
                                                }
                                            ]
                                            + [
                                                {"label": f"🏢 {s}", "value": s}
                                                for s in ALL_SITES
                                            ],
                                            placeholder="Select properties...",
                                            value=["ALL_SITES"],
                                            multi=True,
                                            clearable=True,
                                            persistence=True,
                                            persistence_type="session",
                                            style=INPUT_STYLE,
                                        ),
                                    ],
                                    xs=12,
                                    sm=12,
                                    md=4,
                                    lg=4,
                                    className="mb-3 mb-md-0",
                                ),
                                # Date Range - Full width on mobile
                                dbc.Col(
                                    [
                                        html.Label(
                                            "Date Range",
                                            style={
                                                "fontSize": "0.8rem",
                                                "fontWeight": "500",
                                                "color": "#A1A1AA",
                                                "textTransform": "uppercase",
                                                "letterSpacing": "0.025em",
                                                "marginBottom": "8px",
                                                "display": "block",
                                            },
                                        ),
                                        html.Div(
                                            dcc.DatePickerRange(
                                                id="overview-range",
                                                start_date=str(DATE_INDEX[0].date()),
                                                end_date=str(DATE_INDEX[-1].date()),
                                                clearable=False,
                                                display_format="DD/MM/YY",
                                                style={"width": "100%"},
                                            ),
                                            className="date-picker-responsive",
                                        ),
                                    ],
                                    xs=12,
                                    sm=6,
                                    md=5,
                                    lg=5,
                                    className="mb-3 mb-md-0 text-center",
                                ),
                                # Pause Toggle - Touch-friendly
                                dbc.Col(
                                    [
                                        html.Label(
                                            "Options",
                                            style={
                                                "fontSize": "0.8rem",
                                                "fontWeight": "500",
                                                "color": "#A1A1AA",
                                                "textTransform": "uppercase",
                                                "letterSpacing": "0.025em",
                                                "marginBottom": "8px",
                                                "display": "block",
                                            },
                                        ),
                                        dbc.Checklist(
                                            id="pause-toggle",
                                            options=[
                                                {
                                                    "label": " Pause on incident",
                                                    "value": "pause",
                                                }
                                            ],
                                            value=(
                                                ["pause"]
                                                if cfg.get("replay", {}).get(
                                                    "pause_on_incident_default", True
                                                )
                                                else []
                                            ),
                                            switch=True,
                                            style={"minHeight": "44px"},
                                        ),
                                    ],
                                    xs=12,
                                    sm=6,
                                    md=3,
                                    lg=3,
                                ),
                            ],
                            className="g-2 g-md-3 mb-4",
                        ),
                        # Action Buttons - Professional styling
                        html.Div(
                            [
                                dbc.Button(
                                    [html.Span("▶️", className="me-2"), "Run Replay"],
                                    id="btn-replay",
                                    color="primary",
                                    className="flex-grow-1 flex-md-grow-0 me-2 mb-2 mb-md-0",
                                    style={
                                        **BUTTON_STYLE,
                                        "background": "linear-gradient(135deg, #3B82F6 0%, #2563EB 100%)",
                                        "border": "none",
                                        "boxShadow": "0 4px 12px rgba(59, 130, 246, 0.3)",
                                    },
                                ),
                                dbc.Button(
                                    [html.Span("⏭️", className="me-2"), "Resume"],
                                    id="btn-resume",
                                    color="info",
                                    outline=True,
                                    className="flex-grow-1 flex-md-grow-0 me-2 mb-2 mb-md-0",
                                    style=BUTTON_STYLE,
                                ),
                                dbc.Button(
                                    [html.Span("➡️", className="me-2"), "+1 Day"],
                                    id="btn-step",
                                    color="secondary",
                                    outline=True,
                                    className="flex-grow-1 flex-md-grow-0",
                                    style=BUTTON_STYLE,
                                ),
                                html.Span(id="run-toast"),
                            ],
                            className="d-flex flex-wrap",
                            style={"gap": "8px"},
                        ),
                        # Status - Responsive text
                        html.Div(
                            id="analysis-status",
                            style={
                                "fontSize": "clamp(0.8rem, 2vw, 0.9rem)",
                                "color": "#71717A",
                                "marginTop": "16px",
                            },
                        ),
                        # Replay Log - Collapsible on mobile
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Small(
                                        "📋 Replay Log",
                                        style={
                                            "fontSize": "0.75rem",
                                            "fontWeight": "500",
                                            "color": "#71717A",
                                            "textTransform": "uppercase",
                                            "letterSpacing": "0.05em",
                                        },
                                    ),
                                    html.Pre(
                                        id="analysis-log",
                                        style={
                                            "whiteSpace": "pre-wrap",
                                            "fontFamily": "Fira Code, monospace",
                                            "margin": 0,
                                            "marginTop": "8px",
                                            "fontSize": "clamp(0.7rem, 1.5vw, 0.8rem)",
                                            "maxHeight": "100px",
                                            "overflowY": "auto",
                                            "color": "#A1A1AA",
                                        },
                                        className="replay-log",
                                    ),
                                ],
                                style={
                                    "padding": "12px 16px",
                                    "backgroundColor": "rgba(0,0,0,0.2)",
                                    "borderRadius": "12px",
                                },
                            ),
                            className="mt-3 d-none d-sm-block",
                            style={
                                "backgroundColor": "transparent",
                                "border": "1px solid rgba(255,255,255,0.06)",
                            },
                        ),
                    ]
                )
            ],
            style={"padding": "24px"},
        ),
        className="shadow-sm control-panel",
        style=CARD_STYLE,
    )


def create_overview_tab():
    """
    Create the Overview tab content.

    RESPONSIVE BEHAVIOR:
    - Mobile: 2x2 KPI grid, stacked charts
    - Tablet: 4-column KPIs, side-by-side charts
    - Desktop: Full layout with larger charts

    UI UX Pro Max: Bento Box Grid (#39) + Data-Dense Dashboard (#28)
    """
    # KPI Card style with gradient accent
    kpi_card_style = {
        "background": "linear-gradient(135deg, #1C1C1F 0%, #18181B 100%)",
        "border": "1px solid rgba(255, 255, 255, 0.08)",
        "borderRadius": "16px",
        "position": "relative",
        "overflow": "hidden",
        "transition": "all 0.2s ease",
    }

    return dbc.Tab(
        label="📊 Overview",
        tab_id="tab-overview",
        tab_style={"borderRadius": "8px"},
        label_style={"fontWeight": "500", "fontSize": "0.9rem"},
        children=[
            html.Div(style={"height": "16px"}),
            # KPI Cards Row - Professional Bento Grid
            dbc.Row(
                [
                    # Total Leaks KPI
                    dbc.Col(
                        dbc.Card(
                            [
                                # Accent line
                                html.Div(
                                    style={
                                        "position": "absolute",
                                        "top": "0",
                                        "left": "0",
                                        "right": "0",
                                        "height": "3px",
                                        "background": "linear-gradient(90deg, #EF4444, #F87171)",
                                    }
                                ),
                                dbc.CardBody(
                                    dcc.Graph(
                                        id="kpi-total-leaks",
                                        config={
                                            "displayModeBar": False,
                                            "responsive": True,
                                        },
                                        style={"height": "100px"},
                                        className="kpi-graph",
                                    ),
                                    style={"padding": "16px"},
                                ),
                            ],
                            className="kpi-card kpi-card--danger",
                            style=kpi_card_style,
                        ),
                        xs=6,
                        sm=6,
                        md=3,
                        lg=3,
                        className="mb-3",
                    ),
                    # Volume Lost KPI
                    dbc.Col(
                        dbc.Card(
                            [
                                html.Div(
                                    style={
                                        "position": "absolute",
                                        "top": "0",
                                        "left": "0",
                                        "right": "0",
                                        "height": "3px",
                                        "background": "linear-gradient(90deg, #3B82F6, #60A5FA)",
                                    }
                                ),
                                dbc.CardBody(
                                    dcc.Graph(
                                        id="kpi-volume",
                                        config={
                                            "displayModeBar": False,
                                            "responsive": True,
                                        },
                                        style={"height": "100px"},
                                        className="kpi-graph",
                                    ),
                                    style={"padding": "16px"},
                                ),
                            ],
                            className="kpi-card kpi-card--primary",
                            style=kpi_card_style,
                        ),
                        xs=6,
                        sm=6,
                        md=3,
                        lg=3,
                        className="mb-3",
                    ),
                    # Duration KPI
                    dbc.Col(
                        dbc.Card(
                            [
                                html.Div(
                                    style={
                                        "position": "absolute",
                                        "top": "0",
                                        "left": "0",
                                        "right": "0",
                                        "height": "3px",
                                        "background": "linear-gradient(90deg, #F59E0B, #FBBF24)",
                                    }
                                ),
                                dbc.CardBody(
                                    dcc.Graph(
                                        id="kpi-duration",
                                        config={
                                            "displayModeBar": False,
                                            "responsive": True,
                                        },
                                        style={"height": "100px"},
                                        className="kpi-graph",
                                    ),
                                    style={"padding": "16px"},
                                ),
                            ],
                            className="kpi-card kpi-card--warning",
                            style=kpi_card_style,
                        ),
                        xs=6,
                        sm=6,
                        md=3,
                        lg=3,
                        className="mb-3",
                    ),
                    # MNF KPI
                    dbc.Col(
                        dbc.Card(
                            [
                                html.Div(
                                    style={
                                        "position": "absolute",
                                        "top": "0",
                                        "left": "0",
                                        "right": "0",
                                        "height": "3px",
                                        "background": "linear-gradient(90deg, #06B6D4, #22D3EE)",
                                    }
                                ),
                                dbc.CardBody(
                                    dcc.Graph(
                                        id="kpi-mnf",
                                        config={
                                            "displayModeBar": False,
                                            "responsive": True,
                                        },
                                        style={"height": "100px"},
                                        className="kpi-graph",
                                    ),
                                    style={"padding": "16px"},
                                ),
                            ],
                            className="kpi-card kpi-card--info",
                            style=kpi_card_style,
                        ),
                        xs=6,
                        sm=6,
                        md=3,
                        lg=3,
                        className="mb-3",
                    ),
                ],
                className="g-3",
            ),
            # Filter and Export Row - Enhanced
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label(
                                "Filter by Category",
                                style={
                                    "fontSize": "0.8rem",
                                    "fontWeight": "500",
                                    "color": "#A1A1AA",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "0.025em",
                                    "marginBottom": "8px",
                                    "display": "block",
                                },
                            ),
                            dcc.Dropdown(
                                id="category-filter",
                                options=[],
                                value=None,
                                multi=True,
                                placeholder="Select category...",
                                disabled=False,
                                style=INPUT_STYLE,
                            ),
                        ],
                        xs=12,
                        sm=8,
                        md=4,
                        lg=4,
                        className="mb-3 mb-md-0",
                    ),
                    dbc.Col(
                        [
                            html.Label(
                                "Export Data",
                                style={
                                    "fontSize": "0.8rem",
                                    "fontWeight": "500",
                                    "color": "#A1A1AA",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "0.025em",
                                    "marginBottom": "8px",
                                    "display": "block",
                                },
                            ),
                            html.Div(
                                [
                                    dbc.Button(
                                        [
                                            html.Span("📥", className="me-2"),
                                            "Export CSV",
                                        ],
                                        id="btn-export-summary",
                                        color="success",
                                        outline=True,
                                        className="w-100",
                                        style={
                                            **BUTTON_STYLE,
                                            "borderColor": "#22C55E",
                                            "color": "#22C55E",
                                        },
                                    ),
                                    dcc.Download(id="download-summary-csv"),
                                ]
                            ),
                        ],
                        xs=12,
                        sm=4,
                        md=2,
                        lg=2,
                    ),
                ],
                className="mb-4",
            ),
            # Charts Row - Professional Card Style
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H6(
                                        "📈 Duration vs Volume",
                                        style={
                                            "margin": "0",
                                            "fontSize": "0.95rem",
                                            "fontWeight": "600",
                                            "color": "#F4F4F5",
                                        },
                                    ),
                                    style={
                                        "backgroundColor": "transparent",
                                        "borderBottom": "1px solid rgba(255,255,255,0.06)",
                                        "padding": "16px 20px",
                                    },
                                ),
                                dbc.CardBody(
                                    dcc.Graph(
                                        id="ov-scatter",
                                        figure=fig_placeholder(
                                            "", "Run analysis to populate"
                                        ),
                                        config={"responsive": True},
                                        style={"height": "280px"},
                                    ),
                                    style={"padding": "12px"},
                                ),
                            ],
                            className="chart-card",
                            style=CARD_STYLE,
                        ),
                        xs=12,
                        sm=12,
                        md=6,
                        lg=6,
                        className="mb-3 mb-md-0",
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H6(
                                        "🏷️ Count by Category",
                                        style={
                                            "margin": "0",
                                            "fontSize": "0.95rem",
                                            "fontWeight": "600",
                                            "color": "#F4F4F5",
                                        },
                                    ),
                                    style={
                                        "backgroundColor": "transparent",
                                        "borderBottom": "1px solid rgba(255,255,255,0.06)",
                                        "padding": "16px 20px",
                                    },
                                ),
                                dbc.CardBody(
                                    dcc.Graph(
                                        id="ov-bar",
                                        figure=fig_placeholder(
                                            "", "Run analysis to populate"
                                        ),
                                        config={"responsive": True},
                                        style={"height": "280px"},
                                    ),
                                    style={"padding": "12px"},
                                ),
                            ],
                            className="chart-card",
                            style=CARD_STYLE,
                        ),
                        xs=12,
                        sm=12,
                        md=6,
                        lg=6,
                    ),
                ],
                className="g-3",
            ),
            html.Div(style={"height": "24px"}),
        ],
    )


def create_events_tab():
    """
    Create the Events & Actions tab content.

    RESPONSIVE BEHAVIOR:
    - Mobile: Incident list on top, details below (stacked)
    - Tablet: Side-by-side with collapsible details
    - Desktop: Full 4+8 column layout

    UI UX Pro Max: Card-Based Interface (#22) + Split-Screen Layout (#38)
    """
    # Section header style
    section_header_style = {
        "fontSize": "1rem",
        "fontWeight": "600",
        "color": "#F4F4F5",
        "display": "flex",
        "alignItems": "center",
        "gap": "8px",
        "marginBottom": "16px",
    }

    return dbc.Tab(
        label="🚨 Events",
        tab_id="tab-events",
        tab_style={"borderRadius": "8px"},
        label_style={"fontWeight": "500", "fontSize": "0.9rem"},
        children=[
            html.Div(style={"height": "16px"}),
            dbc.Row(
                [
                    # Incident List - Sidebar style
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.Div(
                                        [
                                            html.H5(
                                                [
                                                    html.Span("📋", className="me-2"),
                                                    "Incidents",
                                                ],
                                                style=section_header_style,
                                            ),
                                            html.Span(
                                                id="incident-count-badge",
                                                className="status-badge status-badge--warning",
                                                style={
                                                    "fontSize": "0.75rem",
                                                    "padding": "4px 10px",
                                                },
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "justifyContent": "space-between",
                                            "alignItems": "center",
                                        },
                                    ),
                                    style={
                                        "backgroundColor": "transparent",
                                        "borderBottom": "1px solid rgba(255,255,255,0.06)",
                                        "padding": "16px 20px",
                                    },
                                ),
                                dbc.CardBody(
                                    dcc.Loading(
                                        id="loading-incidents",
                                        type="circle",
                                        color="#3B82F6",
                                        children=html.Div(
                                            id="incident-list",
                                            children=[
                                                dbc.Alert(
                                                    [
                                                        html.Div(
                                                            "🔍",
                                                            style={
                                                                "fontSize": "2rem",
                                                                "marginBottom": "12px",
                                                            },
                                                        ),
                                                        html.Div(
                                                            "No incidents detected",
                                                            style={
                                                                "fontWeight": "500",
                                                                "marginBottom": "4px",
                                                            },
                                                        ),
                                                        html.Small(
                                                            "Run analysis to detect leaks",
                                                            style={"opacity": "0.7"},
                                                        ),
                                                    ],
                                                    color="dark",
                                                    className="text-center",
                                                    style={
                                                        "backgroundColor": "rgba(255,255,255,0.02)",
                                                        "border": "1px dashed rgba(255,255,255,0.1)",
                                                        "borderRadius": "12px",
                                                        "padding": "32px",
                                                    },
                                                )
                                            ],
                                            className="incident-list",
                                            style={
                                                "maxHeight": "75vh",
                                                "overflowY": "auto",
                                                "paddingRight": "4px",
                                            },
                                        ),
                                    ),
                                    style={"padding": "8px"},
                                ),
                            ],
                            style=CARD_STYLE,
                        ),
                        xs=12,
                        sm=12,
                        md=5,
                        lg=5,
                        className="mb-3 mb-md-0",
                    ),
                    # Event Details Panel
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H5(
                                        [
                                            html.Span("🔍", className="me-2"),
                                            "Event Details",
                                        ],
                                        style=section_header_style,
                                    ),
                                    style={
                                        "backgroundColor": "transparent",
                                        "borderBottom": "1px solid rgba(255,255,255,0.06)",
                                        "padding": "16px 20px",
                                    },
                                ),
                                dbc.CardBody(
                                    [
                                        html.Div(
                                            id="event-detail-header", className="mb-3"
                                        ),
                                        # Metrics Row - Gauge + Evolution + Sub-signals in one row
                                        dbc.Row(
                                            [
                                                # Confidence Gauge - Compact
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                "Confidence",
                                                                style={
                                                                    "fontSize": "0.7rem",
                                                                    "color": "#71717A",
                                                                    "textTransform": "uppercase",
                                                                    "letterSpacing": "0.05em",
                                                                    "textAlign": "center",
                                                                    "marginBottom": "4px",
                                                                },
                                                            ),
                                                            dcc.Graph(
                                                                id="gauge-confidence",
                                                                config={
                                                                    "displayModeBar": False,
                                                                    "responsive": True,
                                                                },
                                                                style={
                                                                    "height": "120px",
                                                                    "width": "100%",
                                                                },
                                                            ),
                                                        ],
                                                        style={
                                                            "backgroundColor": "rgba(255,255,255,0.02)",
                                                            "border": "1px solid rgba(255,255,255,0.06)",
                                                            "borderRadius": "12px",
                                                            "padding": "12px 8px 8px 8px",
                                                            "height": "100%",
                                                        },
                                                    ),
                                                    xs=4,
                                                    sm=4,
                                                    md=3,
                                                    lg=3,
                                                    className="d-flex",
                                                ),
                                                # Confidence Evolution - Takes more space
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                "Confidence Trend",
                                                                style={
                                                                    "fontSize": "0.7rem",
                                                                    "color": "#71717A",
                                                                    "textTransform": "uppercase",
                                                                    "letterSpacing": "0.05em",
                                                                    "marginBottom": "4px",
                                                                },
                                                            ),
                                                            dcc.Graph(
                                                                id="chart-confidence-evolution",
                                                                config={
                                                                    "displayModeBar": False,
                                                                    "responsive": True,
                                                                },
                                                                style={
                                                                    "height": "120px",
                                                                    "width": "100%",
                                                                },
                                                            ),
                                                        ],
                                                        style={
                                                            "backgroundColor": "rgba(255,255,255,0.02)",
                                                            "border": "1px solid rgba(255,255,255,0.06)",
                                                            "borderRadius": "12px",
                                                            "padding": "12px 8px 8px 8px",
                                                            "height": "100%",
                                                        },
                                                    ),
                                                    xs=8,
                                                    sm=8,
                                                    md=5,
                                                    lg=5,
                                                    className="d-flex",
                                                ),
                                                # Sub-signals - Compact vertical list
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                "Detection Signals",
                                                                style={
                                                                    "fontSize": "0.7rem",
                                                                    "color": "#71717A",
                                                                    "textTransform": "uppercase",
                                                                    "letterSpacing": "0.05em",
                                                                    "marginBottom": "8px",
                                                                },
                                                            ),
                                                            html.Div(
                                                                id="detail-subscores",
                                                                style={
                                                                    "display": "flex",
                                                                    "flexDirection": "column",
                                                                    "gap": "4px",
                                                                },
                                                            ),
                                                        ],
                                                        style={
                                                            "backgroundColor": "rgba(255,255,255,0.02)",
                                                            "border": "1px solid rgba(255,255,255,0.06)",
                                                            "borderRadius": "12px",
                                                            "padding": "12px",
                                                            "height": "100%",
                                                        },
                                                    ),
                                                    xs=12,
                                                    sm=12,
                                                    md=4,
                                                    lg=4,
                                                    className="d-flex mt-2 mt-md-0",
                                                ),
                                            ],
                                            className="g-2 mb-3 align-items-stretch",
                                        ),
                                        # Detail tabs with professional styling
                                        dbc.Card(
                                            [
                                                dbc.CardHeader(
                                                    dbc.Tabs(
                                                        [
                                                            dbc.Tab(
                                                                label="📈 Timeline",
                                                                tab_id="tab-timeline",
                                                                label_style={
                                                                    "fontSize": "0.85rem",
                                                                    "padding": "10px 16px",
                                                                    "fontWeight": "500",
                                                                },
                                                            ),
                                                            dbc.Tab(
                                                                label="📊 Stats",
                                                                tab_id="tab-statistical",
                                                                label_style={
                                                                    "fontSize": "0.85rem",
                                                                    "padding": "10px 16px",
                                                                    "fontWeight": "500",
                                                                },
                                                            ),
                                                            dbc.Tab(
                                                                label="🔍 Pattern",
                                                                tab_id="tab-pattern",
                                                                label_style={
                                                                    "fontSize": "0.85rem",
                                                                    "padding": "10px 16px",
                                                                    "fontWeight": "500",
                                                                },
                                                            ),
                                                            dbc.Tab(
                                                                label="💰 Impact",
                                                                tab_id="tab-impact",
                                                                label_style={
                                                                    "fontSize": "0.85rem",
                                                                    "padding": "10px 16px",
                                                                    "fontWeight": "500",
                                                                },
                                                            ),
                                                        ],
                                                        id="detail-tabs",
                                                        active_tab="tab-timeline",
                                                        className="nav-tabs-enhanced",
                                                    ),
                                                    style={
                                                        "backgroundColor": "rgba(0,0,0,0.2)",
                                                        "padding": "0",
                                                        "borderBottom": "1px solid rgba(255,255,255,0.06)",
                                                    },
                                                ),
                                                dbc.CardBody(
                                                    html.Div(
                                                        id="detail-tabs-content",
                                                        className="tab-content",
                                                    ),
                                                    style={
                                                        "padding": "16px",
                                                        "minHeight": "350px",
                                                    },
                                                ),
                                            ],
                                            style={
                                                "backgroundColor": "rgba(255,255,255,0.02)",
                                                "border": "1px solid rgba(255,255,255,0.06)",
                                                "borderRadius": "12px",
                                            },
                                        ),
                                    ],
                                    style={"padding": "16px"},
                                ),
                            ],
                            style=CARD_STYLE,
                        ),
                        xs=12,
                        sm=12,
                        md=7,
                        lg=7,
                        className="event-detail-panel",
                    ),
                ],
                className="g-3 mb-4",
            ),
        ],
    )


def create_log_tab():
    """
    Create the Action Log tab content.

    RESPONSIVE: Table scrolls horizontally on mobile (UX guideline #71)

    UI UX Pro Max: Data Table Patterns (#24) + Zebra Striping for readability
    """
    return dbc.Tab(
        label="📝 Log",
        tab_id="tab-log",
        tab_style={"borderRadius": "8px"},
        label_style={"fontWeight": "500", "fontSize": "0.9rem"},
        children=[
            html.Div(style={"height": "16px"}),
            dbc.Card(
                [
                    dbc.CardHeader(
                        html.Div(
                            [
                                html.H5(
                                    [
                                        html.Span("📋", className="me-2"),
                                        "Action Log",
                                    ],
                                    style={
                                        "margin": "0",
                                        "fontSize": "1rem",
                                        "fontWeight": "600",
                                        "color": "#F4F4F5",
                                    },
                                ),
                                html.Small(
                                    "Track all leak detection actions and resolutions",
                                    style={
                                        "color": "#71717A",
                                        "marginLeft": "auto",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "flexWrap": "wrap",
                                "gap": "8px",
                            },
                        ),
                        style={
                            "backgroundColor": "transparent",
                            "borderBottom": "1px solid rgba(255,255,255,0.06)",
                            "padding": "16px 20px",
                        },
                    ),
                    dbc.CardBody(
                        html.Div(
                            dash_table.DataTable(
                                id="action-table",
                                columns=[
                                    {"name": c.replace("_", " ").title(), "id": c}
                                    for c in [
                                        "timestamp",
                                        "site_id",
                                        "event_id",
                                        "start_day",
                                        "end_day",
                                        "status",
                                        "action",
                                        "notes",
                                    ]
                                ],
                                data=safe_read_actions()
                                .sort_values("timestamp", ascending=False)
                                .to_dict("records"),
                                page_size=10,
                                page_action="native",
                                sort_action="native",
                                filter_action="native",
                                style_table={
                                    "overflowX": "auto",
                                    "minWidth": "100%",
                                    "borderRadius": "8px",
                                    "border": "1px solid rgba(255,255,255,0.06)",
                                },
                                style_header={
                                    "backgroundColor": "rgba(59, 130, 246, 0.1)",
                                    "fontWeight": "600",
                                    "fontSize": "0.8rem",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "0.025em",
                                    "color": "#E4E4E7",
                                    "borderBottom": "2px solid rgba(59, 130, 246, 0.3)",
                                    "padding": "12px 16px",
                                },
                                style_cell={
                                    "backgroundColor": "#18181B",
                                    "border": "none",
                                    "borderBottom": "1px solid rgba(255,255,255,0.04)",
                                    "color": "#D4D4D8",
                                    "fontSize": "0.85rem",
                                    "fontFamily": "Fira Sans, system-ui, sans-serif",
                                    "padding": "12px 16px",
                                    "minWidth": "100px",
                                    "maxWidth": "250px",
                                    "overflow": "hidden",
                                    "textOverflow": "ellipsis",
                                    "textAlign": "left",
                                },
                                style_data_conditional=[
                                    {
                                        "if": {"row_index": "odd"},
                                        "backgroundColor": "rgba(255,255,255,0.02)",
                                    },
                                    {
                                        "if": {"state": "selected"},
                                        "backgroundColor": "rgba(59, 130, 246, 0.15)",
                                        "border": "1px solid rgba(59, 130, 246, 0.3)",
                                    },
                                    {
                                        "if": {"state": "active"},
                                        "backgroundColor": "rgba(59, 130, 246, 0.2)",
                                        "border": "1px solid rgba(59, 130, 246, 0.4)",
                                    },
                                    # Status-based coloring
                                    {
                                        "if": {
                                            "filter_query": '{status} = "resolved"',
                                            "column_id": "status",
                                        },
                                        "color": "#22C55E",
                                        "fontWeight": "500",
                                    },
                                    {
                                        "if": {
                                            "filter_query": '{status} = "pending"',
                                            "column_id": "status",
                                        },
                                        "color": "#F59E0B",
                                        "fontWeight": "500",
                                    },
                                    {
                                        "if": {
                                            "filter_query": '{status} = "investigating"',
                                            "column_id": "status",
                                        },
                                        "color": "#3B82F6",
                                        "fontWeight": "500",
                                    },
                                ],
                                style_filter={
                                    "backgroundColor": "#1C1C1F",
                                    "color": "#E4E4E7",
                                    "border": "1px solid rgba(255,255,255,0.1)",
                                },
                            ),
                            className="data-table-professional",
                        ),
                        style={"padding": "16px"},
                    ),
                ],
                style=CARD_STYLE,
            ),
            html.Div(id="action-log-refresh"),
            html.Div(style={"height": "24px"}),
            # Recorded Patterns Section
            dbc.Card(
                [
                    dbc.CardHeader(
                        html.Div(
                            [
                                html.H5(
                                    [
                                        html.Span("🧠", className="me-2"),
                                        "Recorded False Alarm Patterns",
                                    ],
                                    style={
                                        "margin": "0",
                                        "fontSize": "1rem",
                                        "fontWeight": "600",
                                        "color": "#F4F4F5",
                                    },
                                ),
                                html.Div(
                                    [
                                        dbc.Button(
                                            "🔄 Refresh",
                                            id="btn-refresh-patterns",
                                            color="secondary",
                                            size="sm",
                                            className="me-2",
                                        ),
                                    ],
                                ),
                            ],
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "flexWrap": "wrap",
                                "gap": "8px",
                            },
                        ),
                        style={
                            "backgroundColor": "transparent",
                            "borderBottom": "1px solid rgba(255,255,255,0.06)",
                            "padding": "16px 20px",
                        },
                    ),
                    dbc.CardBody(
                        [
                            html.P(
                                [
                                    "These patterns are recorded when you click ",
                                    html.Strong("'🧠 Ignore & Learn Pattern'"),
                                    " in the Ignore modal. The system uses these to detect and flag similar events in the future.",
                                ],
                                className="text-muted small mb-3",
                            ),
                            html.Div(id="patterns-table-container"),
                        ],
                        style={"padding": "16px"},
                    ),
                ],
                style=CARD_STYLE,
            ),
            html.Div(style={"height": "24px"}),
        ],
    )


def create_layout():
    """
    Create the complete dashboard layout.

    RESPONSIVE DESIGN:
    - Fluid container that adapts to all screen sizes
    - Mobile-first approach with progressive enhancement
    - Touch-friendly spacing and targets (44px minimum)
    - Tested at: 320px, 375px, 414px, 768px, 1024px, 1440px+

    UI UX Pro Max: Professional Dashboard Header + Navigation System
    """
    controls = create_controls()
    tabs = dbc.Tabs(
        [create_overview_tab(), create_events_tab(), create_log_tab()],
        id="tabs",
        active_tab="tab-overview",
        persistence=True,
        className="nav-tabs-enhanced",
    )

    return dbc.Container(
        [
            # Professional Dashboard Header
            html.Header(
                dbc.Card(
                    dbc.CardBody(
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div(
                                            [
                                                html.Span(
                                                    "💧",
                                                    style={
                                                        "fontSize": "2rem",
                                                        "marginRight": "12px",
                                                    },
                                                ),
                                                html.Div(
                                                    [
                                                        html.H1(
                                                            APP_TITLE,
                                                            style={
                                                                "fontSize": "clamp(1.1rem, 3vw, 1.5rem)",
                                                                "fontWeight": "700",
                                                                "margin": "0",
                                                                "color": "#F4F4F5",
                                                                "lineHeight": "1.2",
                                                            },
                                                        ),
                                                        html.P(
                                                            "National Water Infrastructure Monitoring System",
                                                            style={
                                                                "fontSize": "0.8rem",
                                                                "color": "#71717A",
                                                                "margin": "0",
                                                                "marginTop": "2px",
                                                            },
                                                            className="d-none d-sm-block",
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            style={
                                                "display": "flex",
                                                "alignItems": "center",
                                            },
                                        ),
                                    ],
                                    xs=12,
                                    sm=8,
                                    md=6,
                                    className="mb-2 mb-sm-0",
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            # Status indicator
                                            html.Div(
                                                [
                                                    html.Span(
                                                        "●",
                                                        style={
                                                            "color": "#22C55E",
                                                            "marginRight": "6px",
                                                            "fontSize": "0.7rem",
                                                        },
                                                    ),
                                                    html.Span(
                                                        "System Online",
                                                        style={
                                                            "fontSize": "0.75rem",
                                                            "color": "#A1A1AA",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "display": "flex",
                                                    "alignItems": "center",
                                                    "marginRight": "20px",
                                                },
                                                className="d-none d-md-flex",
                                            ),
                                            # Version badge
                                            html.Span(
                                                "v2.0 Pro",
                                                style={
                                                    "fontSize": "0.7rem",
                                                    "backgroundColor": "rgba(59, 130, 246, 0.15)",
                                                    "color": "#60A5FA",
                                                    "padding": "4px 10px",
                                                    "borderRadius": "20px",
                                                    "fontWeight": "500",
                                                },
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "justifyContent": "flex-end",
                                        },
                                    ),
                                    xs=12,
                                    sm=4,
                                    md=6,
                                    style={"textAlign": "right"},
                                ),
                            ],
                            className="align-items-center",
                        ),
                        style={"padding": "16px 20px"},
                    ),
                    style={
                        "background": "linear-gradient(135deg, rgba(28, 28, 31, 0.95) 0%, rgba(24, 24, 27, 0.95) 100%)",
                        "backdropFilter": "blur(20px)",
                        "border": "1px solid rgba(255, 255, 255, 0.06)",
                        "borderRadius": "16px",
                        "marginBottom": "20px",
                    },
                ),
                style={"marginTop": "16px"},
            ),
            # Quick instructions - collapsible on mobile
            html.Div(
                [
                    html.Span(
                        "📋 Quick Start: ",
                        style={"fontWeight": "600", "color": "#E4E4E7"},
                    ),
                    html.Span(
                        "Select a site → Set date range → Click 'Run Analysis' to detect leaks",
                        style={"color": "#A1A1AA"},
                    ),
                ],
                style={
                    "fontSize": "0.85rem",
                    "padding": "12px 16px",
                    "backgroundColor": "rgba(59, 130, 246, 0.08)",
                    "border": "1px solid rgba(59, 130, 246, 0.15)",
                    "borderRadius": "10px",
                    "marginBottom": "16px",
                },
                className="d-none d-md-block",
            ),
            controls,
            html.Div(style={"height": "16px"}),
            tabs,
            # Invisible stores
            dcc.Store(id="store-confirmed"),
            dcc.Store(id="store-selected-event"),
            dcc.Store(
                id="store-cutoff-date", data=str(DATE_INDEX[DEFAULT_CUTOFF_IDX].date())
            ),
            dcc.Store(
                id="store-replay",
                data={"current": None, "start": None, "end": None, "reported": []},
            ),
            dcc.Store(id="store-action-context"),
            # ============================================
            # ACTION MODALS - Responsive for all devices
            # Full-screen on mobile, centered on desktop
            # ============================================
            # Acknowledge Modal
            dbc.Modal(
                [
                    dbc.ModalHeader(
                        dbc.ModalTitle(
                            "✓ Acknowledge Leak",
                            style={"fontSize": "clamp(1rem, 3vw, 1.25rem)"},
                        ),
                        close_button=True,
                    ),
                    dbc.ModalBody(
                        [
                            html.P(
                                "Mark this leak as acknowledged and assign for review.",
                                className="mb-3",
                                style={"fontSize": "clamp(0.875rem, 2vw, 1rem)"},
                            ),
                            html.Div(id="modal-ack-details", className="mb-3"),
                            dbc.Label("Assigned to:", className="fw-medium"),
                            dbc.Input(
                                id="modal-ack-user",
                                placeholder="Enter name or team...",
                                className="mb-2",
                                style={"minHeight": "44px"},
                            ),
                            dbc.Label("Notes (optional):", className="fw-medium"),
                            dbc.Textarea(
                                id="modal-ack-notes",
                                placeholder="Any additional context...",
                                rows=2,
                                style={"minHeight": "88px"},
                            ),
                        ]
                    ),
                    dbc.ModalFooter(
                        [
                            dbc.Button(
                                "Cancel",
                                id="modal-ack-cancel",
                                color="secondary",
                                className="flex-grow-1 flex-md-grow-0",
                                style={"minHeight": "44px", "minWidth": "100px"},
                            ),
                            dbc.Button(
                                "✓ Confirm",
                                id="modal-ack-confirm",
                                color="success",
                                className="flex-grow-1 flex-md-grow-0",
                                style={"minHeight": "44px", "minWidth": "120px"},
                            ),
                        ],
                        className="d-flex flex-wrap gap-2",
                    ),
                ],
                id="modal-acknowledge",
                is_open=False,
                backdrop="static",
                centered=True,
                fullscreen="md-down",  # Full-screen on mobile
            ),
            # Watch Modal
            dbc.Modal(
                [
                    dbc.ModalHeader(
                        dbc.ModalTitle(
                            "👁️ Watch Leak",
                            style={"fontSize": "clamp(1rem, 3vw, 1.25rem)"},
                        ),
                        close_button=True,
                    ),
                    dbc.ModalBody(
                        [
                            html.P(
                                "Monitor this leak for additional days before taking action.",
                                className="mb-3",
                            ),
                            html.Div(id="modal-watch-details", className="mb-3"),
                            dbc.Label("Why are you watching?", className="fw-medium"),
                            dbc.Select(
                                id="modal-watch-reason",
                                options=[
                                    {
                                        "label": "Waiting for more data",
                                        "value": "waiting_data",
                                    },
                                    {
                                        "label": "Checking for maintenance activity",
                                        "value": "check_maintenance",
                                    },
                                    {
                                        "label": "Pool fill suspected",
                                        "value": "pool_fill",
                                    },
                                    {
                                        "label": "Low confidence - need confirmation",
                                        "value": "low_confidence",
                                    },
                                    {"label": "Other", "value": "other"},
                                ],
                                value="waiting_data",
                                className="mb-2",
                                style={"minHeight": "44px"},
                            ),
                            dbc.Label("Review in:", className="fw-medium"),
                            dbc.Select(
                                id="modal-watch-days",
                                options=[
                                    {"label": "1 day", "value": "1"},
                                    {"label": "2 days", "value": "2"},
                                    {"label": "3 days", "value": "3"},
                                    {"label": "1 week", "value": "7"},
                                ],
                                value="3",
                                className="mb-2",
                                style={"minHeight": "44px"},
                            ),
                            dbc.Label(
                                "Additional notes (optional):", className="fw-medium"
                            ),
                            dbc.Textarea(
                                id="modal-watch-notes",
                                placeholder="Any additional context...",
                                rows=2,
                                style={"minHeight": "88px"},
                            ),
                        ]
                    ),
                    dbc.ModalFooter(
                        [
                            dbc.Button(
                                "Cancel",
                                id="modal-watch-cancel",
                                color="secondary",
                                className="flex-grow-1 flex-md-grow-0",
                                style={"minHeight": "44px", "minWidth": "100px"},
                            ),
                            dbc.Button(
                                "👁️ Confirm",
                                id="modal-watch-confirm",
                                color="info",
                                className="flex-grow-1 flex-md-grow-0",
                                style={"minHeight": "44px", "minWidth": "120px"},
                            ),
                        ],
                        className="d-flex flex-wrap gap-2",
                    ),
                ],
                id="modal-watch",
                is_open=False,
                backdrop="static",
                centered=True,
                fullscreen="md-down",
            ),
            # Escalate Modal
            dbc.Modal(
                [
                    dbc.ModalHeader(
                        dbc.ModalTitle(
                            "🚨 Escalate Leak",
                            style={"fontSize": "clamp(1rem, 3vw, 1.25rem)"},
                        ),
                        close_button=True,
                    ),
                    dbc.ModalBody(
                        [
                            html.P(
                                "Escalate this leak to senior management or emergency services.",
                                className="mb-3",
                                style={"fontSize": "clamp(0.875rem, 2vw, 1rem)"},
                            ),
                            html.Div(id="modal-escalate-details", className="mb-3"),
                            dbc.Label("Escalate to:", className="fw-medium"),
                            dbc.Checklist(
                                id="modal-escalate-to",
                                options=[
                                    {
                                        "label": "Facilities Manager",
                                        "value": "facilities",
                                    },
                                    {"label": "Regional Manager", "value": "regional"},
                                    {"label": "Emergency Plumber", "value": "plumber"},
                                    {"label": "Property Manager", "value": "property"},
                                ],
                                value=["facilities"],
                                className="mb-2",
                                style={"minHeight": "44px"},
                            ),
                            dbc.Label("Urgency Level:", className="fw-medium"),
                            dbc.RadioItems(
                                id="modal-escalate-urgency",
                                options=[
                                    {
                                        "label": "Standard (next business day)",
                                        "value": "standard",
                                    },
                                    {
                                        "label": "Urgent (within 24 hours)",
                                        "value": "urgent",
                                    },
                                    {
                                        "label": "Emergency (immediate response)",
                                        "value": "emergency",
                                    },
                                ],
                                value="urgent",
                                className="mb-2",
                            ),
                            dbc.Label("Reason for escalation:", className="fw-medium"),
                            dbc.Textarea(
                                id="modal-escalate-notes",
                                placeholder="E.g., 'Visible flooding reported', 'Large leak with high cost'...",
                                rows=3,
                                style={"minHeight": "100px"},
                            ),
                        ]
                    ),
                    dbc.ModalFooter(
                        [
                            dbc.Button(
                                "Cancel",
                                id="modal-escalate-cancel",
                                color="secondary",
                                className="flex-grow-1 flex-md-grow-0",
                                style={"minHeight": "44px", "minWidth": "100px"},
                            ),
                            dbc.Button(
                                "🚨 Escalate",
                                id="modal-escalate-confirm",
                                color="danger",
                                className="flex-grow-1 flex-md-grow-0",
                                style={"minHeight": "44px", "minWidth": "120px"},
                            ),
                        ],
                        className="d-flex flex-wrap gap-2",
                    ),
                ],
                id="modal-escalate",
                is_open=False,
                backdrop="static",
                centered=True,
                fullscreen="md-down",
            ),
            # Resolved Modal
            dbc.Modal(
                [
                    dbc.ModalHeader(
                        dbc.ModalTitle(
                            "✅ Mark as Resolved",
                            style={"fontSize": "clamp(1rem, 3vw, 1.25rem)"},
                        ),
                        close_button=True,
                    ),
                    dbc.ModalBody(
                        [
                            html.P(
                                "Record that this leak has been fixed or confirmed as false alarm.",
                                className="mb-3",
                                style={"fontSize": "clamp(0.875rem, 2vw, 1rem)"},
                            ),
                            html.Div(id="modal-resolved-details", className="mb-3"),
                            dbc.Label("Resolution type:", className="fw-medium"),
                            dbc.Select(
                                id="modal-resolved-type",
                                options=[
                                    {"label": "Leak fixed", "value": "fixed"},
                                    {
                                        "label": "False alarm confirmed",
                                        "value": "false_alarm",
                                    },
                                    {"label": "No leak found", "value": "not_found"},
                                    {"label": "Other", "value": "other"},
                                ],
                                value="fixed",
                                className="mb-2",
                                style={"minHeight": "44px"},
                            ),
                            dbc.Label("What was found?", className="fw-medium"),
                            dbc.Select(
                                id="modal-resolved-cause",
                                options=[
                                    {"label": "Running toilet", "value": "toilet"},
                                    {"label": "Pipe leak", "value": "pipe"},
                                    {
                                        "label": "Irrigation valve",
                                        "value": "irrigation",
                                    },
                                    {"label": "Pool fill", "value": "pool"},
                                    {"label": "Tap left open", "value": "tap"},
                                    {"label": "No leak found", "value": "none"},
                                    {"label": "Other", "value": "other"},
                                ],
                                value="toilet",
                                className="mb-2",
                                style={"minHeight": "44px"},
                            ),
                            dbc.Label("Resolved by:", className="fw-medium"),
                            dbc.Select(
                                id="modal-resolved-by",
                                options=[
                                    {
                                        "label": "Maintenance team",
                                        "value": "maintenance",
                                    },
                                    {"label": "Plumber", "value": "plumber"},
                                    {"label": "Self-resolved", "value": "self"},
                                    {"label": "Other", "value": "other"},
                                ],
                                value="maintenance",
                                className="mb-2",
                                style={"minHeight": "44px"},
                            ),
                            dbc.Label("Repair cost (optional):", className="fw-medium"),
                            dbc.Input(
                                id="modal-resolved-cost",
                                placeholder="E.g., 120.50",
                                type="number",
                                min=0,
                                step=0.01,
                                className="mb-2",
                                style={"minHeight": "44px"},
                            ),
                            dbc.Label("Additional notes:", className="fw-medium"),
                            dbc.Textarea(
                                id="modal-resolved-notes",
                                placeholder="Details about the resolution...",
                                rows=2,
                                style={"minHeight": "88px"},
                            ),
                        ]
                    ),
                    dbc.ModalFooter(
                        [
                            dbc.Button(
                                "Cancel",
                                id="modal-resolved-cancel",
                                color="secondary",
                                className="flex-grow-1 flex-md-grow-0",
                                style={"minHeight": "44px", "minWidth": "100px"},
                            ),
                            dbc.Button(
                                "✅ Confirm",
                                id="modal-resolved-confirm",
                                color="primary",
                                className="flex-grow-1 flex-md-grow-0",
                                style={"minHeight": "44px", "minWidth": "120px"},
                            ),
                        ],
                        className="d-flex flex-wrap gap-2",
                    ),
                ],
                id="modal-resolved",
                is_open=False,
                backdrop="static",
                centered=True,
                fullscreen="md-down",
                scrollable=True,  # Allow scroll for long content on mobile
            ),
            # Ignore Modal - Enhanced with Pattern Recording
            dbc.Modal(
                [
                    dbc.ModalHeader(
                        dbc.ModalTitle(
                            "🚫 Ignore Leak & Record Pattern",
                            style={"fontSize": "clamp(1rem, 3vw, 1.25rem)"},
                        ),
                        close_button=True,
                    ),
                    dbc.ModalBody(
                        [
                            dbc.Alert(
                                [
                                    html.Strong("⚠️ Warning: "),
                                    "This will mark these dates as non-leak events. ",
                                    html.Br(),
                                    html.Small(
                                        "💡 Recording the pattern helps the model learn to recognize similar false alarms in the future."
                                    ),
                                ],
                                color="warning",
                                className="mb-3",
                                style={"fontSize": "clamp(0.875rem, 2vw, 1rem)"},
                            ),
                            html.Div(id="modal-ignore-details", className="mb-3"),
                            # Reason Section
                            html.Hr(className="my-3"),
                            html.H6(
                                "📋 Reason for Ignoring", className="text-info mb-2"
                            ),
                            dbc.Label(
                                "Category (required):", className="fw-medium small"
                            ),
                            dbc.Select(
                                id="modal-ignore-reason",
                                options=[
                                    {
                                        "label": "❌ False alarm - Detection error",
                                        "value": "false_alarm",
                                    },
                                    {
                                        "label": "🏊 Pool fill - Scheduled filling",
                                        "value": "pool_fill",
                                    },
                                    {
                                        "label": "🔥 Fire system test",
                                        "value": "fire_test",
                                    },
                                    {
                                        "label": "🔧 Planned maintenance",
                                        "value": "maintenance",
                                    },
                                    {
                                        "label": "📊 Data error/sensor issue",
                                        "value": "data_error",
                                    },
                                    {
                                        "label": "💧 Known temporary usage",
                                        "value": "temp_usage",
                                    },
                                    {
                                        "label": "🌿 Irrigation schedule",
                                        "value": "irrigation",
                                    },
                                    {"label": "❄️ HVAC system", "value": "hvac"},
                                    {
                                        "label": "🧹 Cleaning schedule",
                                        "value": "cleaning",
                                    },
                                    {"label": "📅 Scheduled event", "value": "event"},
                                    {"label": "❓ Other", "value": "other"},
                                ],
                                value="false_alarm",
                                className="mb-2",
                                style={"minHeight": "44px"},
                            ),
                            dbc.Label("Explanation:", className="fw-medium small mt-2"),
                            dbc.Textarea(
                                id="modal-ignore-notes",
                                placeholder="Provide detailed explanation for ignoring this leak...",
                                rows=2,
                                style={"minHeight": "70px"},
                            ),
                            # Pattern Recording Section
                            html.Hr(className="my-3"),
                            html.H6(
                                "🧠 Pattern Learning (Optional)",
                                className="text-primary mb-2",
                            ),
                            html.Small(
                                "Recording patterns helps automatically detect and flag similar events in the future.",
                                className="text-muted d-block mb-3",
                            ),
                            # Is Recurring Checkbox
                            dbc.Checkbox(
                                id="modal-ignore-is-recurring",
                                label="This is a recurring/expected event",
                                value=False,
                                className="mb-2",
                            ),
                            # Recurrence Options (collapsed by default)
                            dbc.Collapse(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            # Recurrence Type
                                            dbc.Label(
                                                "Recurrence Pattern:",
                                                className="fw-medium small",
                                            ),
                                            dbc.Select(
                                                id="modal-ignore-recurrence-type",
                                                options=[
                                                    {
                                                        "label": "Weekly (same day each week)",
                                                        "value": "weekly",
                                                    },
                                                    {
                                                        "label": "Daily (every day)",
                                                        "value": "daily",
                                                    },
                                                    {
                                                        "label": "Monthly (same date each month)",
                                                        "value": "monthly",
                                                    },
                                                    {
                                                        "label": "Yearly (same date each year)",
                                                        "value": "yearly",
                                                    },
                                                ],
                                                value="weekly",
                                                className="mb-3",
                                                style={"minHeight": "40px"},
                                            ),
                                            # Days of Week (for weekly)
                                            html.Div(
                                                [
                                                    dbc.Label(
                                                        "Expected Days:",
                                                        className="fw-medium small",
                                                    ),
                                                    dbc.Checklist(
                                                        id="modal-ignore-recurrence-days",
                                                        options=[
                                                            {
                                                                "label": "Mon",
                                                                "value": 0,
                                                            },
                                                            {
                                                                "label": "Tue",
                                                                "value": 1,
                                                            },
                                                            {
                                                                "label": "Wed",
                                                                "value": 2,
                                                            },
                                                            {
                                                                "label": "Thu",
                                                                "value": 3,
                                                            },
                                                            {
                                                                "label": "Fri",
                                                                "value": 4,
                                                            },
                                                            {
                                                                "label": "Sat",
                                                                "value": 5,
                                                            },
                                                            {
                                                                "label": "Sun",
                                                                "value": 6,
                                                            },
                                                        ],
                                                        value=[],
                                                        inline=True,
                                                        className="mb-3",
                                                        labelClassName="me-2",
                                                        inputClassName="me-1",
                                                    ),
                                                ],
                                                id="modal-ignore-days-container",
                                            ),
                                            # Time Window
                                            dbc.Label(
                                                "Expected Time Window:",
                                                className="fw-medium small",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Label(
                                                                "From:",
                                                                className="small text-muted",
                                                            ),
                                                            dbc.Input(
                                                                id="modal-ignore-time-start",
                                                                type="time",
                                                                value="00:00",
                                                                style={
                                                                    "minHeight": "40px"
                                                                },
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label(
                                                                "To:",
                                                                className="small text-muted",
                                                            ),
                                                            dbc.Input(
                                                                id="modal-ignore-time-end",
                                                                type="time",
                                                                value="06:00",
                                                                style={
                                                                    "minHeight": "40px"
                                                                },
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                ],
                                                className="mb-2 g-2",
                                            ),
                                        ]
                                    ),
                                    className="bg-dark border-secondary",
                                ),
                                id="modal-ignore-recurrence-collapse",
                                is_open=False,
                                className="mb-3",
                            ),
                            # Auto-suppress checkbox
                            html.Hr(className="my-3"),
                            html.H6(
                                "⚙️ Auto-Suppress Settings",
                                className="text-warning mb-2",
                            ),
                            dbc.Checkbox(
                                id="modal-ignore-auto-suppress",
                                label=html.Span(
                                    [
                                        html.Strong(
                                            "Auto-suppress similar events in future"
                                        ),
                                        html.Br(),
                                        html.Small(
                                            "When enabled, matching events will be automatically marked as 'Suppressed' instead of alerting.",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                value=False,
                                className="mb-2",
                            ),
                            dbc.Alert(
                                [
                                    html.I(className="bi bi-info-circle me-2"),
                                    "Auto-suppressed events can still be reviewed in the 'Suppressed' tab. ",
                                    "If a suppressed event turns out to be a real leak, you can mark it as such to improve pattern accuracy.",
                                ],
                                color="info",
                                className="mb-0 small py-2",
                                id="modal-ignore-auto-suppress-info",
                                style={"display": "none"},
                            ),
                        ]
                    ),
                    dbc.ModalFooter(
                        [
                            dbc.Button(
                                "Cancel",
                                id="modal-ignore-cancel",
                                color="secondary",
                                className="flex-grow-1 flex-md-grow-0",
                                style={"minHeight": "44px", "minWidth": "100px"},
                            ),
                            dbc.Button(
                                "🚫 Ignore Only",
                                id="modal-ignore-confirm",
                                color="dark",
                                className="flex-grow-1 flex-md-grow-0",
                                style={"minHeight": "44px", "minWidth": "120px"},
                            ),
                            dbc.Button(
                                "🧠 Ignore & Learn Pattern",
                                id="modal-ignore-confirm-with-pattern",
                                color="primary",
                                className="flex-grow-1 flex-md-grow-0",
                                style={"minHeight": "44px", "minWidth": "160px"},
                            ),
                        ],
                        className="d-flex flex-wrap gap-2",
                    ),
                ],
                id="modal-ignore",
                is_open=False,
                backdrop="static",
                centered=True,
                fullscreen="md-down",
                scrollable=True,
                size="lg",
            ),
            # Toast for notifications - Responsive positioning
            html.Div(
                dbc.Toast(
                    id="action-toast-body",
                    header="Action Recorded",
                    icon="success",
                    duration=4000,
                    is_open=False,
                    style={
                        "position": "fixed",
                        "top": "10px",
                        "right": "10px",
                        "width": "min(350px, calc(100vw - 20px))",
                        "zIndex": 9999,
                    },
                ),
                id="action-toast",
            ),
        ],
        fluid=True,
        className="pb-5 px-2 px-sm-3 px-md-4",  # Responsive padding
        style={"maxWidth": "1600px", "margin": "0 auto"},  # Max width on large screens
    )
