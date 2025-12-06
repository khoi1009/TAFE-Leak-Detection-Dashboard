"""
GIS Map Tab Component for NSW Schools Leak Detection Dashboard
Provides an interactive map tab showing all school leak statuses
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import dash_leaflet as dl
from dash import dcc, html
import dash_bootstrap_components as dbc

# =============================================================================
# STYLING CONSTANTS - Match UI UX Pro Max Design System
# =============================================================================

CARD_STYLE = {
    "background": "linear-gradient(135deg, #1C1C1F 0%, #18181B 100%)",
    "border": "1px solid rgba(255, 255, 255, 0.08)",
    "borderRadius": "16px",
}

# Status color mapping
STATUS_COLORS = {
    "normal": "#22C55E",  # Green
    "warning": "#F59E0B",  # Amber
    "leak": "#F97316",  # Orange
    "critical": "#EF4444",  # Red
    "unknown": "#6B7280",  # Gray
}

# =============================================================================
# GIS DATA LOADING
# =============================================================================


def get_gis_data_path() -> str:
    """Get the path to the GIS JSON file"""
    possible_paths = [
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "Production_Web_App",
            "School GIS info",
            "3e6d5f6a-055c-440d-a690-fc0537c31095.json",
        ),
        r"D:\End Use Projects\NSW - TAFE Leak detection model\Production_Web_App\School GIS info\3e6d5f6a-055c-440d-a690-fc0537c31095.json",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return possible_paths[0]


def load_school_locations() -> List[Dict]:
    """Load school locations from NSW Education GIS data"""
    try:
        gis_path = get_gis_data_path()

        with open(gis_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        fields = data.get("fields", [])
        records = data.get("records", [])

        field_map = {field["id"]: idx for idx, field in enumerate(fields)}

        schools = []
        for record in records:
            try:
                lat_str = record[field_map.get("Latitude", 41)]
                lng_str = record[field_map.get("Longitude", 42)]

                if not lat_str or not lng_str or lat_str == "np" or lng_str == "np":
                    continue

                lat = float(lat_str)
                lng = float(lng_str)

                if not (-44 < lat < -10 and 110 < lng < 160):
                    continue

                school = {
                    "school_code": record[field_map.get("School_code", 0)] or "",
                    "school_name": record[field_map.get("School_name", 2)]
                    or "Unknown School",
                    "suburb": record[field_map.get("Town_suburb", 4)] or "",
                    "postcode": record[field_map.get("Postcode", 5)] or "",
                    "region": record[field_map.get("Operational_directorate", 31)]
                    or "",
                    "latitude": lat,
                    "longitude": lng,
                }
                schools.append(school)

            except (ValueError, IndexError, TypeError):
                continue

        return schools

    except Exception as e:
        print(f"Error loading school locations: {e}")
        return []


def get_leak_status_for_schools(schools: List[Dict]) -> List[Dict]:
    """Add simulated leak status to school data"""
    random.seed(42)

    status_options = [
        ("normal", 0.70),
        ("warning", 0.15),
        ("leak", 0.10),
        ("critical", 0.05),
    ]

    for school in schools:
        r = random.random()
        cumulative = 0
        for status, prob in status_options:
            cumulative += prob
            if r < cumulative:
                school["leak_status"] = status
                break

        school["daily_usage"] = round(random.uniform(5, 50), 1)
        school["baseline"] = round(school["daily_usage"] * random.uniform(0.8, 1.0), 1)
        school["variance"] = round(
            (school["daily_usage"] - school["baseline"]) / school["baseline"] * 100, 1
        )

    return schools


# =============================================================================
# MAP COMPONENTS
# =============================================================================


def get_marker_color_name(status: str) -> str:
    """Map status to leaflet-color-markers color name"""
    mapping = {
        "normal": "green",
        "warning": "gold",
        "leak": "orange",
        "critical": "red",
        "unknown": "grey",
    }
    return mapping.get(status, "grey")


def create_marker_icon(status: str) -> dict:
    """Create a colored marker icon based on status"""
    return dict(
        iconUrl=f"https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-{get_marker_color_name(status)}.png",
        shadowUrl="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png",
        iconSize=[25, 41],
        iconAnchor=[12, 41],
        popupAnchor=[1, -34],
        shadowSize=[41, 41],
    )


def create_school_marker(school: Dict) -> dl.Marker:
    """Create a map marker for a single school"""
    status = school.get("leak_status", "unknown")
    color = STATUS_COLORS.get(status, STATUS_COLORS["unknown"])

    popup_content = html.Div(
        [
            html.H6(
                school["school_name"],
                style={"marginBottom": "5px", "color": "#333", "fontWeight": "bold"},
            ),
            html.P(
                [
                    html.Strong("Status: "),
                    html.Span(
                        status.upper(),
                        style={
                            "color": color,
                            "fontWeight": "bold",
                            "backgroundColor": f"{color}20",
                            "padding": "2px 6px",
                            "borderRadius": "3px",
                        },
                    ),
                ],
                style={"margin": "3px 0"},
            ),
            html.P(
                [
                    html.Strong("Location: "),
                    f"{school['suburb']}, {school['postcode']}",
                ],
                style={"margin": "3px 0"},
            ),
            html.P(
                [html.Strong("Region: "), school["region"]], style={"margin": "3px 0"}
            ),
            html.Hr(style={"margin": "8px 0"}),
            html.P(
                [
                    html.Strong("Daily Usage: "),
                    f"{school.get('daily_usage', 'N/A')} kL",
                ],
                style={"margin": "3px 0"},
            ),
            html.P(
                [
                    html.Strong("Variance: "),
                    html.Span(
                        f"{school.get('variance', 0):+.1f}%",
                        style={
                            "color": (
                                "red" if school.get("variance", 0) > 10 else "green"
                            )
                        },
                    ),
                ],
                style={"margin": "3px 0"},
            ),
        ],
        style={"minWidth": "200px", "fontSize": "12px"},
    )

    return dl.Marker(
        position=[school["latitude"], school["longitude"]],
        children=[dl.Tooltip(school["school_name"]), dl.Popup(popup_content)],
        icon=create_marker_icon(status),
    )


def create_map_legend() -> html.Div:
    """Create the map legend component"""
    legend_items = [
        ("Normal", STATUS_COLORS["normal"]),
        ("Warning", STATUS_COLORS["warning"]),
        ("Leak Detected", STATUS_COLORS["leak"]),
        ("Critical", STATUS_COLORS["critical"]),
    ]

    return html.Div(
        [
            html.H6(
                "Leak Status",
                style={"marginBottom": "10px", "fontWeight": "bold", "color": "#333"},
            ),
            *[
                html.Div(
                    [
                        html.Span(
                            style={
                                "display": "inline-block",
                                "width": "12px",
                                "height": "12px",
                                "backgroundColor": color,
                                "borderRadius": "50%",
                                "marginRight": "8px",
                            }
                        ),
                        html.Span(label, style={"fontSize": "12px", "color": "#333"}),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "marginBottom": "5px",
                    },
                )
                for label, color in legend_items
            ],
        ],
        style={
            "position": "absolute",
            "bottom": "30px",
            "right": "10px",
            "backgroundColor": "white",
            "padding": "10px 15px",
            "borderRadius": "8px",
            "boxShadow": "0 2px 6px rgba(0,0,0,0.3)",
            "zIndex": "1000",
        },
    )


def create_stats_cards(schools: List[Dict]) -> dbc.Row:
    """Create statistics summary cards"""
    total = len(schools)
    normal = len([s for s in schools if s.get("leak_status") == "normal"])
    warning = len([s for s in schools if s.get("leak_status") == "warning"])
    leak = len([s for s in schools if s.get("leak_status") == "leak"])
    critical = len([s for s in schools if s.get("leak_status") == "critical"])

    card_style = {
        "background": "linear-gradient(135deg, #1C1C1F 0%, #18181B 100%)",
        "border": "1px solid rgba(255, 255, 255, 0.08)",
        "borderRadius": "12px",
    }

    cards = [
        ("Total Schools", total, "#3B82F6", "🏫"),
        ("Normal", normal, STATUS_COLORS["normal"], "✅"),
        ("Warning", warning, STATUS_COLORS["warning"], "⚠️"),
        ("Leak", leak, STATUS_COLORS["leak"], "💧"),
        ("Critical", critical, STATUS_COLORS["critical"], "🚨"),
    ]

    return dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.Div(
                                        [
                                            html.Span(
                                                icon, style={"fontSize": "1.5rem"}
                                            ),
                                        ],
                                        style={"float": "right"},
                                    ),
                                    html.H3(
                                        count,
                                        className="mb-0",
                                        style={"color": color, "fontWeight": "bold"},
                                    ),
                                    html.P(
                                        label,
                                        className="mb-0",
                                        style={
                                            "color": "#A1A1AA",
                                            "fontSize": "0.8rem",
                                        },
                                    ),
                                ],
                                style={"padding": "12px"},
                            )
                        ],
                        style={**card_style, "borderLeft": f"3px solid {color}"},
                    )
                ],
                xs=6,
                sm=4,
                md=True,
                className="mb-3",
            )
            for label, count, color, icon in cards
        ],
        className="g-3",
    )


def create_map_component(schools: List[Dict], height: str = "500px") -> html.Div:
    """Create the main interactive map component"""
    # Calculate center
    if schools:
        avg_lat = sum(s["latitude"] for s in schools) / len(schools)
        avg_lng = sum(s["longitude"] for s in schools) / len(schools)
        center = [avg_lat, avg_lng]
    else:
        center = [-33.8688, 151.2093]  # Sydney default

    # Create markers
    markers = [create_school_marker(school) for school in schools]

    map_component = dl.Map(
        id="schools-leak-map",
        center=center,
        zoom=6,
        style={"width": "100%", "height": height, "borderRadius": "12px"},
        children=[
            dl.TileLayer(
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                attribution="&copy; OpenStreetMap contributors",
            ),
            dl.LayerGroup(id="school-markers", children=markers),
            dl.ScaleControl(position="bottomleft"),
        ],
    )

    return html.Div(
        [
            map_component,
            create_map_legend(),
        ],
        style={"position": "relative", "borderRadius": "12px", "overflow": "hidden"},
    )


# =============================================================================
# MAP TAB CREATION
# =============================================================================


def create_map_tab() -> dbc.Tab:
    """
    Create the GIS Map tab for the dashboard.

    UI UX Pro Max: Professional map visualization with consistent styling
    """
    # Load and process school data
    schools = load_school_locations()
    schools = get_leak_status_for_schools(schools)

    return dbc.Tab(
        label="🗺️ GIS Map",
        tab_id="tab-map",
        tab_style={"borderRadius": "8px"},
        label_style={"fontWeight": "500", "fontSize": "0.9rem"},
        children=[
            html.Div(style={"height": "16px"}),
            # Header
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    html.Span(
                                                        "🗺️",
                                                        style={
                                                            "fontSize": "1.75rem",
                                                            "marginRight": "12px",
                                                        },
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.H4(
                                                                "NSW Schools Water Monitoring Map",
                                                                className="mb-0",
                                                                style={
                                                                    "fontSize": "1.25rem",
                                                                    "fontWeight": "600",
                                                                    "color": "#F4F4F5",
                                                                },
                                                            ),
                                                            html.Small(
                                                                f"Showing {len(schools):,} schools across NSW",
                                                                style={
                                                                    "color": "#71717A"
                                                                },
                                                            ),
                                                        ]
                                                    ),
                                                ],
                                                style={
                                                    "display": "flex",
                                                    "alignItems": "center",
                                                },
                                            )
                                        ],
                                        xs=12,
                                        md=8,
                                    ),
                                    dbc.Col(
                                        [
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
                                                        "Live Data",
                                                        style={
                                                            "fontSize": "0.75rem",
                                                            "color": "#A1A1AA",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "display": "flex",
                                                    "alignItems": "center",
                                                    "justifyContent": "flex-end",
                                                },
                                            )
                                        ],
                                        xs=12,
                                        md=4,
                                        className="text-end",
                                    ),
                                ],
                                className="align-items-center",
                            )
                        ],
                        style={"padding": "16px 20px"},
                    )
                ],
                style=CARD_STYLE,
                className="mb-3",
            ),
            # Statistics Cards
            create_stats_cards(schools),
            html.Div(style={"height": "16px"}),
            # Map Card
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        [
                                            html.Span(
                                                "📍", style={"marginRight": "8px"}
                                            ),
                                            "Interactive School Map",
                                        ],
                                        style={
                                            "color": "#F4F4F5",
                                            "marginBottom": "12px",
                                        },
                                    ),
                                    html.P(
                                        "Click on markers to view detailed leak information for each school.",
                                        style={
                                            "color": "#71717A",
                                            "fontSize": "0.85rem",
                                            "marginBottom": "16px",
                                        },
                                    ),
                                ]
                            ),
                            create_map_component(schools, height="550px"),
                        ],
                        style={"padding": "20px"},
                    )
                ],
                style=CARD_STYLE,
            ),
            html.Div(style={"height": "24px"}),
        ],
    )


# For testing
if __name__ == "__main__":
    schools = load_school_locations()
    print(f"Loaded {len(schools)} schools")
    schools = get_leak_status_for_schools(schools)

    summary = {
        "total": len(schools),
        "normal": len([s for s in schools if s.get("leak_status") == "normal"]),
        "warning": len([s for s in schools if s.get("leak_status") == "warning"]),
        "leak": len([s for s in schools if s.get("leak_status") == "leak"]),
        "critical": len([s for s in schools if s.get("leak_status") == "critical"]),
    }
    print(f"Summary: {summary}")
