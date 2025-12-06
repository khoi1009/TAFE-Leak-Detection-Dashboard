"""
GIS Map Tab Component for NSW Schools Leak Detection Dashboard
Provides an interactive map tab showing all school leak statuses
Supports toggling between "All Schools" and "Leak Alerts Only" views
"""

import json
import os
import random
import pandas as pd
from typing import Dict, List, Optional, Tuple

import dash_leaflet as dl
from dash import dcc, html, callback, Output, Input, State
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
    "investigating": "#8B5CF6",  # Purple - for currently investigated
}

# =============================================================================
# PROPERTY-TO-SCHOOL MAPPING
# =============================================================================

_PROPERTY_SCHOOL_MAP = None  # Cache for property-to-school mapping


def load_property_school_mapping() -> Dict[str, Dict]:
    """Load the property-to-school mapping from CSV"""
    global _PROPERTY_SCHOOL_MAP

    if _PROPERTY_SCHOOL_MAP is not None:
        return _PROPERTY_SCHOOL_MAP

    # Try demo mapping first (faster), then full mapping
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "demo_school_mapping.csv"),
        os.path.join(os.path.dirname(__file__), "property_school_mapping.csv"),
    ]

    mapping_path = None
    for path in possible_paths:
        if os.path.exists(path):
            mapping_path = path
            break

    if mapping_path is None:
        print(f"Warning: No property-school mapping found")
        _PROPERTY_SCHOOL_MAP = {}
        return _PROPERTY_SCHOOL_MAP

    try:
        df = pd.read_csv(mapping_path)
        _PROPERTY_SCHOOL_MAP = {}
        for _, row in df.iterrows():
            _PROPERTY_SCHOOL_MAP[row["property_id"]] = {
                "school_code": str(row["school_code"]),
                "school_name": row["school_name"],
                "suburb": row["suburb"],
                "postcode": str(row["postcode"]),
                "region": row["region"],
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "school_type": row.get("school_type", "Unknown"),
                "enrolment": row.get("enrolment", 0),
            }
        print(f"Loaded {len(_PROPERTY_SCHOOL_MAP)} property-to-school mappings")
    except Exception as e:
        print(f"Error loading property-school mapping: {e}")
        _PROPERTY_SCHOOL_MAP = {}

    return _PROPERTY_SCHOOL_MAP


def get_school_for_property(property_id: str) -> Optional[Dict]:
    """Get school info for a given property ID"""
    mapping = load_property_school_mapping()
    return mapping.get(property_id)


# =============================================================================
# GIS DATA LOADING
# =============================================================================


def get_gis_data_path() -> str:
    """Get the path to the GIS JSON file"""
    possible_paths = [
        # Demo data (50 schools) for fast loading
        os.path.join(os.path.dirname(__file__), "demo_schools_gis.json"),
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


def get_leak_schools_from_incidents(incidents: List[Dict]) -> List[Dict]:
    """
    Convert leak incidents from analysis to school markers.

    Args:
        incidents: List of incident dicts from the leak detection analysis
                   Each incident has site_id (property ID), status, confidence, etc.

    Returns:
        List of school dicts with leak information for mapping
    """
    if not incidents:
        return []

    mapping = load_property_school_mapping()
    leak_schools = []

    # Group incidents by property/site
    property_incidents = {}
    for inc in incidents:
        site_id = inc.get("site_id", "")
        if site_id not in property_incidents:
            property_incidents[site_id] = []
        property_incidents[site_id].append(inc)

    for property_id, prop_incidents in property_incidents.items():
        school_info = mapping.get(property_id)
        if not school_info:
            continue

        # Get the most severe/recent incident for this property
        # Sort by confidence and date
        sorted_incs = sorted(
            prop_incidents,
            key=lambda x: (
                x.get("confidence", 0),
                str(x.get("last_day", x.get("start_day", ""))),
            ),
            reverse=True,
        )
        latest_inc = sorted_incs[0] if sorted_incs else {}

        # Determine status based on incident status
        inc_status = latest_inc.get("status", "unknown")
        if inc_status in ["unconfirmed", "confirmed", "new"]:
            leak_status = "leak"
        elif inc_status == "investigating":
            leak_status = "investigating"
        elif inc_status in ["dismissed", "false_positive"]:
            leak_status = "warning"
        else:
            leak_status = (
                "critical" if latest_inc.get("confidence", 0) > 0.8 else "leak"
            )

        school = {
            "school_code": school_info["school_code"],
            "school_name": school_info["school_name"],
            "suburb": school_info["suburb"],
            "postcode": school_info["postcode"],
            "region": school_info["region"],
            "latitude": school_info["latitude"],
            "longitude": school_info["longitude"],
            "property_id": property_id,
            "leak_status": leak_status,
            "incident_status": inc_status,
            "confidence": latest_inc.get("confidence", 0),
            "volume_lost_kL": latest_inc.get(
                "ui_total_volume_kL", latest_inc.get("volume_lost_kL", 0)
            ),
            "event_id": latest_inc.get("event_id", ""),
            "start_day": str(latest_inc.get("start_day", "")),
            "last_day": str(latest_inc.get("last_day", "")),
            "incident_count": len(prop_incidents),
        }
        leak_schools.append(school)

    return leak_schools


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
        "investigating": "violet",
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


def create_school_marker(school: Dict, is_leak_alert: bool = False) -> dl.Marker:
    """Create a map marker for a single school

    Args:
        school: School data dictionary
        is_leak_alert: If True, show enhanced leak alert information
    """
    status = school.get("leak_status", "unknown")
    color = STATUS_COLORS.get(status, STATUS_COLORS["unknown"])

    # Basic info section
    popup_items = [
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
        html.P([html.Strong("Region: "), school["region"]], style={"margin": "3px 0"}),
    ]

    # Add leak-specific information if this is a leak alert
    if is_leak_alert and school.get("event_id"):
        popup_items.extend(
            [
                html.Hr(style={"margin": "8px 0", "borderColor": color}),
                html.Div(
                    [
                        html.Span("🚨 ", style={"fontSize": "1rem"}),
                        html.Strong("LEAK ALERT", style={"color": color}),
                    ],
                    style={"marginBottom": "5px"},
                ),
                html.P(
                    [
                        html.Strong("Property: "),
                        school.get("property_id", "N/A"),
                    ],
                    style={"margin": "3px 0"},
                ),
                html.P(
                    [
                        html.Strong("Confidence: "),
                        html.Span(
                            f"{school.get('confidence', 0):.0f}%",
                            style={"color": color, "fontWeight": "bold"},
                        ),
                    ],
                    style={"margin": "3px 0"},
                ),
                html.P(
                    [
                        html.Strong("Est. Volume Lost: "),
                        f"{school.get('volume_lost_kL', 0):.1f} kL",
                    ],
                    style={"margin": "3px 0"},
                ),
                html.P(
                    [
                        html.Strong("Period: "),
                        f"{school.get('start_day', 'N/A')[:10]} → {school.get('last_day', 'N/A')[:10]}",
                    ],
                    style={"margin": "3px 0", "fontSize": "11px"},
                ),
            ]
        )
    else:
        # Show simulated data for "All Schools" view
        popup_items.extend(
            [
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
            ]
        )

    popup_content = html.Div(
        popup_items,
        style={"minWidth": "220px", "fontSize": "12px"},
    )

    return dl.Marker(
        position=[school["latitude"], school["longitude"]],
        children=[dl.Tooltip(school["school_name"]), dl.Popup(popup_content)],
        icon=create_marker_icon(status),
    )


def create_map_legend(is_leak_view: bool = False) -> html.Div:
    """Create the map legend component

    Args:
        is_leak_view: If True, show leak-specific legend items
    """
    if is_leak_view:
        legend_items = [
            ("Leak Detected", STATUS_COLORS["leak"]),
            ("Critical", STATUS_COLORS["critical"]),
            ("Investigating", STATUS_COLORS["investigating"]),
            ("Warning/Dismissed", STATUS_COLORS["warning"]),
        ]
    else:
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


def create_map_component(
    schools: List[Dict], height: str = "500px", is_leak_view: bool = False
) -> html.Div:
    """Create the main interactive map component

    Args:
        schools: List of school dictionaries with location and status info
        height: CSS height for the map
        is_leak_view: If True, markers are from leak alerts (show enhanced popups)
    """
    # Calculate center
    if schools:
        avg_lat = sum(s["latitude"] for s in schools) / len(schools)
        avg_lng = sum(s["longitude"] for s in schools) / len(schools)
        center = [avg_lat, avg_lng]
        zoom = 6 if len(schools) > 10 else 8
    else:
        center = [-33.8688, 151.2093]  # Sydney default
        zoom = 6

    # Create markers with appropriate type
    markers = [
        create_school_marker(school, is_leak_alert=is_leak_view) for school in schools
    ]

    map_component = dl.Map(
        id="schools-leak-map",
        center=center,
        zoom=zoom,
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
            create_map_legend(is_leak_view=is_leak_view),
        ],
        style={"position": "relative", "borderRadius": "12px", "overflow": "hidden"},
    )


# =============================================================================
# MAP TAB CREATION
# =============================================================================


def create_view_toggle() -> html.Div:
    """Create the toggle buttons for switching between All Schools and Leak Alerts views"""

    button_base_style = {
        "padding": "10px 20px",
        "border": "none",
        "borderRadius": "8px",
        "fontWeight": "500",
        "fontSize": "0.9rem",
        "cursor": "pointer",
        "transition": "all 0.2s ease",
        "marginRight": "8px",
    }

    return html.Div(
        [
            dbc.ButtonGroup(
                [
                    dbc.Button(
                        [
                            html.Span("🏫", style={"marginRight": "8px"}),
                            "All Schools",
                        ],
                        id="btn-map-all-schools",
                        n_clicks=0,
                        color="primary",
                        outline=False,
                        className="map-view-btn active",
                        style={
                            **button_base_style,
                            "backgroundColor": "#3B82F6",
                            "color": "white",
                        },
                    ),
                    dbc.Button(
                        [
                            html.Span("🚨", style={"marginRight": "8px"}),
                            "Leak Alerts Only",
                        ],
                        id="btn-map-leak-alerts",
                        n_clicks=0,
                        color="danger",
                        outline=True,
                        className="map-view-btn",
                        style={
                            **button_base_style,
                            "backgroundColor": "transparent",
                            "color": "#EF4444",
                            "border": "1px solid #EF4444",
                        },
                    ),
                ],
                className="mb-3",
            ),
            # Hidden store for current view state
            dcc.Store(id="store-map-view", data="all"),
        ],
        style={"display": "flex", "alignItems": "center"},
    )


def create_leak_alerts_stats(leak_schools: List[Dict]) -> dbc.Row:
    """Create statistics cards for leak alerts view"""
    total = len(leak_schools)
    critical = len([s for s in leak_schools if s.get("leak_status") == "critical"])
    leak = len([s for s in leak_schools if s.get("leak_status") == "leak"])
    investigating = len(
        [s for s in leak_schools if s.get("leak_status") == "investigating"]
    )
    warning = len([s for s in leak_schools if s.get("leak_status") == "warning"])

    # Calculate total volume lost
    total_volume = sum(s.get("volume_lost_kL", 0) for s in leak_schools)

    card_style = {
        "background": "linear-gradient(135deg, #1C1C1F 0%, #18181B 100%)",
        "border": "1px solid rgba(255, 255, 255, 0.08)",
        "borderRadius": "12px",
    }

    cards = [
        ("Active Alerts", total, "#EF4444", "🚨"),
        ("Critical", critical, STATUS_COLORS["critical"], "⚠️"),
        ("Leak Detected", leak, STATUS_COLORS["leak"], "💧"),
        ("Investigating", investigating, STATUS_COLORS["investigating"], "🔍"),
        ("Est. Volume Lost", f"{total_volume:.0f} kL", "#F59E0B", "📊"),
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


def create_map_tab() -> dbc.Tab:
    """
    Create the GIS Map tab for the dashboard.

    UI UX Pro Max: Professional map visualization with consistent styling.
    Features toggle between "All Schools" and "Leak Alerts Only" views.
    """
    # Load and process school data for initial "All Schools" view
    schools = load_school_locations()
    schools = get_leak_status_for_schools(schools)

    return dbc.Tab(
        label="🗺️ GIS Map",
        tab_id="tab-map",
        tab_style={"borderRadius": "8px"},
        label_style={"fontWeight": "500", "fontSize": "0.9rem"},
        children=[
            html.Div(style={"height": "16px"}),
            # Header with View Toggle
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
                                                                id="map-subtitle",
                                                                children=f"Showing {len(schools):,} schools across NSW",
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
                                        md=6,
                                    ),
                                    dbc.Col(
                                        [
                                            create_view_toggle(),
                                        ],
                                        xs=12,
                                        md=6,
                                        className="text-md-end mt-3 mt-md-0",
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
            # Dynamic Statistics Cards Container
            html.Div(
                id="map-stats-container",
                children=[create_stats_cards(schools)],
            ),
            html.Div(style={"height": "16px"}),
            # Map Card with Dynamic Content
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        id="map-card-title",
                                        children=[
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
                                        id="map-card-description",
                                        children="Click on markers to view detailed leak information for each school.",
                                        style={
                                            "color": "#71717A",
                                            "fontSize": "0.85rem",
                                            "marginBottom": "16px",
                                        },
                                    ),
                                ]
                            ),
                            # Dynamic map container
                            html.Div(
                                id="map-container",
                                children=[
                                    create_map_component(
                                        schools, height="550px", is_leak_view=False
                                    )
                                ],
                            ),
                        ],
                        style={"padding": "20px"},
                    )
                ],
                style=CARD_STYLE,
            ),
            html.Div(style={"height": "24px"}),
            # Info panel for leak alerts (hidden by default)
            html.Div(
                id="leak-alerts-info-panel",
                style={"display": "none"},
                children=[
                    dbc.Alert(
                        [
                            html.H5("💡 Tip", className="alert-heading"),
                            html.P(
                                "The 'Leak Alerts Only' view shows schools with detected leaks from your analysis. "
                                "Run the leak detection analysis first to see alerts on the map.",
                                className="mb-0",
                            ),
                        ],
                        color="info",
                        className="mt-3",
                    ),
                ],
            ),
        ],
    )


# =============================================================================
# MAP CALLBACKS
# =============================================================================


def register_map_callbacks(app):
    """
    Register callbacks for the GIS Map tab.

    This function should be called from callbacks.py to set up the map toggle functionality.

    Args:
        app: The Dash application instance
    """
    from dash import callback_context
    from dash.exceptions import PreventUpdate

    @app.callback(
        [
            Output("map-container", "children"),
            Output("map-stats-container", "children"),
            Output("map-subtitle", "children"),
            Output("map-card-title", "children"),
            Output("map-card-description", "children"),
            Output("btn-map-all-schools", "style"),
            Output("btn-map-leak-alerts", "style"),
            Output("leak-alerts-info-panel", "style"),
            Output("store-map-view", "data"),
        ],
        [
            Input("btn-map-all-schools", "n_clicks"),
            Input("btn-map-leak-alerts", "n_clicks"),
        ],
        [
            State("store-confirmed", "data"),
            State("store-map-view", "data"),
        ],
        prevent_initial_call=True,
    )
    def update_map_view(all_clicks, leak_clicks, confirmed_data, current_view):
        """Handle map view toggle between All Schools and Leak Alerts"""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Define button styles
        active_style = {
            "padding": "10px 20px",
            "border": "none",
            "borderRadius": "8px",
            "fontWeight": "500",
            "fontSize": "0.9rem",
            "cursor": "pointer",
            "transition": "all 0.2s ease",
            "marginRight": "8px",
            "backgroundColor": "#3B82F6",
            "color": "white",
        }

        inactive_primary_style = {
            "padding": "10px 20px",
            "border": "1px solid #3B82F6",
            "borderRadius": "8px",
            "fontWeight": "500",
            "fontSize": "0.9rem",
            "cursor": "pointer",
            "transition": "all 0.2s ease",
            "marginRight": "8px",
            "backgroundColor": "transparent",
            "color": "#3B82F6",
        }

        active_danger_style = {
            "padding": "10px 20px",
            "border": "none",
            "borderRadius": "8px",
            "fontWeight": "500",
            "fontSize": "0.9rem",
            "cursor": "pointer",
            "transition": "all 0.2s ease",
            "marginRight": "8px",
            "backgroundColor": "#EF4444",
            "color": "white",
        }

        inactive_danger_style = {
            "padding": "10px 20px",
            "border": "1px solid #EF4444",
            "borderRadius": "8px",
            "fontWeight": "500",
            "fontSize": "0.9rem",
            "cursor": "pointer",
            "transition": "all 0.2s ease",
            "marginRight": "8px",
            "backgroundColor": "transparent",
            "color": "#EF4444",
        }

        if triggered_id == "btn-map-all-schools":
            # Show all schools view
            schools = load_school_locations()
            schools = get_leak_status_for_schools(schools)

            return (
                create_map_component(schools, height="550px", is_leak_view=False),
                create_stats_cards(schools),
                f"Showing {len(schools):,} schools across NSW",
                [
                    html.Span("📍", style={"marginRight": "8px"}),
                    "Interactive School Map",
                ],
                "Click on markers to view detailed water usage information for each school.",
                active_style,
                inactive_danger_style,
                {"display": "none"},
                "all",
            )

        elif triggered_id == "btn-map-leak-alerts":
            # Show leak alerts view
            # Get incidents from confirmed store
            incidents = confirmed_data if confirmed_data else []

            if not incidents:
                # No leak data yet - show empty state
                return (
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        "🔍",
                                        style={
                                            "fontSize": "3rem",
                                            "marginBottom": "16px",
                                        },
                                    ),
                                    html.H5(
                                        "No Leak Alerts Yet",
                                        style={
                                            "color": "#F4F4F5",
                                            "marginBottom": "8px",
                                        },
                                    ),
                                    html.P(
                                        "Run the leak detection analysis to see alerts on the map.",
                                        style={
                                            "color": "#71717A",
                                            "marginBottom": "16px",
                                        },
                                    ),
                                    html.P(
                                        "Go to the Analysis tab and use the simulation controls to detect leaks.",
                                        style={
                                            "color": "#A1A1AA",
                                            "fontSize": "0.85rem",
                                        },
                                    ),
                                ],
                                style={
                                    "textAlign": "center",
                                    "padding": "60px 20px",
                                    "backgroundColor": "rgba(255,255,255,0.02)",
                                    "borderRadius": "12px",
                                    "border": "1px dashed rgba(255,255,255,0.1)",
                                },
                            ),
                        ],
                    ),
                    html.Div(
                        [
                            dbc.Alert(
                                [
                                    html.Span("ℹ️ ", style={"marginRight": "8px"}),
                                    "No leak incidents detected. Run the analysis first.",
                                ],
                                color="secondary",
                            ),
                        ]
                    ),
                    "No leak alerts - Run analysis first",
                    [html.Span("🚨", style={"marginRight": "8px"}), "Leak Alert Map"],
                    "Showing schools with detected water leaks from your analysis.",
                    inactive_primary_style,
                    active_danger_style,
                    {"display": "block"},
                    "leaks",
                )

            # Convert incidents to school markers
            leak_schools = get_leak_schools_from_incidents(incidents)

            if not leak_schools:
                # Incidents exist but no school mapping
                return (
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        "⚠️",
                                        style={
                                            "fontSize": "3rem",
                                            "marginBottom": "16px",
                                        },
                                    ),
                                    html.H5(
                                        "No School Mapping Available",
                                        style={
                                            "color": "#F4F4F5",
                                            "marginBottom": "8px",
                                        },
                                    ),
                                    html.P(
                                        f"Found {len(incidents)} incidents but couldn't map them to schools.",
                                        style={
                                            "color": "#71717A",
                                            "marginBottom": "16px",
                                        },
                                    ),
                                    html.P(
                                        "Check the property_school_mapping.csv file.",
                                        style={
                                            "color": "#A1A1AA",
                                            "fontSize": "0.85rem",
                                        },
                                    ),
                                ],
                                style={
                                    "textAlign": "center",
                                    "padding": "60px 20px",
                                    "backgroundColor": "rgba(255,255,255,0.02)",
                                    "borderRadius": "12px",
                                    "border": "1px dashed rgba(255,255,255,0.1)",
                                },
                            ),
                        ],
                    ),
                    html.Div(
                        [
                            dbc.Alert(
                                [
                                    html.Span("⚠️ ", style={"marginRight": "8px"}),
                                    f"{len(incidents)} incidents found but no school mapping available.",
                                ],
                                color="warning",
                            ),
                        ]
                    ),
                    f"{len(incidents)} incidents - No school mapping",
                    [html.Span("🚨", style={"marginRight": "8px"}), "Leak Alert Map"],
                    "Showing schools with detected water leaks from your analysis.",
                    inactive_primary_style,
                    active_danger_style,
                    {"display": "block"},
                    "leaks",
                )

            # Show leak alerts on map
            return (
                create_map_component(leak_schools, height="550px", is_leak_view=True),
                create_leak_alerts_stats(leak_schools),
                f"Showing {len(leak_schools)} schools with leak alerts",
                [html.Span("🚨", style={"marginRight": "8px"}), "Leak Alert Map"],
                "Click on markers to view detailed leak information. Red markers indicate critical leaks.",
                inactive_primary_style,
                active_danger_style,
                {"display": "none"},
                "leaks",
            )

        raise PreventUpdate


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

    # Test property mapping
    mapping = load_property_school_mapping()
    print(f"Loaded {len(mapping)} property-to-school mappings")
