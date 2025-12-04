# components.py
# -*- coding: utf-8 -*-
"""
UI components for incident cards and modals.

UI UX Pro Max Design System Integration:
- Card-Based Interface (#22)
- Status Indicators (#14)
- Action Buttons with proper hierarchy
- Professional typography and spacing
"""

import logging
import pandas as pd
from dash import html
import dash_bootstrap_components as dbc
from utils import mini_progress, incident_badges, get_confidence_interpretation

log = logging.getLogger(__name__)

# Design System Constants
INCIDENT_CARD_STYLE = {
    "background": "linear-gradient(135deg, #1C1C1F 0%, #18181B 100%)",
    "border": "1px solid rgba(255, 255, 255, 0.08)",
    "borderRadius": "16px",
    "marginBottom": "12px",
    "transition": "all 0.2s ease",
    "overflow": "hidden",
}

ACTION_BUTTON_PRIMARY = {
    "borderRadius": "8px",
    "fontWeight": "500",
    "fontSize": "0.8rem",
    "padding": "8px 14px",
    "transition": "all 0.2s ease",
}

ACTION_BUTTON_SECONDARY = {
    "borderRadius": "8px",
    "fontWeight": "500",
    "fontSize": "0.75rem",
    "padding": "6px 12px",
    "transition": "all 0.2s ease",
}


def make_incident_card(site_id, inc, detector):
    """
    Create a card UI component for displaying an incident.

    Args:
        site_id: Property ID
        inc: Incident dictionary
        detector: SchoolLeakDetector instance

    Returns:
        dbc.Card component
    """
    # DEBUG: Log what we receive
    event_id = inc.get("event_id", "unknown")
    subs_raw = inc.get("subscores_ui")
    log.info(f"[CARD] {event_id}: subscores_ui from inc = {subs_raw}")
    """
    Create a card UI component for displaying an incident.

    Args:
        site_id: Property ID
        inc: Incident dictionary
        detector: SchoolLeakDetector instance

    Returns:
        dbc.Card component
    """
    subs = inc.get("subscores_ui")
    leak_score = inc.get("leak_score_ui")

    # Ensure subs is a dict with proper structure
    if not isinstance(subs, dict):
        subs = None

    if subs is None or leak_score is None:
        # Try to get values from detector
        try:
            if (
                detector
                and hasattr(detector, "signals_and_score")
                and hasattr(detector, "daily")
            ):
                last_day = pd.to_datetime(inc["last_day"])
                if detector.daily is not None and last_day in detector.daily.index:
                    subs, leak_score, *_ = detector.signals_and_score(last_day)
                else:
                    subs = {
                        "MNF": 0,
                        "RESIDUAL": 0,
                        "CUSUM": 0,
                        "AFTERHRS": 0,
                        "BURSTBF": 0,
                    }
                    leak_score = 0
            else:
                subs = {
                    "MNF": 0,
                    "RESIDUAL": 0,
                    "CUSUM": 0,
                    "AFTERHRS": 0,
                    "BURSTBF": 0,
                }
                leak_score = 0
        except Exception:
            subs = {"MNF": 0, "RESIDUAL": 0, "CUSUM": 0, "AFTERHRS": 0, "BURSTBF": 0}
            leak_score = 0

    # Ensure subs is a dict (defensive)
    if not isinstance(subs, dict):
        subs = {"MNF": 0, "RESIDUAL": 0, "CUSUM": 0, "AFTERHRS": 0, "BURSTBF": 0}

    vol_kl = inc.get("ui_total_volume_kL", inc.get("volume_lost_kL", 0.0))
    event_id = inc.get("event_id", "unknown")

    sub_list = [
        mini_progress(
            "MNF",
            subs.get("MNF", 0),
            tooltip_text="Minimum Night Flow: Detects elevated flow during night hours (12am-4am).",
            tooltip_id=f"tt-mnf-{event_id}",
        ),
        mini_progress(
            "RESIDUAL",
            subs.get("RESIDUAL", 0),
            tooltip_text="Residual Analysis: Compares actual after-hours flow to expected patterns.",
            tooltip_id=f"tt-res-{event_id}",
        ),
        mini_progress(
            "CUSUM",
            subs.get("CUSUM", 0),
            tooltip_text="Cumulative Sum: Statistical test detecting sustained shifts in consumption.",
            tooltip_id=f"tt-cusum-{event_id}",
        ),
        mini_progress(
            "AFTERHRS",
            subs.get("AFTERHRS", 0),
            tooltip_text="After Hours: Checks if consumption outside business hours is abnormally high.",
            tooltip_id=f"tt-afthr-{event_id}",
        ),
        mini_progress(
            "BURST/BF",
            subs.get("BURSTBF", 0),
            tooltip_text="Burst/Between-Fixture: Detects sudden spikes or erratic patterns.",
            tooltip_id=f"tt-burst-{event_id}",
        ),
    ]

    body = dbc.CardBody(
        [
            # Card Header with Event ID
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "🔎",
                                style={"fontSize": "1.2rem", "marginRight": "8px"},
                            ),
                            html.Span(
                                "Event ",
                                style={
                                    "color": "#A1A1AA",
                                    "fontSize": "0.9rem",
                                    "fontWeight": "500",
                                },
                            ),
                            html.Code(
                                inc["event_id"],
                                style={
                                    "backgroundColor": "rgba(59, 130, 246, 0.15)",
                                    "color": "#60A5FA",
                                    "padding": "2px 8px",
                                    "borderRadius": "4px",
                                    "fontSize": "0.85rem",
                                    "fontFamily": "Fira Code, monospace",
                                },
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center"},
                    ),
                    html.Div(incident_badges(inc)),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "flexWrap": "wrap",
                    "gap": "8px",
                    "marginBottom": "16px",
                    "paddingBottom": "12px",
                    "borderBottom": "1px solid rgba(255,255,255,0.06)",
                },
            ),
            # Date Information Row
            html.Div(
                [
                    html.Div(
                        [
                            html.Small(
                                "Start",
                                style={
                                    "color": "#71717A",
                                    "fontSize": "0.7rem",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "0.03em",
                                },
                            ),
                            html.Div(
                                str(pd.to_datetime(inc["start_day"]).date()),
                                style={
                                    "fontWeight": "600",
                                    "fontSize": "0.9rem",
                                    "color": "#E4E4E7",
                                },
                            ),
                        ],
                        style={"flex": "1", "textAlign": "center"},
                    ),
                    html.Div(
                        style={
                            "width": "1px",
                            "height": "30px",
                            "backgroundColor": "rgba(255,255,255,0.1)",
                        }
                    ),
                    html.Div(
                        [
                            html.Small(
                                "Last",
                                style={
                                    "color": "#71717A",
                                    "fontSize": "0.7rem",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "0.03em",
                                },
                            ),
                            html.Div(
                                str(pd.to_datetime(inc["last_day"]).date()),
                                style={
                                    "fontWeight": "600",
                                    "fontSize": "0.9rem",
                                    "color": "#E4E4E7",
                                },
                            ),
                        ],
                        style={"flex": "1", "textAlign": "center"},
                    ),
                    html.Div(
                        style={
                            "width": "1px",
                            "height": "30px",
                            "backgroundColor": "rgba(255,255,255,0.1)",
                        }
                    ),
                    html.Div(
                        [
                            html.Small(
                                "Alert",
                                style={
                                    "color": "#71717A",
                                    "fontSize": "0.7rem",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "0.03em",
                                },
                            ),
                            html.Div(
                                str(
                                    pd.to_datetime(
                                        inc.get("alert_date", inc["last_day"])
                                    ).date()
                                ),
                                style={
                                    "fontWeight": "600",
                                    "fontSize": "0.9rem",
                                    "color": "#F59E0B",
                                },
                            ),
                        ],
                        style={"flex": "1", "textAlign": "center"},
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "backgroundColor": "rgba(0,0,0,0.2)",
                    "borderRadius": "10px",
                    "padding": "12px",
                    "marginBottom": "16px",
                },
            ),
            # Key Metrics Row
            html.Div(
                [
                    # Delta NF
                    html.Div(
                        [
                            html.Small(
                                [
                                    "Max ΔNF (L/h) ",
                                    html.Span(
                                        "ℹ️",
                                        id=f"tooltip-deltanf-{inc['event_id']}",
                                        style={"cursor": "help", "fontSize": "0.65rem"},
                                    ),
                                ],
                                style={
                                    "color": "#71717A",
                                    "fontSize": "0.7rem",
                                },
                            ),
                            dbc.Tooltip(
                                "Delta Night Flow: Increase in night-time water flow above normal baseline.",
                                target=f"tooltip-deltanf-{inc['event_id']}",
                                placement="top",
                            ),
                            html.Div(
                                f"{float(inc.get('max_deltaNF',0)):.1f}",
                                style={
                                    "fontSize": "1.1rem",
                                    "fontWeight": "700",
                                    "color": "#3B82F6",
                                },
                            ),
                        ],
                        style={"textAlign": "center", "flex": "1"},
                    ),
                    # Leak Score
                    html.Div(
                        [
                            html.Small(
                                [
                                    "Leak Score ",
                                    html.Span(
                                        "ℹ️",
                                        id=f"tooltip-score-{inc['event_id']}",
                                        style={"cursor": "help", "fontSize": "0.65rem"},
                                    ),
                                ],
                                style={
                                    "color": "#71717A",
                                    "fontSize": "0.7rem",
                                },
                            ),
                            dbc.Tooltip(
                                "Combined severity score from all 5 detection signals.",
                                target=f"tooltip-score-{inc['event_id']}",
                                placement="top",
                            ),
                            html.Div(
                                f"{float(leak_score):.0f}%",
                                style={
                                    "fontSize": "1.1rem",
                                    "fontWeight": "700",
                                    "color": "#F59E0B",
                                },
                            ),
                        ],
                        style={"textAlign": "center", "flex": "1"},
                    ),
                    # Volume Lost
                    html.Div(
                        [
                            html.Small(
                                [
                                    "Volume (kL) ",
                                    html.Span(
                                        "ℹ️",
                                        id=f"tooltip-vol-{inc['event_id']}",
                                        style={"cursor": "help", "fontSize": "0.65rem"},
                                    ),
                                ],
                                style={
                                    "color": "#71717A",
                                    "fontSize": "0.7rem",
                                },
                            ),
                            dbc.Tooltip(
                                "Total water wasted since leak started. 1 kL = 1,000 liters.",
                                target=f"tooltip-vol-{inc['event_id']}",
                                placement="top",
                            ),
                            html.Div(
                                f"{float(vol_kl):.1f}",
                                style={
                                    "fontSize": "1.1rem",
                                    "fontWeight": "700",
                                    "color": "#EF4444",
                                },
                            ),
                        ],
                        style={"textAlign": "center", "flex": "1"},
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "8px",
                    "marginBottom": "16px",
                },
            ),
            # Confidence Progress Bar
            html.Div(
                [
                    html.Div(
                        [
                            html.Small(
                                [
                                    "Confidence ",
                                    html.Span(
                                        "ℹ️",
                                        id=f"tooltip-conf-{inc['event_id']}",
                                        style={"cursor": "help", "fontSize": "0.65rem"},
                                    ),
                                ],
                                style={"color": "#A1A1AA", "fontSize": "0.75rem"},
                            ),
                            dbc.Tooltip(
                                "System's certainty this is a real leak (not false alarm).",
                                target=f"tooltip-conf-{inc['event_id']}",
                                placement="top",
                            ),
                            html.Span(
                                f"{float(inc.get('confidence', 0)):.0f}%",
                                style={
                                    "fontWeight": "600",
                                    "color": "#E4E4E7",
                                    "marginLeft": "8px",
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "marginBottom": "6px",
                        },
                    ),
                    dbc.Progress(
                        value=float(inc.get("confidence", 0)),
                        style={
                            "height": "8px",
                            "borderRadius": "4px",
                            "backgroundColor": "rgba(255,255,255,0.1)",
                        },
                        className="confidence-progress",
                    ),
                ],
                style={"marginBottom": "12px"},
            ),
            # Confidence interpretation alert
            html.Div(
                [
                    dbc.Alert(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        get_confidence_interpretation(
                                            inc.get("confidence", 0)
                                        )[3],
                                        style={"marginRight": "6px"},
                                    ),
                                    html.Strong(
                                        get_confidence_interpretation(
                                            inc.get("confidence", 0)
                                        )[1],
                                        style={"marginRight": "4px"},
                                    ),
                                ],
                                style={"marginBottom": "4px"},
                            ),
                            html.Small(
                                get_confidence_interpretation(inc.get("confidence", 0))[
                                    2
                                ],
                                style={"lineHeight": "1.3", "opacity": "0.9"},
                            ),
                        ],
                        color=get_confidence_interpretation(inc.get("confidence", 0))[
                            0
                        ],
                        className="py-2 px-3",
                        style={
                            "fontSize": "0.8rem",
                            "borderRadius": "10px",
                            "border": "none",
                        },
                    ),
                ],
                style={"marginBottom": "16px"},
            ),
            # Sub-signals Section
            html.Div(
                [
                    html.Small(
                        "Detection Signals",
                        style={
                            "color": "#71717A",
                            "fontSize": "0.7rem",
                            "textTransform": "uppercase",
                            "letterSpacing": "0.05em",
                            "marginBottom": "8px",
                            "display": "block",
                        },
                    ),
                    html.Div(sub_list),
                ],
                style={
                    "backgroundColor": "rgba(0,0,0,0.15)",
                    "borderRadius": "10px",
                    "padding": "12px",
                    "marginBottom": "16px",
                },
            ),
            # Action Status
            html.Div(
                id={"type": "action-status", "index": inc["event_id"]},
                style={"marginBottom": "12px"},
            ),
            # Primary Action Buttons
            html.Div(
                [
                    dbc.Button(
                        [html.Span("🎯", className="me-1"), "Select"],
                        id={"type": "evt-select", "index": inc["event_id"]},
                        color="light",
                        size="sm",
                        style={
                            **ACTION_BUTTON_PRIMARY,
                            "backgroundColor": "rgba(255,255,255,0.1)",
                            "border": "1px solid rgba(255,255,255,0.15)",
                            "color": "#E4E4E7",
                        },
                    ),
                    dbc.Button(
                        [html.Span("✓", className="me-1"), "Acknowledge"],
                        id={
                            "type": "evt-btn",
                            "index": f"{site_id}||{inc['event_id']}||Acknowledge",
                        },
                        color="success",
                        size="sm",
                        style=ACTION_BUTTON_PRIMARY,
                    ),
                    dbc.Button(
                        [html.Span("👁️", className="me-1"), "Watch"],
                        id={
                            "type": "evt-btn",
                            "index": f"{site_id}||{inc['event_id']}||Watch",
                        },
                        color="info",
                        size="sm",
                        style=ACTION_BUTTON_PRIMARY,
                    ),
                    dbc.Button(
                        [html.Span("🚨", className="me-1"), "Escalate"],
                        id={
                            "type": "evt-btn",
                            "index": f"{site_id}||{inc['event_id']}||Escalate",
                        },
                        color="danger",
                        size="sm",
                        style=ACTION_BUTTON_PRIMARY,
                    ),
                ],
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "gap": "8px",
                    "marginBottom": "10px",
                },
            ),
            # Secondary Action Buttons
            html.Div(
                [
                    dbc.Button(
                        [html.Span("✅", className="me-1"), "Resolved"],
                        id={
                            "type": "evt-btn",
                            "index": f"{site_id}||{inc['event_id']}||Resolved",
                        },
                        color="primary",
                        size="sm",
                        outline=True,
                        style=ACTION_BUTTON_SECONDARY,
                    ),
                    dbc.Button(
                        [html.Span("🚫", className="me-1"), "Ignore"],
                        id={
                            "type": "evt-btn",
                            "index": f"{site_id}||{inc['event_id']}||Ignore",
                        },
                        color="secondary",
                        size="sm",
                        outline=True,
                        style=ACTION_BUTTON_SECONDARY,
                    ),
                ],
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "gap": "8px",
                },
            ),
        ],
        style={"padding": "20px"},
    )
    return dbc.Card(body, className="incident-card", style=INCIDENT_CARD_STYLE)
