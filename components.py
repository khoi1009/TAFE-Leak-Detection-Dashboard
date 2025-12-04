# components.py
# -*- coding: utf-8 -*-
"""
UI components for incident cards and modals.

UI UX Pro Max Design System Integration - COMPACT VERSION:
- High information density
- Minimal wasted space
- Professional and clean
"""

import logging
import pandas as pd
from dash import html
import dash_bootstrap_components as dbc
from utils import mini_progress, incident_badges, get_confidence_interpretation

log = logging.getLogger(__name__)

# Design System Constants - COMPACT
INCIDENT_CARD_STYLE = {
    "background": "linear-gradient(135deg, #1C1C1F 0%, #18181B 100%)",
    "border": "1px solid rgba(255, 255, 255, 0.1)",
    "borderRadius": "10px",
    "marginBottom": "8px",
    "transition": "all 0.2s ease",
    "overflow": "hidden",
}

ACTION_BUTTON_COMPACT = {
    "borderRadius": "6px",
    "fontWeight": "500",
    "fontSize": "0.7rem",
    "padding": "4px 8px",
    "transition": "all 0.2s ease",
}


def make_incident_card(site_id, inc, detector):
    """
    Create a COMPACT card UI component for displaying an incident.
    Redesigned for maximum information density with minimal wasted space.
    """
    event_id = inc.get("event_id", "unknown")
    subs_raw = inc.get("subscores_ui")
    log.info(f"[CARD] {event_id}: subscores_ui from inc = {subs_raw}")

    subs = inc.get("subscores_ui")
    leak_score = inc.get("leak_score_ui")

    # Ensure subs is a dict with proper structure
    if not isinstance(subs, dict):
        subs = None

    if subs is None or leak_score is None:
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

    if not isinstance(subs, dict):
        subs = {"MNF": 0, "RESIDUAL": 0, "CUSUM": 0, "AFTERHRS": 0, "BURSTBF": 0}

    vol_kl = inc.get("ui_total_volume_kL", inc.get("volume_lost_kL", 0.0))
    confidence = float(inc.get("confidence", 0))
    delta_nf = float(inc.get("max_deltaNF", 0))

    # Compact signal bars - inline horizontal
    def compact_signal(name, value, color):
        val = float(value) if value else 0
        return html.Div(
            [
                html.Span(
                    name,
                    style={
                        "fontSize": "0.6rem",
                        "color": "#888",
                        "width": "45px",
                        "display": "inline-block",
                    },
                ),
                html.Div(
                    html.Div(
                        style={
                            "width": f"{min(val, 100)}%",
                            "height": "100%",
                            "backgroundColor": color,
                            "borderRadius": "2px",
                        }
                    ),
                    style={
                        "flex": "1",
                        "height": "6px",
                        "backgroundColor": "rgba(255,255,255,0.1)",
                        "borderRadius": "2px",
                        "marginRight": "4px",
                    },
                ),
                html.Span(
                    f"{val:.0f}",
                    style={
                        "fontSize": "0.6rem",
                        "color": "#aaa",
                        "width": "20px",
                        "textAlign": "right",
                    },
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "4px",
                "marginBottom": "2px",
            },
        )

    signal_colors = {
        "MNF": "#3B82F6",
        "RESIDUAL": "#8B5CF6",
        "CUSUM": "#EC4899",
        "AFTERHRS": "#F59E0B",
        "BURSTBF": "#EF4444",
    }
    signal_bars = html.Div(
        [
            compact_signal("MNF", subs.get("MNF", 0), signal_colors["MNF"]),
            compact_signal("RES", subs.get("RESIDUAL", 0), signal_colors["RESIDUAL"]),
            compact_signal("CUM", subs.get("CUSUM", 0), signal_colors["CUSUM"]),
            compact_signal("AFT", subs.get("AFTERHRS", 0), signal_colors["AFTERHRS"]),
            compact_signal("BUR", subs.get("BURSTBF", 0), signal_colors["BURSTBF"]),
        ],
        style={
            "padding": "6px 8px",
            "backgroundColor": "rgba(0,0,0,0.2)",
            "borderRadius": "6px",
        },
    )

    # Pattern match indicator - compact
    pattern_indicator = None
    if inc.get("suppressed_by_pattern"):
        pattern_indicator = html.Div(
            [
                html.Span("🚫", style={"fontSize": "0.85rem"}),
                html.Span(
                    " SUPPRESSED ",
                    style={
                        "fontWeight": "600",
                        "color": "#94A3B8",
                        "fontSize": "0.65rem",
                    },
                ),
                html.Span(
                    f"({inc.get('pattern_match_score', 0):.0%})",
                    style={"color": "#64748B", "fontSize": "0.6rem"},
                ),
            ],
            style={
                "backgroundColor": "rgba(100,116,139,0.2)",
                "borderRadius": "4px",
                "padding": "3px 6px",
                "marginBottom": "6px",
                "display": "inline-block",
            },
        )
    elif inc.get("has_pattern_match"):
        matches = inc.get("pattern_matches", [])
        top_match = matches[0] if matches else {}
        pattern_indicator = html.Div(
            [
                html.Span("🧠", style={"fontSize": "0.85rem"}),
                html.Span(
                    " POSSIBLE FALSE ALARM ",
                    style={
                        "fontWeight": "600",
                        "color": "#FCD34D",
                        "fontSize": "0.65rem",
                    },
                ),
                html.Span(
                    f"({top_match.get('final_score', 0):.0%})",
                    style={"color": "#A1A1AA", "fontSize": "0.6rem"},
                ),
            ],
            style={
                "backgroundColor": "rgba(252,211,77,0.15)",
                "borderRadius": "4px",
                "padding": "3px 6px",
                "marginBottom": "6px",
                "display": "inline-block",
            },
        )

    # Confidence color
    conf_color = (
        "#10B981" if confidence >= 80 else "#F59E0B" if confidence >= 50 else "#EF4444"
    )

    body = dbc.CardBody(
        [
            # Row 1: Event ID + Badges (very compact header)
            html.Div(
                [
                    html.Div(
                        [
                            html.Code(
                                event_id[:12],  # Truncate long IDs
                                style={
                                    "backgroundColor": "rgba(59,130,246,0.15)",
                                    "color": "#60A5FA",
                                    "padding": "1px 5px",
                                    "borderRadius": "3px",
                                    "fontSize": "0.7rem",
                                    "fontFamily": "monospace",
                                },
                            ),
                            html.Div(
                                incident_badges(inc),
                                style={
                                    "display": "inline-flex",
                                    "gap": "3px",
                                    "marginLeft": "6px",
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "flexWrap": "wrap",
                            "gap": "4px",
                        },
                    ),
                    # Select button in header
                    dbc.Button(
                        "🎯",
                        id={"type": "evt-select", "index": event_id},
                        color="link",
                        size="sm",
                        style={
                            "padding": "2px 6px",
                            "fontSize": "0.85rem",
                            "opacity": "0.7",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "marginBottom": "6px",
                    "paddingBottom": "4px",
                    "borderBottom": "1px solid rgba(255,255,255,0.05)",
                },
            ),
            # Pattern indicator (if present)
            pattern_indicator if pattern_indicator else html.Span(),
            # Row 2: Key metrics in a tight grid
            html.Div(
                [
                    # Left column: Dates
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("📅 ", style={"fontSize": "0.7rem"}),
                                    html.Span(
                                        str(
                                            pd.to_datetime(inc["start_day"]).strftime(
                                                "%m/%d"
                                            )
                                        ),
                                        style={
                                            "fontSize": "0.75rem",
                                            "color": "#E4E4E7",
                                        },
                                    ),
                                    html.Span(
                                        " → ",
                                        style={"color": "#555", "fontSize": "0.7rem"},
                                    ),
                                    html.Span(
                                        str(
                                            pd.to_datetime(inc["last_day"]).strftime(
                                                "%m/%d"
                                            )
                                        ),
                                        style={
                                            "fontSize": "0.75rem",
                                            "color": "#E4E4E7",
                                        },
                                    ),
                                ],
                                style={"marginBottom": "2px"},
                            ),
                        ],
                        style={"flex": "1"},
                    ),
                    # Right column: Key numbers
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("💧", style={"fontSize": "0.7rem"}),
                                    html.Span(
                                        f" {vol_kl:.1f}kL",
                                        style={
                                            "fontSize": "0.8rem",
                                            "fontWeight": "600",
                                            "color": "#EF4444",
                                        },
                                    ),
                                ],
                                style={"marginBottom": "1px"},
                            ),
                            html.Div(
                                [
                                    html.Span("📊", style={"fontSize": "0.65rem"}),
                                    html.Span(
                                        f" {float(leak_score):.0f}%",
                                        style={
                                            "fontSize": "0.75rem",
                                            "fontWeight": "600",
                                            "color": "#F59E0B",
                                        },
                                    ),
                                    html.Span(
                                        " / ",
                                        style={"color": "#555", "fontSize": "0.7rem"},
                                    ),
                                    html.Span(
                                        f"Δ{delta_nf:.0f}",
                                        style={
                                            "fontSize": "0.7rem",
                                            "color": "#3B82F6",
                                        },
                                    ),
                                ]
                            ),
                        ],
                        style={"textAlign": "right"},
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "flex-start",
                    "marginBottom": "6px",
                },
            ),
            # Row 3: Confidence bar (compact)
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                f"Confidence: ",
                                style={"fontSize": "0.65rem", "color": "#888"},
                            ),
                            html.Span(
                                f"{confidence:.0f}%",
                                style={
                                    "fontSize": "0.75rem",
                                    "fontWeight": "600",
                                    "color": conf_color,
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "4px",
                            "marginBottom": "3px",
                        },
                    ),
                    html.Div(
                        html.Div(
                            style={
                                "width": f"{confidence}%",
                                "height": "100%",
                                "backgroundColor": conf_color,
                                "borderRadius": "2px",
                                "transition": "width 0.3s ease",
                            }
                        ),
                        style={
                            "height": "5px",
                            "backgroundColor": "rgba(255,255,255,0.1)",
                            "borderRadius": "3px",
                            "overflow": "hidden",
                        },
                    ),
                ],
                style={"marginBottom": "6px"},
            ),
            # Row 4: Signal bars
            signal_bars,
            # Row 5: Action buttons - compact row
            html.Div(
                [
                    dbc.Button(
                        [html.Span("✓")],
                        id={
                            "type": "evt-btn",
                            "index": f"{site_id}||{event_id}||Acknowledge",
                        },
                        color="success",
                        size="sm",
                        style=ACTION_BUTTON_COMPACT,
                        title="Acknowledge",
                    ),
                    dbc.Button(
                        [html.Span("👁️")],
                        id={
                            "type": "evt-btn",
                            "index": f"{site_id}||{event_id}||Watch",
                        },
                        color="info",
                        size="sm",
                        style=ACTION_BUTTON_COMPACT,
                        title="Watch",
                    ),
                    dbc.Button(
                        [html.Span("🚨")],
                        id={
                            "type": "evt-btn",
                            "index": f"{site_id}||{event_id}||Escalate",
                        },
                        color="danger",
                        size="sm",
                        style=ACTION_BUTTON_COMPACT,
                        title="Escalate",
                    ),
                    dbc.Button(
                        [html.Span("✅")],
                        id={
                            "type": "evt-btn",
                            "index": f"{site_id}||{event_id}||Resolved",
                        },
                        color="primary",
                        size="sm",
                        outline=True,
                        style=ACTION_BUTTON_COMPACT,
                        title="Resolved",
                    ),
                    dbc.Button(
                        [html.Span("🚫")],
                        id={
                            "type": "evt-btn",
                            "index": f"{site_id}||{event_id}||Ignore",
                        },
                        color="secondary",
                        size="sm",
                        outline=True,
                        style=ACTION_BUTTON_COMPACT,
                        title="Ignore",
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "4px",
                    "marginTop": "8px",
                    "flexWrap": "wrap",
                },
            ),
            # Action status area
            html.Div(
                id={"type": "action-status", "index": event_id},
                style={"marginTop": "4px"},
            ),
        ],
        style={"padding": "10px"},
    )

    return dbc.Card(body, className="incident-card", style=INCIDENT_CARD_STYLE)
