# utils.py
# -*- coding: utf-8 -*-
"""
Utility functions for charts, UI components, and data processing.

UI UX Pro Max Design System Integration:
- Analytics Dashboard color palette (#8)
- Real-Time Monitoring style (#31)
- Dashboard Data typography (Fira Sans + Fira Code)
"""

import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from dash import html
import dash_bootstrap_components as dbc
from config import ACTION_LOG, log


# ============================================
# DESIGN SYSTEM CONSTANTS
# Based on UI UX Pro Max Guidelines
# ============================================

# Color palette - Analytics Dashboard
COLORS = {
    # Primary
    "primary": "#3B82F6",
    "primary_light": "#60A5FA",
    "primary_dark": "#2563EB",
    # Status colors - Real-time monitoring
    "success": "#22C55E",
    "success_light": "#4ADE80",
    "warning": "#F59E0B",
    "warning_light": "#FBBF24",
    "danger": "#EF4444",
    "danger_light": "#F87171",
    "info": "#06B6D4",
    "info_light": "#22D3EE",
    # Background - Dark mode OLED
    "bg_base": "#0A0A0B",
    "bg_surface": "#111113",
    "bg_card": "#1C1C1F",
    "bg_hover": "#27272A",
    # Text
    "text_primary": "#F4F4F5",
    "text_secondary": "#A1A1AA",
    "text_muted": "#71717A",
    # Chart specific
    "chart_grid": "rgba(255, 255, 255, 0.06)",
    "chart_line": "#3B82F6",
    "chart_area": "rgba(59, 130, 246, 0.2)",
}

# Chart color sequence for multi-series
CHART_COLORS = [
    "#3B82F6",  # Primary blue
    "#22C55E",  # Success green
    "#F59E0B",  # Warning amber
    "#EF4444",  # Danger red
    "#06B6D4",  # Info cyan
    "#8B5CF6",  # Purple
    "#EC4899",  # Pink
    "#F97316",  # Orange
]


# ============================================
# PLOTLY THEME - Professional Dark
# ============================================


def get_chart_theme():
    """Return consistent chart theme based on UI UX Pro Max guidelines."""
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {
            "family": "Fira Sans, -apple-system, BlinkMacSystemFont, sans-serif",
            "color": COLORS["text_secondary"],
            "size": 12,
        },
        "title": {
            "font": {
                "family": "Fira Sans, sans-serif",
                "size": 16,
                "color": COLORS["text_primary"],
            },
            "x": 0,
            "xanchor": "left",
        },
        "xaxis": {
            "gridcolor": COLORS["chart_grid"],
            "linecolor": COLORS["chart_grid"],
            "tickfont": {"size": 11},
            "title_font": {"size": 12},
            "zeroline": False,
        },
        "yaxis": {
            "gridcolor": COLORS["chart_grid"],
            "linecolor": COLORS["chart_grid"],
            "tickfont": {"size": 11},
            "title_font": {"size": 12},
            "zeroline": False,
        },
        "colorway": CHART_COLORS,
        "hoverlabel": {
            "bgcolor": "#1C1C1F",
            "bordercolor": "rgba(255, 255, 255, 0.1)",
            "font": {
                "family": "Fira Sans, sans-serif",
                "size": 13,
                "color": "#F4F4F5",  # Light text for dark background
            },
        },
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(0,0,0,0)",
            "font": {"size": 11},
        },
        "margin": {"l": 50, "r": 20, "t": 50, "b": 40},
    }


def apply_chart_theme(fig):
    """Apply consistent theme to any plotly figure."""
    theme = get_chart_theme()
    fig.update_layout(**theme)
    return fig


# ============================================
# KPI CARD FIGURES
# ============================================


def create_kpi_figure(
    value,
    title,
    subtitle=None,
    delta=None,
    delta_ref=None,
    prefix="",
    suffix="",
    color=None,
):
    """
    Create a professional KPI indicator figure.

    Args:
        value: The main value to display
        title: KPI title
        subtitle: Optional subtitle/description
        delta: Optional delta value for trend
        delta_ref: Reference value for delta calculation
        prefix: Value prefix (e.g., "$")
        suffix: Value suffix (e.g., "kL")
        color: Override color for the value

    Returns:
        Plotly Figure with KPI indicator
    """
    # Determine color based on context
    if color is None:
        color = COLORS["text_primary"]

    # Build indicator
    fig = go.Figure()

    indicator_config = {
        "mode": "number",
        "value": value,
        "number": {
            "prefix": prefix,
            "suffix": suffix,
            "font": {
                "family": "Fira Code, monospace",
                "size": 36,
                "color": color,
            },
            "valueformat": ",.0f" if value >= 100 else ",.1f",
        },
        "title": {
            "text": f"<b>{title}</b>"
            + (
                f"<br><span style='font-size:11px;color:{COLORS['text_muted']}'>{subtitle}</span>"
                if subtitle
                else ""
            ),
            "font": {
                "family": "Fira Sans, sans-serif",
                "size": 13,
                "color": COLORS["text_secondary"],
            },
        },
        "domain": {"x": [0, 1], "y": [0.15, 1]},
    }

    # Add delta if provided
    if delta is not None:
        indicator_config["mode"] = "number+delta"
        indicator_config["delta"] = {
            "reference": delta_ref if delta_ref is not None else value - delta,
            "relative": True,
            "position": "bottom",
            "valueformat": ".1%",
            "increasing": {"color": COLORS["danger"]},  # Leak increase is bad
            "decreasing": {"color": COLORS["success"]},
            "font": {"size": 12},
        }

    fig.add_trace(go.Indicator(**indicator_config))

    # Apply theme
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        height=100,
    )

    return fig


def create_gauge_figure(value, title, thresholds=None):
    """
    Create a professional gauge chart for confidence/score.

    Args:
        value: Current value (0-100)
        title: Gauge title
        thresholds: Optional dict with low/medium/high thresholds

    Returns:
        Plotly Figure with gauge
    """
    if thresholds is None:
        thresholds = {"low": 30, "medium": 70, "high": 90}

    # Determine color based on value
    if value >= thresholds["high"]:
        bar_color = COLORS["danger"]
    elif value >= thresholds["medium"]:
        bar_color = COLORS["warning"]
    elif value >= thresholds["low"]:
        bar_color = COLORS["info"]
    else:
        bar_color = COLORS["success"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=max(0, min(100, float(value or 0))),
            number={
                "suffix": "%",
                "font": {
                    "family": "Fira Code, monospace",
                    "size": 22,
                    "color": COLORS["text_primary"],
                },
            },
            title={
                "text": title,
                "font": {
                    "family": "Fira Sans, sans-serif",
                    "size": 11,
                    "color": COLORS["text_secondary"],
                },
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": COLORS["chart_grid"],
                    "tickfont": {"size": 9, "color": COLORS["text_muted"]},
                },
                "bar": {"color": bar_color, "thickness": 0.6},
                "bgcolor": COLORS["bg_hover"],
                "borderwidth": 0,
                "steps": [
                    {
                        "range": [0, thresholds["low"]],
                        "color": "rgba(34, 197, 94, 0.1)",
                    },
                    {
                        "range": [thresholds["low"], thresholds["medium"]],
                        "color": "rgba(6, 182, 212, 0.1)",
                    },
                    {
                        "range": [thresholds["medium"], thresholds["high"]],
                        "color": "rgba(245, 158, 11, 0.1)",
                    },
                    {
                        "range": [thresholds["high"], 100],
                        "color": "rgba(239, 68, 68, 0.1)",
                    },
                ],
                "threshold": {
                    "line": {"color": COLORS["text_muted"], "width": 2},
                    "thickness": 0.75,
                    "value": value,
                },
            },
            domain={"x": [0.1, 0.9], "y": [0.15, 0.85]},
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5, r=5, t=5, b=5),
        height=120,
        autosize=True,
    )

    return fig


# ============================================
# CHART HELPERS
# ============================================


def fig_placeholder(title, subtitle="No data to display"):
    """Create a professional placeholder figure."""
    fig = go.Figure()
    fig.add_annotation(
        text=f"<span style='color:{COLORS['text_muted']}'>{subtitle}</span>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(
            family="Fira Sans, sans-serif",
            size=14,
            color=COLORS["text_muted"],
        ),
    )
    fig.update_layout(
        title={
            "text": title,
            "font": {
                "family": "Fira Sans, sans-serif",
                "size": 16,
                "color": COLORS["text_primary"],
            },
            "x": 0,
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=20, t=50, b=30),
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def gauge_figure(title, value):
    """Create a gauge chart for confidence/score display (legacy compatibility)."""
    return create_gauge_figure(value, title)


def mini_progress(label, value, tooltip_text=None, tooltip_id=None):
    """
    Create a professional signal progress bar with enhanced styling.

    Based on UI UX Pro Max signal indicator design.
    """
    value = max(0, min(1, float(value or 0)))
    pct = round(value * 100)

    # Determine color based on value (for leak detection, higher = worse)
    if pct >= 70:
        color_class = "danger"
        fill_color = COLORS["danger"]
    elif pct >= 40:
        color_class = "warning"
        fill_color = COLORS["warning"]
    else:
        color_class = "success"
        fill_color = COLORS["success"]

    # Label with optional tooltip
    label_content = html.Span(
        [
            label,
            (
                html.Span(
                    " ℹ️",
                    id=tooltip_id,
                    style={"cursor": "help", "fontSize": "0.65rem", "opacity": "0.7"},
                )
                if tooltip_text and tooltip_id
                else None
            ),
        ],
        className="signal-label",
        style={
            "fontSize": "0.75rem",
            "fontWeight": "500",
            "color": COLORS["text_muted"],
            "textTransform": "uppercase",
            "letterSpacing": "0.025em",
        },
    )

    # Value display with color
    value_display = html.Span(
        f"{pct}%",
        className=f"signal-value--{color_class}",
        style={
            "fontSize": "0.875rem",
            "fontWeight": "600",
            "fontFamily": "Fira Code, monospace",
            "color": fill_color,
        },
    )

    # Progress bar
    progress_bar = html.Div(
        html.Div(
            style={
                "width": f"{pct}%",
                "height": "100%",
                "backgroundColor": fill_color,
                "borderRadius": "9999px",
                "transition": "width 0.3s ease-out",
            }
        ),
        style={
            "width": "100%",
            "height": "4px",
            "backgroundColor": COLORS["bg_hover"],
            "borderRadius": "9999px",
            "overflow": "hidden",
            "marginTop": "4px",
        },
    )

    elements = [
        html.Div(
            [label_content, value_display],
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
            },
        ),
        progress_bar,
    ]

    # Add tooltip if provided
    if tooltip_text and tooltip_id:
        elements.append(
            dbc.Tooltip(
                tooltip_text,
                target=tooltip_id,
                placement="top",
                style={"fontSize": "0.85rem"},
            )
        )

    return html.Div(
        elements,
        className="signal-item",
        style={
            "padding": "8px 12px",
            "backgroundColor": COLORS["bg_hover"],
            "borderRadius": "8px",
            "marginBottom": "8px",
        },
    )


# -------------------------
# Action Log Functions
# -------------------------


def safe_read_actions():
    """Read action log CSV file safely."""
    if os.path.exists(ACTION_LOG):
        try:
            df = pd.read_csv(
                ACTION_LOG, parse_dates=["timestamp", "start_day", "end_day"]
            )
            return df
        except Exception as e:
            log.error(f"Failed reading Action_Log.csv: {e}")
    return pd.DataFrame(
        columns=[
            "timestamp",
            "site_id",
            "event_id",
            "start_day",
            "end_day",
            "status",
            "action",
            "notes",
        ]
    )


def append_action_row(
    site_id, event_id, start_day, end_day, status, action, notes="", **kwargs
):
    """
    Enhanced action logging with support for additional metadata.

    Args:
        site_id: Property ID
        event_id: Leak event ID
        start_day: Event start date
        end_day: Event end date
        status: Current status (WATCH/INVESTIGATE/CALL)
        action: Action taken (Acknowledge/Watch/Escalate/Resolved/Ignore)
        notes: Free text notes
        **kwargs: Additional fields (user, resolution_type, cost, etc.)
    """
    row = {
        "timestamp": pd.Timestamp.now(),
        "site_id": site_id,
        "event_id": event_id,
        "start_day": pd.to_datetime(start_day) if pd.notna(start_day) else pd.NaT,
        "end_day": pd.to_datetime(end_day) if pd.notna(end_day) else pd.NaT,
        "status": status,
        "action": action,
        "notes": notes or "",
        "user": kwargs.get("user", ""),
        "resolution_type": kwargs.get("resolution_type", ""),
        "resolution_cause": kwargs.get("resolution_cause", ""),
        "cost": kwargs.get("cost", ""),
        "resolved_by": kwargs.get("resolved_by", ""),
        "escalated_to": kwargs.get("escalated_to", ""),
        "urgency": kwargs.get("urgency", ""),
        "reminder_date": kwargs.get("reminder_date", ""),
        "reason": kwargs.get("reason", ""),
    }
    df = safe_read_actions()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(ACTION_LOG, index=False)
    log.info(
        f"Action logged: {action} on {event_id} by {kwargs.get('user', 'Unknown')}"
    )
    return df


# -------------------------
# UI Helpers
# -------------------------


def month_marks_from_dateindex(idx):
    """Generate month marks for date slider."""
    marks = {}
    for i, d in enumerate(idx):
        if pd.Timestamp(d).day == 1:
            marks[i] = pd.to_datetime(d).strftime("%b %Y")
    if not marks:
        marks[0] = pd.to_datetime(idx[0]).strftime("%d %b %Y")
        marks[len(idx) - 1] = pd.to_datetime(idx[-1]).strftime("%d %b %Y")
    return marks


def get_confidence_interpretation(confidence):
    """
    Returns color, text, and action guidance based on confidence level.

    Enhanced with UI UX Pro Max status colors and clearer guidance.

    Returns: (color, level_text, action_text, icon)
    """
    conf = float(confidence or 0)

    if conf >= 90:
        return (
            "danger",
            "Critical (90-100%)",
            "🚨 URGENT: Take immediate action - confirmed major leak",
            "🔴",
        )
    elif conf >= 70:
        return (
            "warning",
            "High (70-89%)",
            "⚠️ ACTION: Investigate immediately - likely leak detected",
            "🟠",
        )
    elif conf >= 50:
        return (
            "info",
            "Moderate (50-69%)",
            "👁️ WATCH: Monitor closely - possible leak activity",
            "🟡",
        )
    elif conf >= 30:
        return (
            "secondary",
            "Low (30-49%)",
            "📋 REVIEW: Continue monitoring - may be normal variation",
            "⚪",
        )
    else:
        return (
            "success",
            "Normal (<30%)",
            "✓ OK: Within normal consumption patterns",
            "🟢",
        )


def create_tooltip(target_id, text):
    """Helper to create a Bootstrap tooltip with consistent styling."""
    return dbc.Tooltip(
        text,
        target=target_id,
        placement="top",
        style={
            "fontSize": "0.85rem",
            "backgroundColor": COLORS["bg_card"],
            "borderColor": COLORS["chart_grid"],
        },
    )


def incident_badges(inc):
    """
    Create professional status badges for incident display.

    Based on UI UX Pro Max status badge system.
    """
    chips = []

    # Status badge with appropriate color
    status = inc.get("status", "WATCH")
    status_colors = {
        "WATCH": {
            "bg": "rgba(6, 182, 212, 0.15)",
            "color": "#06B6D4",
            "border": "rgba(6, 182, 212, 0.3)",
        },
        "INVESTIGATE": {
            "bg": "rgba(245, 158, 11, 0.15)",
            "color": "#F59E0B",
            "border": "rgba(245, 158, 11, 0.3)",
        },
        "CALL": {
            "bg": "rgba(239, 68, 68, 0.15)",
            "color": "#EF4444",
            "border": "rgba(239, 68, 68, 0.3)",
        },
        "RESOLVED": {
            "bg": "rgba(34, 197, 94, 0.15)",
            "color": "#22C55E",
            "border": "rgba(34, 197, 94, 0.3)",
        },
        "IGNORED": {
            "bg": "rgba(113, 113, 122, 0.15)",
            "color": "#71717A",
            "border": "rgba(113, 113, 122, 0.3)",
        },
        "Suppressed": {
            "bg": "rgba(100, 116, 139, 0.2)",
            "color": "#94A3B8",
            "border": "rgba(100, 116, 139, 0.3)",
        },
    }
    status_style = status_colors.get(status, status_colors["WATCH"])

    chips.append(
        html.Span(
            [
                html.Span(
                    style={
                        "width": "6px",
                        "height": "6px",
                        "borderRadius": "50%",
                        "backgroundColor": status_style["color"],
                        "display": "inline-block",
                        "marginRight": "6px",
                    }
                ),
                status,
            ],
            className="status-badge",
            style={
                "display": "inline-flex",
                "alignItems": "center",
                "padding": "4px 10px",
                "fontSize": "0.7rem",
                "fontWeight": "600",
                "textTransform": "uppercase",
                "letterSpacing": "0.05em",
                "borderRadius": "9999px",
                "backgroundColor": status_style["bg"],
                "color": status_style["color"],
                "border": f"1px solid {status_style['border']}",
                "marginRight": "6px",
            },
        )
    )

    # Severity badge
    severity = inc.get("severity_max", "S1")
    severity_colors = {
        "S1": {"bg": "rgba(239, 68, 68, 0.15)", "color": "#EF4444"},
        "S2": {"bg": "rgba(245, 158, 11, 0.15)", "color": "#F59E0B"},
        "S3": {"bg": "rgba(6, 182, 212, 0.15)", "color": "#06B6D4"},
    }
    sev_style = severity_colors.get(severity, severity_colors["S3"])

    chips.append(
        html.Span(
            severity,
            style={
                "padding": "4px 8px",
                "fontSize": "0.7rem",
                "fontWeight": "600",
                "borderRadius": "6px",
                "backgroundColor": sev_style["bg"],
                "color": sev_style["color"],
                "marginRight": "6px",
            },
        )
    )

    # Duration badge
    days = inc.get("days_persisted", 0)
    chips.append(
        html.Span(
            f"{days} days",
            style={
                "padding": "4px 8px",
                "fontSize": "0.7rem",
                "fontWeight": "500",
                "fontFamily": "Fira Code, monospace",
                "borderRadius": "6px",
                "backgroundColor": COLORS["bg_hover"],
                "color": COLORS["text_secondary"],
                "marginRight": "6px",
            },
        )
    )

    # Reason code badges
    for sig in sorted(inc.get("reason_codes", []) or []):
        chips.append(
            html.Span(
                sig,
                style={
                    "padding": "4px 8px",
                    "fontSize": "0.65rem",
                    "fontWeight": "500",
                    "fontFamily": "Fira Code, monospace",
                    "borderRadius": "4px",
                    "backgroundColor": "rgba(59, 130, 246, 0.1)",
                    "color": COLORS["primary"],
                    "marginRight": "4px",
                },
            )
        )

    return html.Div(
        chips,
        style={
            "display": "flex",
            "flexWrap": "wrap",
            "alignItems": "center",
            "gap": "4px",
        },
    )


# ============================================
# CHART CREATION HELPERS
# ============================================


def create_time_series_chart(df, x_col, y_col, title, color=None, fill=False):
    """
    Create a professional time series line chart.

    Args:
        df: DataFrame with data
        x_col: Column name for x-axis (time)
        y_col: Column name for y-axis (value)
        title: Chart title
        color: Optional override color
        fill: Whether to fill area under line

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    line_color = color or COLORS["chart_line"]

    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode="lines",
            name=y_col,
            line=dict(color=line_color, width=2),
            fill="tozeroy" if fill else None,
            fillcolor=COLORS["chart_area"] if fill else None,
            hovertemplate=f"<b>%{{x}}</b><br>{y_col}: %{{y:.2f}}<extra></extra>",
        )
    )

    fig.update_layout(title=title)
    return apply_chart_theme(fig)


def create_bar_chart(df, x_col, y_col, title, color=None, horizontal=False):
    """
    Create a professional bar chart.

    Args:
        df: DataFrame with data
        x_col: Column name for categories
        y_col: Column name for values
        title: Chart title
        color: Optional override color or list of colors
        horizontal: If True, create horizontal bar chart

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    bar_color = color or COLORS["primary"]

    if horizontal:
        fig.add_trace(
            go.Bar(
                x=df[y_col],
                y=df[x_col],
                orientation="h",
                marker_color=bar_color,
                hovertemplate=f"<b>%{{y}}</b><br>{y_col}: %{{x:.2f}}<extra></extra>",
            )
        )
    else:
        fig.add_trace(
            go.Bar(
                x=df[x_col],
                y=df[y_col],
                marker_color=bar_color,
                hovertemplate=f"<b>%{{x}}</b><br>{y_col}: %{{y:.2f}}<extra></extra>",
            )
        )

    fig.update_layout(title=title)
    return apply_chart_theme(fig)


def create_scatter_chart(df, x_col, y_col, title, color_col=None, size_col=None):
    """
    Create a professional scatter plot.

    Args:
        df: DataFrame with data
        x_col: Column for x-axis
        y_col: Column for y-axis
        title: Chart title
        color_col: Optional column for color encoding
        size_col: Optional column for size encoding

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    marker_config = {
        "color": COLORS["primary"],
        "size": 10,
        "opacity": 0.7,
        "line": {"width": 1, "color": COLORS["primary_light"]},
    }

    if color_col and color_col in df.columns:
        marker_config["color"] = df[color_col]
        marker_config["colorscale"] = [
            [0, COLORS["success"]],
            [0.5, COLORS["warning"]],
            [1, COLORS["danger"]],
        ]
        marker_config["showscale"] = True

    if size_col and size_col in df.columns:
        marker_config["size"] = df[size_col]
        marker_config["sizemode"] = "area"
        marker_config["sizeref"] = (
            2.0 * max(df[size_col]) / (40.0**2) if max(df[size_col]) > 0 else 1
        )

    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode="markers",
            marker=marker_config,
            hovertemplate=f"<b>%{{customdata}}</b><br>{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}<extra></extra>",
            customdata=df.index if "site_id" not in df.columns else df["site_id"],
        )
    )

    fig.update_layout(title=title)
    return apply_chart_theme(fig)
