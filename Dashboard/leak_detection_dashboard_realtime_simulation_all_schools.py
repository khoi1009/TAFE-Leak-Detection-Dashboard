# dash_dashboard.py
# -*- coding: utf-8 -*-
# %%
import os
import sys
import json
import math
import yaml
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import dash
from dash import dcc, html, Input, Output, State, MATCH, ALL, ctx, dash_table
import dash_bootstrap_components as dbc

import plotly.graph_objects as go
import plotly.io as pio
import sys, io, os, logging

# -------------------------
# Global Theming & Logging
# -------------------------

pio.templates.default = "plotly_dark"  # Dark charts everywhere
APP_TITLE = "💦 Smart Leak Detection — TAFE and Schools"
CONFIG_FILE = "config_leak_detection.yml"
ACTION_LOG = "Action_Log.csv"
# Ensure logs folder exists
os.makedirs("logs", exist_ok=True)

# Safe UTF-8 stdout wrapping (only in real Python console, not Jupyter)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        # Always UTF-8 log file
        logging.FileHandler("logs/dashboard.log", encoding="utf-8"),
        # Stream to console; stderr is safer in Jupyter
        logging.StreamHandler(sys.stderr),
    ],
)

log = logging.getLogger("dashboard")

# -------------------------
# Try importing your engine
# -------------------------

ENGINE_OK = True
try:
    from Model_1_realtime_simulation import (
        load_tafe_data,
        validate_config,
        run_efficient_pipeline,
        process_site,
        SchoolLeakDetector,
    )

    log.info("Model_1.py loaded successfully.")
except Exception as e:
    ENGINE_OK = False
    log.warning(f"Model_1.py not available or failed to import: {e}")

    # Minimal fallbacks so the UI still renders
    def load_tafe_data(path):
        rng = pd.date_range("2025-03-01", "2025-05-31", freq="h")
        sites = ["Property 10001", "Property 11127", "Property 20002"]
        dfs = {}
        for s in sites:
            base = np.random.gamma(shape=2.0, scale=40.0, size=len(rng))
            base += np.where((rng.hour >= 0) & (rng.hour < 4), 80, 0)
            if s == "Property 11127":
                burst_mask = (rng >= "2025-04-10") & (rng <= "2025-04-20")
                base = base + np.where(burst_mask, 200, 0)
            df = pd.DataFrame(
                {
                    "time": rng,
                    "flow": np.clip(base + np.random.normal(0, 8, len(rng)), 0, None),
                }
            )
            dfs[s] = df
        return dfs

    def validate_config(cfg):
        return True

    run_efficient_pipeline = None
    process_site = None

    class SchoolLeakDetector:
        def __init__(self, df, site_id, cfg, **kw):
            self.df = df.copy()
            self.site_id = site_id
            self.cfg = cfg
            self.daily = pd.DataFrame(
                index=pd.to_datetime(sorted(df["time"].dt.date.unique()))
            )
            if "flow" in df:
                nf = (
                    df[df["time"].dt.hour.isin([0, 1, 2, 3])]
                    .groupby(df["time"].dt.date)["flow"]
                    .mean()
                )
                self.daily["NF_d"] = (
                    pd.Series(nf.values, index=pd.to_datetime(nf.index))
                    .reindex(self.daily.index)
                    .fillna(method="ffill")
                    .fillna(0)
                )
                ah = (
                    df[(df["time"].dt.hour >= 16) | (df["time"].dt.hour < 7)]
                    .groupby(df["time"].dt.date)["flow"]
                    .sum()
                )
                self.daily["A_d"] = (
                    pd.Series(ah.values, index=pd.to_datetime(ah.index))
                    .reindex(self.daily.index)
                    .fillna(method="ffill")
                    .fillna(0)
                    / 1000
                )
            else:
                self.daily["NF_d"] = 0
                self.daily["A_d"] = 0
            self.incidents = []

        def signals_and_score(self, d):
            subs = {
                "MNF": 0.2,
                "RESIDUAL": 0.1,
                "CUSUM": 0.1,
                "AFTERHRS": 0.1,
                "BURSTBF": 0.0,
            }
            return subs, 30, 120, 10

        def to_plotly_figs(self, incident, window_days=30):
            def ph(t):
                f = go.Figure()
                f.add_annotation(text=t, x=0.5, y=0.5, showarrow=False)
                f.update_layout(margin=dict(l=30, r=20, t=40, b=30))
                return f

            return (
                ph("No raw flow available"),
                ph("No NF trend"),
                ph("No After-hours"),
                ph("No weekly heatmap"),
            )


# -------------------------
# Config Loading (with safe defaults)
# -------------------------

DEFAULT_CFG = {
    "timezone": "Australia/Sydney",
    "night_start": 0,
    "night_end": 4,
    "after_hours_start": 16,
    "after_hours_end": 7,
    "baseline_window_days": 28,
    "abs_floor_lph": 100,
    "sustained_after_hours_delta_kl": 0.5,
    "spike_multiplier": 2.0,
    "spike_ref_percentile": 95,
    "cusum_k": 0.5,
    "cusum_h": 4.0,
    "score_weights": {
        "MNF": 0.40,
        "RESIDUAL": 0.20,
        "CUSUM": 0.20,
        "AFTERHRS": 0.10,
        "BURSTBF": 0.10,
    },
    "persistence_gates": {
        "<100": {"fast_min": 3, "default_max": 5},
        "100-200": {"fast_min": 3, "default_max": 6},
        "200-1000": {"fast_min": 3, "default_max": 7},
        ">=1000": {"fast_min": 3, "default_max": 8},
    },
    "severity_bands_lph": {
        "S1": [0, 100],
        "S2": [100, 200],
        "S3": [200, 1000],
        "S4": [1000, 3000],
        "S5": [3000, 10_000_000],
    },
    "export_folder": "export",
    "save_dir": "plots",
    "data_path": "data.xlsx",
    "events_tab_filters": {
        "min_leak_score": 30.0,  # % threshold on UI leak score
        "min_volume_kl": 0.10,  # kL threshold (UI total volume)
        "allowed_statuses": ["INVESTIGATE", "CALL"],  # hide WATCH by default
    },
    "replay": {
        "analyze_hour": 6,  # run daily analysis at 06:00
        "pause_on_incident_default": True,
        "max_days_per_run": 90,  # safety guard in UI runs
    },
}

if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        cfg = {**DEFAULT_CFG, **yaml.safe_load(f)}
else:
    cfg = DEFAULT_CFG.copy()
    log.warning("config_leak_detection.yml not found. Using safe defaults.")

try:
    if ENGINE_OK:
        validate_config(cfg)
except Exception as e:
    log.error(f"Config validation failed: {e}. Falling back to defaults.")
    cfg = DEFAULT_CFG.copy()

os.makedirs(cfg.get("export_folder", "export"), exist_ok=True)
os.makedirs(cfg.get("save_dir", "plots"), exist_ok=True)

# -------------------------
# Data Loading (UI renders first; heavy analysis runs on button)
# -------------------------


def safe_load_sites():
    try:
        if ENGINE_OK and os.path.exists(cfg["data_path"]):
            school_dfs = load_tafe_data(cfg["data_path"])
            log.info(f"Loaded {len(school_dfs)} sites from data_path.")
            return school_dfs
        else:
            demo = load_tafe_data(
                cfg["data_path"]
            )  # fallback impl makes demo if ENGINE_OK False
            log.info("Using demo data (engine missing or data_path not found).")
            return demo
    except Exception as e:
        log.error(f"Failed to load data: {e}. Generating small demo.")
        return load_tafe_data(cfg["data_path"])


SCHOOL_DFS = safe_load_sites()
ALL_SITES = sorted(list(SCHOOL_DFS.keys()))


def global_date_bounds():
    frames = []
    for df in SCHOOL_DFS.values():
        if "time" in df:
            frames.append(df["time"])
    all_times = (
        pd.concat(frames)
        if frames
        else pd.Series(pd.date_range("2025-03-01", periods=48, freq="H"))
    )
    start = pd.to_datetime(all_times.min()).normalize()
    end = pd.to_datetime(all_times.max()).normalize()
    return pd.date_range(start, end, freq="D")


DATE_INDEX = global_date_bounds()
DATE_MARKS = {
    i: pd.to_datetime(dt).strftime("%b %Y")
    for i, dt in enumerate(DATE_INDEX)
    if pd.Timestamp(dt).day == 1
}
DEFAULT_CUTOFF_IDX = max(0, len(DATE_INDEX) - 7)  # near last week

# -------------------------
# In-Memory Cache
# -------------------------

# SITE_CACHE[site_id] = {
#   "detector": detector,
#   "daily": detector.daily,
#   "df": detector.df,
#   "incidents": [incident dicts],
#   "confirmed": confirmed_df_for_site
# }
SITE_CACHE = {}


def make_json_safe(inc):
    """
    Convert a single incident dict into JSON-serializable form.
    """
    return {
        **inc,
        "start_day": (
            str(inc.get("start_day")) if inc.get("start_day") is not None else None
        ),
        "last_day": (
            str(inc.get("last_day")) if inc.get("last_day") is not None else None
        ),
        "alert_date": (
            str(inc.get("alert_date")) if inc.get("alert_date") is not None else None
        ),
        "start_time": (
            str(inc.get("start_time")) if inc.get("start_time") is not None else None
        ),
        "end_time": (
            str(inc.get("end_time")) if inc.get("end_time") is not None else None
        ),
        "reason_codes": sorted(list(inc.get("reason_codes", []))),
    }


def normalize_incidents(incidents):
    """
    Apply make_json_safe() to a list of incidents or confirmed records.
    """
    return [make_json_safe(inc) for inc in (incidents or [])]


def dedupe_by_event_id(inc_list):
    """Return unique incidents keyed by event_id.
    Keep the one with later last_day, then higher confidence, then bigger volume."""
    best = {}
    for inc in inc_list or []:
        eid = inc.get("event_id")
        if not eid:
            # fallback if normalize_incident wasn't applied yet
            st = pd.to_datetime(inc.get("start_day")).date()
            en = pd.to_datetime(inc.get("last_day", inc.get("start_day"))).date()
            eid = f"{inc.get('site_id','?')}__{st}__{en}"
            inc["event_id"] = eid

        def key(x):
            return (
                pd.to_datetime(x.get("last_day", x.get("start_day"))),
                float(x.get("confidence", 0.0)),
                float(x.get("ui_total_volume_kL", x.get("volume_lost_kL", 0.0))),
            )

        cur = best.get(eid)
        if (cur is None) or (key(inc) > key(cur)):
            best[eid] = inc
    return list(best.values())


def compute_or_refresh_site(site_id, up_to_date, start_date=None, warmup_days=None):
    """
    Build/refresh the cached detector for a site using a historical slice that
    mimics real-time (<= up_to_date) and optionally includes a warm-up window
    before the requested start_date so baselining is immediately available.

    Parameters
    ----------
    site_id : str
    up_to_date : datetime|str
        Replay cutoff (e.g., 06:00 of the current replay day).
    start_date : datetime|str|None
        If provided, the analysis window starts here; the slice will include
        `warmup_days` *before* this date for baseline context.
    warmup_days : int|None
        If None, defaults to cfg['baseline_window_days'].
    """
    df_full = SCHOOL_DFS.get(site_id, pd.DataFrame())

    if df_full.empty:
        SITE_CACHE[site_id] = {
            "detector": None,
            "daily": pd.DataFrame(),
            "df": pd.DataFrame(),
            "incidents": [],
            "confirmed": pd.DataFrame(),
        }
        return SITE_CACHE[site_id]

    # --- Resolve bounds -------------------------------------------------------
    up_to = pd.to_datetime(up_to_date)

    # Ensure time is datetime-like for robust slicing
    if not np.issubdtype(df_full["time"].dtype, np.datetime64):
        df_full = df_full.copy()
        df_full["time"] = pd.to_datetime(df_full["time"])

    if start_date is not None:
        sd = pd.to_datetime(start_date)
        wd = (
            int(warmup_days)
            if warmup_days is not None
            else int(cfg.get("baseline_window_days", 28))
        )
        lb = sd - pd.Timedelta(days=wd)
        log.info(
            f"{site_id}: Applied warmup of {wd} days; lower bound set to {lb.date()}"
        )
    else:
        # No explicit start_date: use earliest available time (acts like "all history up to up_to")
        lb = pd.to_datetime(df_full["time"].min())
        wd = 0
        log.info(
            f"{site_id}: No start_date provided; using full history from {lb.date()}"
        )

    # Guard: never let lower bound exceed the upper bound
    if lb > up_to:
        lb = up_to

    mask = (df_full["time"] >= lb) & (df_full["time"] <= up_to)
    df_slice = df_full.loc[mask].copy()
    log.info(
        f"{site_id}: Full data rows={len(df_full)}; Sliced rows={len(df_slice)} from {lb.date()} to {up_to.date()}"
    )

    # Console/file logs so devs can follow the replay slicing precisely
    try:
        lb_str = pd.to_datetime(lb).strftime("%Y-%m-%d %H:%M")
    except Exception:
        lb_str = str(lb)
    try:
        up_to_str = pd.to_datetime(up_to).strftime("%Y-%m-%d %H:%M")
    except Exception:
        up_to_str = str(up_to)

    log.info(
        f"{site_id}: slice [{lb_str} → {up_to_str}] | rows={len(df_slice)} | warmup_days={wd}"
    )

    # ✅ FIX: Preserve signal components AND frozen confidence from previous detector
    prev_signal_components = {}
    prev_confidence_by_date = {}
    if site_id in SITE_CACHE:
        prev_detector = SITE_CACHE[site_id].get("detector")
        if prev_detector:
            if hasattr(prev_detector, "signal_components_by_date"):
                prev_signal_components = prev_detector.signal_components_by_date.copy()
                log.info(
                    f"{site_id}: Preserved {len(prev_signal_components)} signal components from previous detector"
                )
            if hasattr(prev_detector, "confidence_by_date"):
                prev_confidence_by_date = prev_detector.confidence_by_date.copy()
                log.info(
                    f"{site_id}: Preserved {len(prev_confidence_by_date)} FROZEN confidence values from previous detector"
                )

        # CRITICAL: Extract from cached incidents - this is the main source of frozen confidence
        cached_incidents = SITE_CACHE[site_id].get("incidents", [])
        for inc in cached_incidents:
            if "signal_components_by_date" in inc:
                for date_key, comp in inc["signal_components_by_date"].items():
                    if "confidence" in comp:
                        prev_confidence_by_date[date_key] = comp["confidence"]

        if prev_confidence_by_date:
            log.info(
                f"{site_id}: Extracted {len(prev_confidence_by_date)} FROZEN confidence values from cached incidents"
            )

    # --- Run engine (prefer compiled/process pool version if available) -------
    detector, confirmed_df = None, pd.DataFrame()
    try:
        if ENGINE_OK and (process_site is not None):
            log.info(f"{site_id}: Using ENGINE path (process_site)")
            # ✅ FIX: Pass frozen confidence values to process_site so they can be restored BEFORE state_machine
            # Use the engine's single-site runner (returns detector + confirmed table)
            _, detector, confirmed_df = process_site(
                (
                    site_id,
                    df_slice,
                    cfg,
                    [],
                    up_to,
                    prev_signal_components,
                    prev_confidence_by_date,
                )
            )
            log.info(
                f"{site_id}: ENGINE returned detector with {len(detector.incidents) if detector and hasattr(detector, 'incidents') else 0} incidents"
            )
            # ✅ FIX: Restore preserved signal components AND frozen confidence AFTER process_site
            if detector:
                if prev_signal_components and hasattr(
                    detector, "signal_components_by_date"
                ):
                    # The detector already ran, but we can merge old cached values
                    # This won't affect this run but helps future incremental updates
                    for date_key, components in prev_signal_components.items():
                        if date_key not in detector.signal_components_by_date:
                            detector.signal_components_by_date[date_key] = components
                    log.info(
                        f"{site_id}: [ENGINE] Merged signal components; total cached dates: {len(detector.signal_components_by_date)}"
                    )
                if prev_confidence_by_date and hasattr(detector, "confidence_by_date"):
                    # Merge frozen confidence values - CRITICAL for preventing recalculation
                    for date_key, conf_val in prev_confidence_by_date.items():
                        if date_key not in detector.confidence_by_date:
                            detector.confidence_by_date[date_key] = conf_val
                    log.info(
                        f"{site_id}: [ENGINE] Merged FROZEN confidence; total frozen dates: {len(detector.confidence_by_date)}"
                    )
        else:
            # Fallback path: construct detector directly if methods exist
            try:
                detector = SchoolLeakDetector(df_slice, site_id, cfg, up_to_date=up_to)

                # ✅ FIX: Restore preserved signal components AND frozen confidence BEFORE running state_machine
                if prev_signal_components and hasattr(
                    detector, "signal_components_by_date"
                ):
                    detector.signal_components_by_date = prev_signal_components.copy()
                    log.info(
                        f"{site_id}: [FALLBACK] Restored {len(prev_signal_components)} signal components BEFORE state_machine"
                    )
                if prev_confidence_by_date and hasattr(detector, "confidence_by_date"):
                    detector.confidence_by_date = prev_confidence_by_date.copy()
                    log.info(
                        f"{site_id}: [FALLBACK] Restored {len(prev_confidence_by_date)} FROZEN confidence values BEFORE state_machine"
                    )

                if hasattr(detector, "preprocess"):
                    detector.preprocess()
                if hasattr(detector, "state_machine"):
                    detector.state_machine()

                # Log final count after state_machine
                if hasattr(detector, "signal_components_by_date"):
                    log.info(
                        f"{site_id}: [FALLBACK] After state_machine: {len(detector.signal_components_by_date)} cached dates"
                    )
                if hasattr(detector, "confidence_by_date"):
                    log.info(
                        f"{site_id}: [FALLBACK] After state_machine: {len(detector.confidence_by_date)} frozen confidence values"
                    )

                # We won't fabricate a confirmed_df here; UI reads incidents from detector
                confirmed_df = pd.DataFrame()
            except Exception as e2:
                log.error(f"{site_id}: fallback detector failed: {e2}")
                detector, confirmed_df = None, pd.DataFrame()
    except Exception as e:
        log.error(f"process_site failed for {site_id}: {e}")
        detector, confirmed_df = None, pd.DataFrame()

    # --- Normalize confirmed incidents table (robust to older schemas) --------
    if isinstance(confirmed_df, pd.DataFrame) and not confirmed_df.empty:
        if ("start_day" in confirmed_df.columns) and (
            "start_time" not in confirmed_df.columns
        ):
            confirmed_df["start_time"] = confirmed_df["start_day"]
        if ("last_day" in confirmed_df.columns) and (
            "end_time" not in confirmed_df.columns
        ):
            confirmed_df["end_time"] = confirmed_df["last_day"]

    # --- Collect incidents from detector --------------------------------------
    # --- Collect incidents from detector --------------------------------------
    incidents = []
    if detector and hasattr(detector, "incidents"):

        for inc in detector.incidents:
            # ✅ DEBUG: Log if incident has signal components
            inc_event_id = inc.get("event_id", "unknown")
            has_sig_comp = "signal_components_by_date" in inc
            sig_comp_keys = (
                list(inc["signal_components_by_date"].keys()) if has_sig_comp else []
            )
            log.info(
                f"[COLLECT] {site_id} - {inc_event_id}: has_signal_components={has_sig_comp}, dates={sig_comp_keys}"
            )

            # >>> ADD THIS BLOCK <<<
            if not inc.get("category"):
                try:
                    if detector and hasattr(detector, "categorize_leak"):
                        cat, _intel = detector.categorize_leak(inc)  # engine method
                        inc["category"] = cat or "Unlabelled"
                except Exception:
                    inc["category"] = inc.get("category") or "Unlabelled"
            # <<< END ADD >>>

            incidents.append(make_json_safe(inc))

    # Stable de-duplication (prefer later last_day, then higher confidence, then higher volume)
    before = len(incidents)
    incidents = dedupe_by_event_id(incidents)
    removed = before - len(incidents)
    if removed > 0:
        log.info(f"{site_id}: de-duplicated {removed} incidents before caching")
    # Stable de-duplication (prefer later last_day, then higher confidence, then higher volume)

    # Propagate categories from incidents → confirmed_df (so Overview bar shows labels)
    if isinstance(confirmed_df, pd.DataFrame) and not confirmed_df.empty:
        confirmed_df = confirmed_df.copy()

        # Ensure event_id exists in confirmed_df (use the same fallback as dedupe_by_event_id)
        if "event_id" not in confirmed_df.columns:
            sd = pd.to_datetime(confirmed_df.get("start_day"), errors="coerce")
            ed = pd.to_datetime(confirmed_df.get("last_day"), errors="coerce")
            ed = ed.fillna(sd)
            sid = confirmed_df.get("site_id")
            confirmed_df["event_id"] = (
                sid.fillna("?").astype(str)
                + "__"
                + sd.dt.date.astype(str)
                + "__"
                + ed.dt.date.astype(str)
            )

        cat_map = {
            inc.get("event_id"): inc.get("category") or "Unlabelled"
            for inc in incidents
        }
        if "category" not in confirmed_df.columns:
            confirmed_df["category"] = None

        confirmed_df["category"] = (
            confirmed_df["event_id"]
            .map(cat_map)
            .fillna(confirmed_df["category"])
            .fillna("Unlabelled")
        )

    # --- Cache & return --------------------------------------------------------
    SITE_CACHE[site_id] = {
        "detector": detector,
        "daily": (
            getattr(detector, "daily", pd.DataFrame()) if detector else pd.DataFrame()
        ),
        "df": getattr(detector, "df", df_slice) if detector else df_slice,
        "incidents": incidents,
        "confirmed": (
            confirmed_df if isinstance(confirmed_df, pd.DataFrame) else pd.DataFrame()
        ),
    }
    return SITE_CACHE[site_id]


# -------------------------
# Utilities (UI/Charts/IO)
# -------------------------


def fig_placeholder(title, subtitle="No data to display"):
    f = go.Figure()
    f.add_annotation(text=subtitle, x=0.5, y=0.5, showarrow=False, font=dict(size=14))
    f.update_layout(title=title, margin=dict(l=30, r=20, t=50, b=30))
    return f


def gauge_figure(title, value):
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=max(0, min(100, float(value or 0))),
            title={"text": title},
            gauge={"axis": {"range": [0, 100]}},
        )
    ).update_layout(margin=dict(l=30, r=20, t=50, b=30))


def mini_progress(label, value, tooltip_text=None, tooltip_id=None):
    value = max(0, min(1, float(value or 0)))

    label_content = [html.Small(label, className="text-muted")]
    if tooltip_text and tooltip_id:
        label_content = [
            html.Small(
                [
                    label,
                    html.Span(
                        " ℹ️",
                        id=tooltip_id,
                        style={"cursor": "help", "fontSize": "0.65rem"},
                    ),
                ],
                className="text-muted",
            )
        ]

    elements = [
        html.Div(
            [
                *label_content,
                html.Small(
                    f"{round(value*100):d}%",
                    className="text-muted",
                    style={"float": "right"},
                ),
            ]
        ),
        dbc.Progress(
            value=value * 100,
            striped=False,
            animated=False,
            className="shadow-sm",
            style={"height": "10px", "backgroundColor": "#2b2b2b"},
        ),
    ]

    if tooltip_text and tooltip_id:
        elements.append(
            dbc.Tooltip(
                tooltip_text,
                target=tooltip_id,
                placement="top",
            )
        )

    return html.Div(elements, className="mb-2")


def safe_read_actions():
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
        **kwargs: Additional fields (user, resolution_type, cost, resolved_by, escalated_to,
                  reminder_date, reason, urgency, etc.)
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
        # Optional enhanced fields
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


def month_marks_from_dateindex(idx):
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

    Returns: (color, level_text, action_text, icon)
    """
    conf = float(confidence or 0)

    if conf >= 90:
        return (
            "danger",  # Bootstrap color
            "Very High (90-100%)",
            "🚨 URGENT: Take immediate action - virtually certain leak",
            "🔴",
        )
    elif conf >= 70:
        return (
            "warning",
            "High (70-89%)",
            "⚠️ ACTION REQUIRED: Investigate now - confirmed leak",
            "🟠",
        )
    elif conf >= 50:
        return (
            "info",
            "Moderate (50-69%)",
            "👁️ MONITOR: Watch closely and investigate if persists",
            "🟡",
        )
    elif conf >= 30:
        return (
            "secondary",
            "Low (30-49%)",
            "📋 WATCH: Continue monitoring - possible false alarm",
            "⚪",
        )
    else:
        return ("light", "Very Low (<30%)", "✓ NORMAL: Likely normal variation", "⚫")


def create_tooltip(target_id, text):
    """Helper to create a Bootstrap tooltip"""
    return dbc.Tooltip(
        text, target=target_id, placement="top", style={"fontSize": "0.85rem"}
    )


def incident_badges(inc):
    chips = []
    chips.append(dbc.Badge(inc.get("status", "WATCH"), color="info", className="me-1"))
    chips.append(
        dbc.Badge(inc.get("severity_max", "S1"), color="danger", className="me-1")
    )
    chips.append(
        dbc.Badge(
            f"{inc.get('days_persisted',0)} days", color="secondary", className="me-1"
        )
    )
    for sig in sorted(inc.get("reason_codes", []) or []):
        chips.append(dbc.Badge(sig, color="dark", className="me-1"))
    return chips


def make_incident_card(site_id, inc, detector):
    subs = inc.get("subscores_ui")
    leak_score = inc.get("leak_score_ui")

    if subs is None or leak_score is None:
        subs, leak_score, *_ = (
            detector.signals_and_score(pd.to_datetime(inc["last_day"]))
            if detector
            else (
                {"MNF": 0, "RESIDUAL": 0, "CUSUM": 0, "AFTERHRS": 0, "BURSTBF": 0},
                0,
                0,
                1,
            )
        )

    # ... later in the "Volume Lost (kL)" block:
    vol_kl = inc.get("ui_total_volume_kL", inc.get("volume_lost_kL", 0.0))
    event_id = inc.get("event_id", "unknown")
    sub_list = [
        mini_progress(
            "MNF",
            subs.get("MNF", 0),
            tooltip_text="Minimum Night Flow: Detects elevated flow during night hours (12am-4am) when usage should be minimal.",
            tooltip_id=f"tt-mnf-{event_id}",
        ),
        mini_progress(
            "RESIDUAL",
            subs.get("RESIDUAL", 0),
            tooltip_text="Residual Analysis: Compares actual after-hours flow to expected hourly patterns based on historical data.",
            tooltip_id=f"tt-res-{event_id}",
        ),
        mini_progress(
            "CUSUM",
            subs.get("CUSUM", 0),
            tooltip_text="Cumulative Sum: Statistical test detecting sustained shifts in water consumption over time.",
            tooltip_id=f"tt-cusum-{event_id}",
        ),
        mini_progress(
            "AFTERHRS",
            subs.get("AFTERHRS", 0),
            tooltip_text="After Hours: Checks if total consumption outside business hours (4pm-7am) is abnormally high.",
            tooltip_id=f"tt-afthr-{event_id}",
        ),
        mini_progress(
            "BURST/BF",
            subs.get("BURSTBF", 0),
            tooltip_text="Burst/Between-Fixture: Detects sudden spikes or erratic patterns suggesting burst pipes or cycling equipment.",
            tooltip_id=f"tt-burst-{event_id}",
        ),
    ]
    body = dbc.CardBody(
        [
            html.Div(
                [
                    html.H5(
                        [
                            html.Span("🔎 Event "),
                            html.Code(inc["event_id"], className="ms-1"),
                        ],
                        className="mb-2",
                    ),
                    html.Div(incident_badges(inc), className="mb-2"),
                ]
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Small("Start"),
                            html.Div(str(pd.to_datetime(inc["start_day"]).date())),
                        ],
                        className="me-3",
                    ),
                    html.Div(
                        [
                            html.Small("Last"),
                            html.Div(str(pd.to_datetime(inc["last_day"]).date())),
                        ],
                        className="me-3",
                    ),
                    html.Div(
                        [
                            html.Small("Alert"),
                            html.Div(
                                str(
                                    pd.to_datetime(
                                        inc.get("alert_date", inc["last_day"])
                                    ).date()
                                )
                            ),
                        ]
                    ),
                ],
                className="d-flex flex-wrap gap-2 mb-2",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Small(
                                [
                                    "Max ΔNF (L/h) ",
                                    html.Span(
                                        "ℹ️",
                                        id=f"tooltip-deltanf-{inc['event_id']}",
                                        style={"cursor": "help", "fontSize": "0.7rem"},
                                    ),
                                ]
                            ),
                            dbc.Tooltip(
                                "Delta Night Flow: Increase in night-time water flow (12am-4am) above normal baseline. Shows leak size.",
                                target=f"tooltip-deltanf-{inc['event_id']}",
                                placement="top",
                            ),
                            html.Div(f"{float(inc.get('max_deltaNF',0)):.1f}"),
                        ],
                        className="me-3",
                    ),
                    html.Div(
                        [
                            html.Small(
                                [
                                    "Confidence ",
                                    html.Span(
                                        "ℹ️",
                                        id=f"tooltip-conf-{inc['event_id']}",
                                        style={"cursor": "help", "fontSize": "0.7rem"},
                                    ),
                                ]
                            ),
                            dbc.Tooltip(
                                "System's certainty this is a real leak (not false alarm). Based on signal strength, persistence, and agreement between detection methods.",
                                target=f"tooltip-conf-{inc['event_id']}",
                                placement="top",
                            ),
                            dbc.Progress(
                                value=float(inc.get("confidence", 0)),
                                className="w-100",
                                style={"height": "10px"},
                            ),
                        ],
                        style={"flex": "1"},
                    ),
                ],
                className="d-flex flex-wrap gap-2 mb-2",
            ),
            # Confidence interpretation badge and action guide
            html.Div(
                [
                    dbc.Alert(
                        [
                            html.Div(
                                [
                                    html.Strong(
                                        f"{get_confidence_interpretation(inc.get('confidence', 0))[3]} Confidence: ",
                                        className="me-2",
                                    ),
                                    html.Span(
                                        f"{float(inc.get('confidence', 0)):.0f}% - {get_confidence_interpretation(inc.get('confidence', 0))[1]}"
                                    ),
                                ],
                                className="mb-1",
                            ),
                            html.Small(
                                get_confidence_interpretation(inc.get("confidence", 0))[
                                    2
                                ],
                                className="d-block",
                                style={"lineHeight": "1.3"},
                            ),
                        ],
                        color=get_confidence_interpretation(inc.get("confidence", 0))[
                            0
                        ],
                        className="mb-2 py-2",
                        style={"fontSize": "0.85rem"},
                    ),
                ],
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Small(
                                [
                                    "Leak Score ",
                                    html.Span(
                                        "ℹ️",
                                        id=f"tooltip-score-{inc['event_id']}",
                                        style={"cursor": "help", "fontSize": "0.7rem"},
                                    ),
                                ]
                            ),
                            dbc.Tooltip(
                                "Combined severity score from all 5 detection signals (MNF, RESIDUAL, CUSUM, AFTERHRS, BURSTBF). Higher = more urgent.",
                                target=f"tooltip-score-{inc['event_id']}",
                                placement="top",
                            ),
                            html.H6(
                                f"{float(leak_score):.0f}%", className="text-warning"
                            ),
                        ],
                        className="me-3",
                    ),
                    html.Div(
                        [
                            html.Small(
                                [
                                    "Volume Lost (kL) ",
                                    html.Span(
                                        "ℹ️",
                                        id=f"tooltip-vol-{inc['event_id']}",
                                        style={"cursor": "help", "fontSize": "0.7rem"},
                                    ),
                                ]
                            ),
                            dbc.Tooltip(
                                "Total water wasted since leak started. 1 kL = 1,000 liters. Estimated at ~$2/kL in water costs.",
                                target=f"tooltip-vol-{inc['event_id']}",
                                placement="top",
                            ),
                            html.H6(f"{float(vol_kl):.1f}"),
                        ]
                    ),
                ],
                className="d-flex flex-wrap gap-3 mb-2",
            ),
            html.Div(sub_list),
            # Action status badge (if incident has been acted upon)
            html.Div(
                id={"type": "action-status", "index": inc["event_id"]},
                className="mb-2",
            ),
            # Primary action buttons
            html.Div(
                [
                    dbc.Button(
                        "Select",
                        id={"type": "evt-select", "index": inc["event_id"]},
                        color="secondary",
                        size="sm",
                        className="me-2",
                    ),
                    dbc.Button(
                        "✓ Acknowledge",
                        id={
                            "type": "evt-btn",
                            "index": f"{site_id}||{inc['event_id']}||Acknowledge",
                        },
                        color="success",
                        size="sm",
                        className="me-2",
                    ),
                    dbc.Button(
                        "👁️ Watch",
                        id={
                            "type": "evt-btn",
                            "index": f"{site_id}||{inc['event_id']}||Watch",
                        },
                        color="info",
                        size="sm",
                        className="me-2",
                    ),
                    dbc.Button(
                        "🚨 Escalate",
                        id={
                            "type": "evt-btn",
                            "index": f"{site_id}||{inc['event_id']}||Escalate",
                        },
                        color="danger",
                        size="sm",
                        className="me-2",
                    ),
                ],
                className="mt-2 mb-2",
            ),
            # Secondary action buttons
            html.Div(
                [
                    dbc.Button(
                        "✅ Resolved",
                        id={
                            "type": "evt-btn",
                            "index": f"{site_id}||{inc['event_id']}||Resolved",
                        },
                        color="primary",
                        size="sm",
                        outline=True,
                        className="me-2",
                    ),
                    dbc.Button(
                        "🚫 Ignore",
                        id={
                            "type": "evt-btn",
                            "index": f"{site_id}||{inc['event_id']}||Ignore",
                        },
                        color="secondary",
                        size="sm",
                        outline=True,
                        className="me-2",
                    ),
                ],
                className="mb-2",
            ),
        ]
    )
    return dbc.Card(body, className="shadow-sm rounded-3 mb-3")


# -------------------------
# App & Layout
# -------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
)
app.title = APP_TITLE
server = app.server

# --- Instant, client-only status so the user sees something immediately ---

# --- Instant, client-only status so the user sees something immediately ---
dash.clientside_callback(
    """
    function(nReplay, nResume, nStep, log){
      const trigList = (dash_clientside.callback_context.triggered || []);
      const trig = trigList.length ? trigList[0].prop_id.split('.')[0] : "";
      if (!log) return "";
      if (trig === "btn-replay") return "▶️ Replay started…";
      if (trig === "btn-resume") return "⏭️ Resuming…";
      if (trig === "btn-step")   return "➡️ Stepped 1 day…";
      if (log.includes("Paused on")) return "⏸️ Paused on incident";
      if (log.includes("Replay complete")) return "✅ Replay complete";
      return "";
    }
    """,
    Output("analysis-status", "children", allow_duplicate=True),
    [
        Input("btn-replay", "n_clicks"),
        Input("btn-resume", "n_clicks"),
        Input("btn-step", "n_clicks"),
        State("analysis-log", "children"),
    ],
    prevent_initial_call=True,
)


controls = dbc.Card(
    dbc.CardBody(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H4(
                                "🚰 Controls",
                                className="mb-1",
                                style={"fontSize": "1.5rem", "marginBottom": "0.5rem"},
                            ),
                            html.Small(
                                "Pick a site and a date range → Replay as if realtime (06:00 daily).",
                                className="text-muted",
                                style={"fontSize": "0.9rem"},
                            ),
                        ],
                        className="mb-3",
                    ),
                    # Row 1: Labels
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Label(
                                    "Site / Property",
                                    style={
                                        "fontSize": "0.95rem",
                                        "fontWeight": "500",
                                    },
                                ),
                                md=4,
                            ),
                            dbc.Col(
                                html.Label(
                                    "Date Range",
                                    style={
                                        "fontSize": "0.95rem",
                                        "fontWeight": "500",
                                        "textAlign": "center",
                                        "display": "block",
                                        "width": "100%",
                                    },
                                ),
                                md=5,
                                style={"textAlign": "center"},
                            ),
                            dbc.Col(
                                html.Label(
                                    "Replay Options",
                                    style={
                                        "fontSize": "0.95rem",
                                        "fontWeight": "500",
                                    },
                                ),
                                md=3,
                            ),
                        ],
                        className="mb-2",
                    ),
                    # Row 2: Controls
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Dropdown(
                                    id="site-dd",
                                    options=[
                                        {
                                            "label": "All Sites (Portfolio)",
                                            "value": "ALL_SITES",
                                        }
                                    ]
                                    + [{"label": s, "value": s} for s in ALL_SITES],
                                    placeholder="Select one or more properties",
                                    value=["ALL_SITES"],
                                    multi=True,
                                    clearable=True,
                                    persistence=True,
                                    persistence_type="session",
                                ),
                                md=4,
                            ),
                            dbc.Col(
                                dcc.DatePickerRange(
                                    id="overview-range",
                                    start_date=str(DATE_INDEX[0].date()),
                                    end_date=str(DATE_INDEX[-1].date()),
                                    clearable=False,
                                    display_format="DD/MM/YYYY",
                                    style={"textAlign": "center"},
                                ),
                                md=5,
                                style={"textAlign": "center"},
                            ),
                            dbc.Col(
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
                                ),
                                md=3,
                            ),
                        ],
                        className="g-2 mb-3",
                    ),
                    # Buttons Row
                    html.Div(
                        [
                            dbc.Button(
                                "▶️ Run Replay",
                                id="btn-replay",
                                color="primary",
                                className="me-2",
                                size="sm",
                            ),
                            dbc.Button(
                                "⏭️ Resume",
                                id="btn-resume",
                                color="info",
                                className="me-2",
                                size="sm",
                            ),
                            dbc.Button(
                                "➡️ Wait 1 more day",
                                id="btn-step",
                                color="secondary",
                                size="sm",
                            ),
                            html.Span(id="run-toast"),
                        ],
                        className="mt-2 mb-2",
                    ),
                    html.Div(
                        id="analysis-status",
                        className="text-muted",
                        style={"fontSize": "0.9rem"},
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Small(
                                    "Replay log:",
                                    className="text-muted",
                                    style={"fontSize": "0.85rem"},
                                ),
                                html.Pre(
                                    id="analysis-log",
                                    style={
                                        "whiteSpace": "pre-wrap",
                                        "fontFamily": "monospace",
                                        "margin": 0,
                                        "fontSize": "0.8rem",
                                        "maxHeight": "150px",
                                        "overflowY": "auto",
                                    },
                                ),
                            ],
                            style={"padding": "0.75rem"},
                        ),
                        className="shadow-sm rounded-3 mt-2",
                    ),
                ]
            )
        ]
    ),
    className="shadow-sm rounded-3",
)


tab_overview = dbc.Tab(
    label="Overview",
    tab_id="tab-overview",
    children=[
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            dcc.Graph(
                                id="kpi-total-leaks",
                                config={"displayModeBar": False},
                                style={"height": "140px"},
                            )
                        ),
                        className="shadow-sm rounded-3",
                    ),
                    md=3,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            dcc.Graph(
                                id="kpi-volume",
                                config={"displayModeBar": False},
                                style={"height": "140px"},
                            )
                        ),
                        className="shadow-sm rounded-3",
                    ),
                    md=3,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            dcc.Graph(
                                id="kpi-duration",
                                config={"displayModeBar": False},
                                style={"height": "140px"},
                            )
                        ),
                        className="shadow-sm rounded-3",
                    ),
                    md=3,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            dcc.Graph(
                                id="kpi-mnf",
                                config={"displayModeBar": False},
                                style={"height": "140px"},
                            )
                        ),
                        className="shadow-sm rounded-3",
                    ),
                    md=3,
                ),
            ],
            className="g-3",
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Filter by Category"),
                        dcc.Dropdown(
                            id="category-filter",
                            options=[],
                            value=None,
                            multi=True,
                            placeholder="Select category...",
                            disabled=False,
                        ),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        html.Label("Export Data"),
                        html.Div(
                            [
                                dbc.Button(
                                    "📥 Download Summary CSV",
                                    id="btn-export-summary",
                                    color="success",
                                    outline=True,
                                    size="sm",
                                    className="w-100",
                                ),
                                dcc.Download(id="download-summary-csv"),
                            ]
                        ),
                    ],
                    md=2,
                ),
            ],
            className="mb-3",
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        id="ov-scatter",
                        figure=fig_placeholder(
                            "📈 Duration vs Volume", "Run analysis to populate"
                        ),
                    ),
                    md=6,
                ),
                dbc.Col(
                    dcc.Graph(
                        id="ov-bar",
                        figure=fig_placeholder(
                            "🏷️ Count by Category", "Run analysis to populate"
                        ),
                    ),
                    md=6,
                ),
            ],
            className="g-3",
        ),
        html.Br(),
    ],
)

tab_events = dbc.Tab(
    label="Events & Actions",
    tab_id="tab-events",
    children=[
        html.Br(),
        # Reference Cards: Confidence Scale & Incident Terminology
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H6(
                                    "📊 Confidence Scale Guide",
                                    className="mb-1",
                                    style={
                                        "fontSize": "1.25rem",
                                        "marginBottom": "0.5rem",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                dbc.Badge(
                                                    "🔴",
                                                    color="danger",
                                                    className="me-1",
                                                ),
                                                html.Strong(
                                                    "90-100%:",
                                                    style={"fontSize": "0.95rem"},
                                                ),
                                                html.Span(
                                                    " Very High - URGENT",
                                                    style={"fontSize": "0.95rem"},
                                                ),
                                            ],
                                            style={"marginBottom": "0.4rem"},
                                        ),
                                        html.Div(
                                            [
                                                dbc.Badge(
                                                    "🟠",
                                                    color="warning",
                                                    className="me-1",
                                                ),
                                                html.Strong(
                                                    "70-89%:",
                                                    style={"fontSize": "0.95rem"},
                                                ),
                                                html.Span(
                                                    " High - ACTION REQUIRED",
                                                    style={"fontSize": "0.95rem"},
                                                ),
                                            ],
                                            style={"marginBottom": "0.4rem"},
                                        ),
                                        html.Div(
                                            [
                                                dbc.Badge(
                                                    "🟡", color="info", className="me-1"
                                                ),
                                                html.Strong(
                                                    "50-69%:",
                                                    style={"fontSize": "0.95rem"},
                                                ),
                                                html.Span(
                                                    " Moderate - MONITOR",
                                                    style={"fontSize": "0.95rem"},
                                                ),
                                            ],
                                            style={"marginBottom": "0.4rem"},
                                        ),
                                        html.Div(
                                            [
                                                dbc.Badge(
                                                    "⚪",
                                                    color="secondary",
                                                    className="me-1",
                                                ),
                                                html.Strong(
                                                    "30-49%:",
                                                    style={"fontSize": "0.95rem"},
                                                ),
                                                html.Span(
                                                    " Low - WATCH",
                                                    style={"fontSize": "0.95rem"},
                                                ),
                                            ],
                                            style={"marginBottom": "0.4rem"},
                                        ),
                                        html.Div(
                                            [
                                                dbc.Badge(
                                                    "⚫",
                                                    color="light",
                                                    className="me-1",
                                                    style={"color": "#000"},
                                                ),
                                                html.Strong(
                                                    "<30%:",
                                                    style={"fontSize": "0.95rem"},
                                                ),
                                                html.Span(
                                                    " Very Low - Normal variation",
                                                    style={"fontSize": "0.95rem"},
                                                ),
                                            ],
                                        ),
                                    ],
                                    style={"lineHeight": "1.4"},
                                ),
                                html.Hr(
                                    className="my-1",
                                    style={
                                        "marginTop": "0.5rem",
                                        "marginBottom": "0.5rem",
                                    },
                                ),
                                html.Small(
                                    "💡 Confidence = signal strength + persistence + agreement",
                                    className="text-muted",
                                    style={"fontSize": "0.9rem", "lineHeight": "1.3"},
                                ),
                            ]
                        ),
                        className="shadow-sm rounded-3 mb-3",
                        style={
                            "backgroundColor": "#1e1e1e",
                            "padding": "0.5rem",
                            "height": "300px",
                            "display": "flex",
                            "flexDirection": "column",
                            "overflowY": "auto",
                        },
                    ),
                    md=6,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H6(
                                    "📋 Incident Terminology Guide",
                                    className="mb-1",
                                    style={
                                        "fontSize": "1.25rem",
                                        "marginBottom": "0.5rem",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Strong(
                                                    "Confidence Score (0-100%)",
                                                    className="text-warning",
                                                    style={"fontSize": "0.95rem"},
                                                ),
                                                html.Br(),
                                                html.Small(
                                                    "Certainty based on signal strength, persistence (consecutive days), & agreement",
                                                    className="text-muted",
                                                    style={
                                                        "fontSize": "0.9rem",
                                                        "lineHeight": "1.3",
                                                    },
                                                ),
                                            ],
                                            style={"marginBottom": "0.5rem"},
                                        ),
                                        html.Div(
                                            [
                                                html.Strong(
                                                    "Severity (S1–S5)",
                                                    className="text-danger",
                                                    style={"fontSize": "0.95rem"},
                                                ),
                                                html.Br(),
                                                html.Small(
                                                    "S1: <100 L/h | S2: 100–200 L/h | S3: 200–1K L/h | S4: 1–3 KL/h | S5: >3 KL/h",
                                                    className="text-muted",
                                                    style={
                                                        "fontSize": "0.9rem",
                                                        "lineHeight": "1.3",
                                                    },
                                                ),
                                            ],
                                            style={"marginBottom": "0.5rem"},
                                        ),
                                        html.Div(
                                            [
                                                html.Strong(
                                                    "Duration (# days)",
                                                    className="text-success",
                                                    style={"fontSize": "0.95rem"},
                                                ),
                                                html.Br(),
                                                html.Small(
                                                    "Consecutive days detected & confirmed. Longer = more reliable",
                                                    className="text-muted",
                                                    style={
                                                        "fontSize": "0.9rem",
                                                        "lineHeight": "1.3",
                                                    },
                                                ),
                                            ],
                                        ),
                                    ],
                                    style={"lineHeight": "1.4"},
                                ),
                                html.Hr(
                                    className="my-1",
                                    style={
                                        "marginTop": "0.5rem",
                                        "marginBottom": "0.5rem",
                                    },
                                ),
                                html.Small(
                                    "💡 Example: 'S2 - 4 days' = 100–200 L/h for 4 consecutive days",
                                    className="text-muted",
                                    style={"fontSize": "0.9rem", "lineHeight": "1.3"},
                                ),
                            ]
                        ),
                        className="shadow-sm rounded-3 mb-3",
                        style={
                            "backgroundColor": "#1e1e1e",
                            "padding": "0.5rem",
                            "height": "300px",
                            "display": "flex",
                            "flexDirection": "column",
                            "overflowY": "auto",
                        },
                    ),
                    md=6,
                ),
            ],
            className="g-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H5("📋 Incident List"),
                        dcc.Loading(
                            id="loading-incidents",
                            type="default",  # uses the default spinner
                            children=html.Div(
                                id="incident-list",
                                children=[
                                    dbc.Alert(
                                        "No site selected or no incidents yet. Run analysis.",
                                        color="dark",
                                    )
                                ],
                            ),
                        ),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        html.H5("🔍 Selected Event Details"),
                        html.Div(id="event-detail-header"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(
                                        id="gauge-confidence",
                                        config={"displayModeBar": False},
                                    ),
                                    md=6,
                                ),
                                dbc.Col(
                                    dcc.Graph(
                                        id="chart-confidence-evolution",
                                        config={"displayModeBar": False},
                                    ),
                                    md=6,
                                ),
                            ],
                            className="g-3",
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("Sub-signals"),
                                    html.Div(id="detail-subscores"),
                                ]
                            ),
                            className="shadow-sm rounded-3 mb-3",
                        ),
                        # Interactive Drill-Down Panel with Tabs
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    dbc.Tabs(
                                        [
                                            dbc.Tab(
                                                label="📈 Timeline View",
                                                tab_id="tab-timeline",
                                                label_style={"fontSize": "0.95rem"},
                                            ),
                                            dbc.Tab(
                                                label="📊 Statistical Analysis",
                                                tab_id="tab-statistical",
                                                label_style={"fontSize": "0.95rem"},
                                            ),
                                            dbc.Tab(
                                                label="🔍 Pattern Analysis",
                                                tab_id="tab-pattern",
                                                label_style={"fontSize": "0.95rem"},
                                            ),
                                            dbc.Tab(
                                                label="💰 Impact Assessment",
                                                tab_id="tab-impact",
                                                label_style={"fontSize": "0.95rem"},
                                            ),
                                        ],
                                        id="detail-tabs",
                                        active_tab="tab-timeline",
                                    ),
                                    className="p-0",
                                ),
                                dbc.CardBody(
                                    html.Div(id="detail-tabs-content"),
                                    className="p-2",
                                    style={"minHeight": "600px"},
                                ),
                            ],
                            className="shadow-sm rounded-3 mt-3",
                        ),
                    ],
                    md=8,
                ),
            ],
            className="g-3 mb-4",
        ),
    ],
)


tab_log = dbc.Tab(
    label="Action Log",
    tab_id="tab-log",
    children=[
        html.Br(),
        dash_table.DataTable(
            id="action-table",
            columns=[
                {"name": c, "id": c}
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
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#222"},
            style_cell={
                "backgroundColor": "#151515",
                "border": "1px solid #333",
                "color": "#ddd",
            },
        ),
        html.Div(id="action-log-refresh"),
    ],
)

tabs = dbc.Tabs(
    [tab_overview, tab_events, tab_log],
    id="tabs",
    active_tab="tab-overview",
    persistence=True,
)


app.layout = dbc.Container(
    [
        html.Br(),
        html.H2(APP_TITLE, className="mb-2"),
        html.Div(
            "Modern, responsive, and production-ready. Pick a date → choose a site → Run.",
            className="text-muted mb-3",
        ),
        controls,
        html.Br(),
        tabs,
        # Invisible stores
        dcc.Store(
            id="store-confirmed"
        ),  # confirmed records for selected site (list of dicts)
        dcc.Store(id="store-selected-event"),  # event_id string
        dcc.Store(
            id="store-cutoff-date", data=str(DATE_INDEX[DEFAULT_CUTOFF_IDX].date())
        ),
        dcc.Store(
            id="store-replay",
            data={"current": None, "start": None, "end": None, "reported": []},
        ),
        dcc.Store(id="store-action-context"),  # Stores context for modal actions
        # Action Modals
        # Acknowledge Modal
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("✓ Acknowledge Leak")),
                dbc.ModalBody(
                    [
                        html.P(
                            "Confirm that you've seen this leak and are taking action.",
                            className="mb-3",
                        ),
                        html.Div(id="modal-ack-details", className="mb-3"),
                        dbc.Label("What action are you taking?"),
                        dbc.Textarea(
                            id="modal-ack-notes",
                            placeholder="E.g., 'Sent maintenance to check toilets', 'Called plumber', 'Site inspection scheduled'...",
                            rows=3,
                            className="mb-2",
                        ),
                        dbc.Label("Your name (optional):"),
                        dbc.Input(
                            id="modal-ack-user",
                            placeholder="E.g., John Smith",
                            type="text",
                        ),
                    ]
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "Cancel",
                            id="modal-ack-cancel",
                            color="secondary",
                            size="sm",
                        ),
                        dbc.Button(
                            "✓ Confirm Acknowledge",
                            id="modal-ack-confirm",
                            color="success",
                            size="sm",
                        ),
                    ]
                ),
            ],
            id="modal-acknowledge",
            is_open=False,
            backdrop="static",
        ),
        # Watch Modal
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("👁️ Watch Leak")),
                dbc.ModalBody(
                    [
                        html.P(
                            "Monitor this leak for additional days before taking action.",
                            className="mb-3",
                        ),
                        html.Div(id="modal-watch-details", className="mb-3"),
                        dbc.Label("Why are you watching?"),
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
                                {"label": "Pool fill suspected", "value": "pool_fill"},
                                {
                                    "label": "Low confidence - need confirmation",
                                    "value": "low_confidence",
                                },
                                {"label": "Other", "value": "other"},
                            ],
                            value="waiting_data",
                            className="mb-2",
                        ),
                        dbc.Label("Review in:"),
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
                        ),
                        dbc.Label("Additional notes (optional):"),
                        dbc.Textarea(
                            id="modal-watch-notes",
                            placeholder="Any additional context...",
                            rows=2,
                        ),
                    ]
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "Cancel",
                            id="modal-watch-cancel",
                            color="secondary",
                            size="sm",
                        ),
                        dbc.Button(
                            "👁️ Confirm Watch",
                            id="modal-watch-confirm",
                            color="info",
                            size="sm",
                        ),
                    ]
                ),
            ],
            id="modal-watch",
            is_open=False,
            backdrop="static",
        ),
        # Escalate Modal
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("🚨 Escalate Leak")),
                dbc.ModalBody(
                    [
                        html.P(
                            "Escalate this leak to senior management or emergency services.",
                            className="mb-3",
                        ),
                        html.Div(id="modal-escalate-details", className="mb-3"),
                        dbc.Label("Escalate to:"),
                        dbc.Checklist(
                            id="modal-escalate-to",
                            options=[
                                {"label": "Facilities Manager", "value": "facilities"},
                                {"label": "Regional Manager", "value": "regional"},
                                {"label": "Emergency Plumber", "value": "plumber"},
                                {"label": "Property Manager", "value": "property"},
                            ],
                            value=["facilities"],
                            className="mb-2",
                        ),
                        dbc.Label("Urgency Level:"),
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
                        dbc.Label("Reason for escalation:"),
                        dbc.Textarea(
                            id="modal-escalate-notes",
                            placeholder="E.g., 'Visible flooding reported', 'Large leak with high cost', 'After-hours emergency'...",
                            rows=3,
                        ),
                    ]
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "Cancel",
                            id="modal-escalate-cancel",
                            color="secondary",
                            size="sm",
                        ),
                        dbc.Button(
                            "🚨 Confirm Escalate",
                            id="modal-escalate-confirm",
                            color="danger",
                            size="sm",
                        ),
                    ]
                ),
            ],
            id="modal-escalate",
            is_open=False,
            backdrop="static",
        ),
        # Resolved Modal
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("✅ Mark as Resolved")),
                dbc.ModalBody(
                    [
                        html.P(
                            "Record that this leak has been fixed or confirmed as false alarm.",
                            className="mb-3",
                        ),
                        html.Div(id="modal-resolved-details", className="mb-3"),
                        dbc.Label("Resolution type:"),
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
                        ),
                        dbc.Label("What was found?"),
                        dbc.Select(
                            id="modal-resolved-cause",
                            options=[
                                {"label": "Running toilet", "value": "toilet"},
                                {"label": "Pipe leak", "value": "pipe"},
                                {"label": "Irrigation valve", "value": "irrigation"},
                                {"label": "Pool fill", "value": "pool"},
                                {"label": "Tap left open", "value": "tap"},
                                {"label": "No leak found", "value": "none"},
                                {"label": "Other", "value": "other"},
                            ],
                            value="toilet",
                            className="mb-2",
                        ),
                        dbc.Label("Resolved by:"),
                        dbc.Select(
                            id="modal-resolved-by",
                            options=[
                                {"label": "Maintenance team", "value": "maintenance"},
                                {"label": "Plumber", "value": "plumber"},
                                {"label": "Self-resolved", "value": "self"},
                                {"label": "Other", "value": "other"},
                            ],
                            value="maintenance",
                            className="mb-2",
                        ),
                        dbc.Label("Repair cost (optional):"),
                        dbc.Input(
                            id="modal-resolved-cost",
                            placeholder="E.g., 120.50",
                            type="number",
                            min=0,
                            step=0.01,
                            className="mb-2",
                        ),
                        dbc.Label("Additional notes:"),
                        dbc.Textarea(
                            id="modal-resolved-notes",
                            placeholder="Details about the resolution...",
                            rows=2,
                        ),
                    ]
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "Cancel",
                            id="modal-resolved-cancel",
                            color="secondary",
                            size="sm",
                        ),
                        dbc.Button(
                            "✅ Confirm Resolved",
                            id="modal-resolved-confirm",
                            color="primary",
                            size="sm",
                        ),
                    ]
                ),
            ],
            id="modal-resolved",
            is_open=False,
            backdrop="static",
        ),
        # Ignore Modal
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("🚫 Ignore Leak")),
                dbc.ModalBody(
                    [
                        dbc.Alert(
                            [
                                html.Strong("⚠️ Warning: "),
                                "This will permanently mark these dates as non-leak events and exclude them from future analysis.",
                            ],
                            color="warning",
                            className="mb-3",
                        ),
                        html.Div(id="modal-ignore-details", className="mb-3"),
                        dbc.Label("Reason for ignoring (required):"),
                        dbc.Select(
                            id="modal-ignore-reason",
                            options=[
                                {"label": "False alarm", "value": "false_alarm"},
                                {"label": "Pool fill", "value": "pool_fill"},
                                {"label": "Fire system test", "value": "fire_test"},
                                {
                                    "label": "Planned maintenance",
                                    "value": "maintenance",
                                },
                                {
                                    "label": "Data error/sensor issue",
                                    "value": "data_error",
                                },
                                {
                                    "label": "Known temporary usage",
                                    "value": "temp_usage",
                                },
                                {"label": "Other", "value": "other"},
                            ],
                            value="false_alarm",
                            className="mb-2",
                        ),
                        dbc.Label("Explanation (required):"),
                        dbc.Textarea(
                            id="modal-ignore-notes",
                            placeholder="Provide detailed explanation for ignoring this leak...",
                            rows=3,
                        ),
                    ]
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "Cancel",
                            id="modal-ignore-cancel",
                            color="secondary",
                            size="sm",
                        ),
                        dbc.Button(
                            "🚫 Confirm Ignore",
                            id="modal-ignore-confirm",
                            color="dark",
                            size="sm",
                        ),
                    ]
                ),
            ],
            id="modal-ignore",
            is_open=False,
            backdrop="static",
        ),
        # Toast for notifications
        html.Div(
            dbc.Toast(
                id="action-toast-body",
                header="Action Recorded",
                icon="success",
                duration=4000,
                is_open=False,
                style={
                    "position": "fixed",
                    "top": 66,
                    "right": 10,
                    "width": 350,
                    "zIndex": 9999,
                },
            ),
            id="action-toast",
        ),
    ],
    fluid=True,
    className="pb-5",
)

# -------------------------
# Callbacks
# -------------------------


@app.callback(
    Output("store-cutoff-date", "data"),
    Input("cutoff-slider", "value"),
    prevent_initial_call=True,
)
def _update_cutoff_date(idx):
    try:
        idx = int(idx or 0)
        idx = max(0, min(len(DATE_INDEX) - 1, idx))
        return str(pd.to_datetime(DATE_INDEX[idx]).date())
    except Exception:
        return str(pd.to_datetime(DATE_INDEX[DEFAULT_CUTOFF_IDX]).date())


@app.callback(
    Output("site-dd", "value"),
    Input("site-dd", "options"),
    State("site-dd", "value"),
    prevent_initial_call=False,
)
def _ensure_site_value(options, current):
    if not options:
        return dash.no_update

    valid_values = {opt["value"] for opt in options if "value" in opt}

    # Normalize current to a list
    cur_list = (
        current
        if isinstance(current, (list, tuple))
        else ([current] if current else [])
    )
    cur_list = [v for v in cur_list if v in valid_values]

    # If nothing valid selected, default to ALL_SITES if present, else first option
    if not cur_list:
        return ["ALL_SITES"] if "ALL_SITES" in valid_values else [options[0]["value"]]

    # If ALL_SITES is selected, collapse to just that (avoid mixing with specific sites)
    if "ALL_SITES" in cur_list:
        return ["ALL_SITES"]

    return cur_list


@app.callback(
    [
        Output("run-toast", "children"),
        Output("store-confirmed", "data"),
        Output("category-filter", "options"),
        Output("incident-list", "children"),
        Output("store-replay", "data"),
        Output("store-selected-event", "data"),
    ],
    [
        Input("btn-replay", "n_clicks"),
        Input("btn-resume", "n_clicks"),
        Input("btn-step", "n_clicks"),
    ],
    [
        State("site-dd", "value"),
        State("overview-range", "start_date"),
        State("overview-range", "end_date"),
        State("pause-toggle", "value"),
        State("store-replay", "data"),
    ],
    prevent_initial_call=True,
)
def run_replay(
    n_run,
    n_resume,
    n_step,
    site_sel,  # list or "ALL_SITES"
    start_date,
    end_date,
    pause_vals,
    replay_state,
):
    """
    Day-by-day replay with multi-site selection.

    Outputs (7):
      1) run-toast.children             -> toast/alert
      2) store-confirmed.data           -> list[dict] of confirmed incidents
      3) category-filter.options        -> list of {"label","value"}
      4) incident-list.children         -> list of cards for today's alerts (if any)
      5) analysis-log.children          -> text log
      6) store-replay.data              -> state dict: {"current","start","end","reported"}
      7) store-selected-event.data      -> selected event_id or None
    """
    import time

    t0 = time.perf_counter()
    log_lines = []

    def log_step(txt, level="info"):
        line = f"[{pd.Timestamp.now():%H:%M:%S}] {txt}"
        log_lines.append(line)
        getattr(log, level)(txt)

    # ---------- helpers ----------
    def _resolve_sites(selection):
        vals = (
            selection
            if isinstance(selection, (list, tuple))
            else ([selection] if selection else [])
        )
        if not vals:
            return []
        if "ALL_SITES" in vals:
            return list(ALL_SITES)
        return [s for s in vals if s in ALL_SITES]

    def _peak_score_and_delta_volume(detector, start_day, last_day):
        peak_ls, vol_kL_from_delta = 0.0, 0.0
        if not detector:
            return peak_ls, vol_kL_from_delta
        for d0 in pd.date_range(
            pd.to_datetime(start_day), pd.to_datetime(last_day), freq="D"
        ):
            try:
                subs, ls, deltaNF, NF_MAD = detector.signals_and_score(
                    pd.to_datetime(d0)
                )
                peak_ls = max(peak_ls, float(ls or 0.0))
                vol_kL_from_delta += max(float(deltaNF), 0.0) * 24.0 / 1000.0
            except Exception:
                continue
        return peak_ls, vol_kL_from_delta

    def _enrich_incident(inc, detector, df, site_id_override=None):
        i = dict(inc)

        # ✅ DEBUG: Log what we received
        event_id = i.get("event_id", "unknown")
        has_signal_comp = "signal_components_by_date" in i
        signal_comp_count = len(i.get("signal_components_by_date", {}))
        log.info(
            f"[ENRICH] {event_id}: has_signal_components={has_signal_comp}, count={signal_comp_count}"
        )

        # Ensure site id in multi-site mode
        if site_id_override and not i.get("site_id"):
            i["site_id"] = site_id_override

        # Peak leak score + delta volume
        peak_ls, vol_kL_from_delta = _peak_score_and_delta_volume(
            detector, i["start_day"], i["last_day"]
        )
        ls_val = float(peak_ls or 0.0)
        i["leak_score_ui"] = ls_val
        i["leak_score"] = ls_val

        try:
            if detector:
                # Get max subscores across ALL days in the event (not just last day)
                max_subs = {
                    "MNF": 0,
                    "RESIDUAL": 0,
                    "CUSUM": 0,
                    "AFTERHRS": 0,
                    "BURSTBF": 0,
                }
                for d0 in pd.date_range(
                    pd.to_datetime(i["start_day"]),
                    pd.to_datetime(i["last_day"]),
                    freq="D",
                ):
                    try:
                        subs_d, _, _, _ = detector.signals_and_score(pd.to_datetime(d0))
                        for key in max_subs:
                            max_subs[key] = max(
                                max_subs[key], float(subs_d.get(key, 0))
                            )
                    except Exception:
                        continue
                subs = max_subs

                # Get last day's leak score
                _, ls_last, _, _ = detector.signals_and_score(
                    pd.to_datetime(i["last_day"])
                )
            else:
                subs, ls_last = {
                    "MNF": 0,
                    "RESIDUAL": 0,
                    "CUSUM": 0,
                    "AFTERHRS": 0,
                    "BURSTBF": 0,
                }, 0.0
        except Exception:
            subs, ls_last = {
                "MNF": 0,
                "RESIDUAL": 0,
                "CUSUM": 0,
                "AFTERHRS": 0,
                "BURSTBF": 0,
            }, 0.0
        i["subscores_ui"] = subs

        # Total volume lost
        vol_model = float(i.get("volume_lost_kL", 0.0))
        if vol_model > 0:
            i["ui_total_volume_kL"] = round(vol_model, 2)
        elif vol_kL_from_delta > 0:
            i["ui_total_volume_kL"] = round(vol_kL_from_delta, 2)
        else:
            try:
                st = pd.to_datetime(i["start_day"]).normalize()
                en = pd.to_datetime(i["last_day"]).normalize()
                if ("time" in df) and ("flow" in df) and len(df):
                    m = (df["time"].dt.normalize() >= st) & (
                        df["time"].dt.normalize() <= en
                    )
                    total_L = float(df.loc[m, "flow"].sum())
                else:
                    total_L = 0.0
                i["ui_total_volume_kL"] = round(total_L / 1000.0, 2)
            except Exception:
                i["ui_total_volume_kL"] = 0.0

        # ✅ FIX: Transfer signal_components_by_date from incident to enriched output
        # This is the PRIMARY mechanism for preserving confidence calculations
        if "signal_components_by_date" in i and i["signal_components_by_date"]:
            # Incident already has signal components - preserve them!
            # (These come from the detector's state_machine)
            pass  # Keep the signal_components_by_date as-is

        # LEGACY: Calculate and store daily confidence evolution (DEPRECATED)
        # This is kept for backward compatibility but should NOT recalculate
        # if signal_components_by_date is available
        try:
            if detector and "signal_components_by_date" not in i:
                # Only populate legacy field if new mechanism not available
                start = pd.to_datetime(i["start_day"])
                end = pd.to_datetime(i["last_day"])

                # Load existing stored confidences to preserve them
                existing_confidences = {}
                if (
                    "confidence_evolution_daily" in i
                    and i["confidence_evolution_daily"]
                ):
                    for entry in i["confidence_evolution_daily"]:
                        existing_confidences[entry["date"]] = entry["confidence"]

                daily_confidences = []
                for d in pd.date_range(start, end, freq="D"):
                    d_str = d.strftime("%Y-%m-%d")

                    # PRESERVE existing values - don't recalculate
                    if d_str in existing_confidences:
                        daily_confidences.append(
                            {"date": d_str, "confidence": existing_confidences[d_str]}
                        )
                        continue

                    if d not in detector.daily.index:
                        continue

                    # Only calculate for NEW days not in existing_confidences
                    # WARNING: This recalculates signals - should use cached values!
                    try:
                        sub_scores, _, deltaNF, NF_MAD = detector.signals_and_score(d)
                        # Use actual calendar days from start, not enumeration index
                        persistence_days = (d - start).days + 1
                        conf = detector.get_confidence(
                            sub_scores, persistence_days, deltaNF, NF_MAD
                        )
                        daily_confidences.append(
                            {"date": d_str, "confidence": float(conf)}
                        )
                    except Exception:
                        continue

                i["confidence_evolution_daily"] = daily_confidences
        except Exception:
            i["confidence_evolution_daily"] = []

        return i

    # ---------- guards / inputs ----------
    try:
        selected_sites = _resolve_sites(site_sel)
        if not selected_sites:
            toast = dbc.Alert(
                "Please select at least one site.",
                color="warning",
                duration=4000,
                is_open=True,
            )
            return (
                toast,
                [],
                [],
                [dbc.Alert("No site selected.", color="dark")],
                "\n".join(log_lines),
                {"current": None, "start": None, "end": None, "reported": []},
                None,
            )

        if not (start_date and end_date):
            toast = dbc.Alert(
                "Please pick a valid date range.",
                color="warning",
                duration=4000,
                is_open=True,
            )
            return (
                toast,
                [],
                [],
                [dbc.Alert("No date range.", color="dark")],
                "\n".join(log_lines),
                {"current": None, "start": None, "end": None, "reported": []},
                None,
            )

        start = pd.to_datetime(start_date).normalize()
        end = pd.to_datetime(end_date).normalize()
        if end < start:
            start, end = end, start

        analyze_hour = int(cfg.get("replay", {}).get("analyze_hour", 6))
        pause_on_incident = "pause" in (pause_vals or [])
        max_days = int(cfg.get("replay", {}).get("max_days_per_run", 90))
        days = list(pd.date_range(start, end, freq="D"))
        if len(days) > max_days:
            days = days[:max_days]
            log_step(f"Replay truncated to first {max_days} day(s) for safety.")

        # Determine invocation
        trig = (
            ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "btn-replay"
        )
        state = replay_state or {
            "current": None,
            "start": None,
            "end": None,
            "reported": [],
        }
        reported = set(state.get("reported") or [])

        if trig == "btn-replay" or state.get("start") is None:
            current = days[0]
            state = {
                "current": current.strftime("%Y-%m-%d"),
                "start": start.strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%d"),
                "reported": [],
            }
            sel_label = (
                "ALL_SITES (Portfolio)"
                if set(selected_sites) == set(ALL_SITES)
                else f"{len(selected_sites)} site(s)"
            )
            log_step(
                f"Replay starting at {current.date()} → {end.date()} for {sel_label}"
            )
        else:
            current = pd.to_datetime(state["current"]).normalize()
            if trig in ("btn-resume", "btn-step"):
                current = current + pd.Timedelta(days=1)
                state["current"] = current.strftime("%Y-%m-%d")

        # ---------- main loop ----------
        selected_event_id = None
        left_panel_children = []
        halted = False
        confirmed_df = pd.DataFrame()

        while current <= end:
            session_incidents = []

            up_to_dt = current + pd.Timedelta(hours=analyze_hour)
            log_step(f"Simulating analysis for {current.date()} (up to {up_to_dt})")

            sid_to_det = {}
            incidents = []
            all_confirmed = []

            # portfolio or subset: iterate the chosen sites
            for sid in selected_sites:
                try:
                    sc = compute_or_refresh_site(
                        site_id=sid,
                        up_to_date=up_to_dt,
                        start_date=start,  # overview date range start (consistent)
                        warmup_days=cfg.get("baseline_window_days", 28),
                    )
                except Exception as e:
                    log_step(
                        f"⚠️ {sid}: compute_or_refresh_site failed: {e}", level="warning"
                    )
                    continue

                det = sc.get("detector")
                df_site = sc.get("df", pd.DataFrame())
                cdf = sc.get("confirmed", pd.DataFrame())
                incs = sc.get("incidents", []) or []

                sid_to_det[sid] = det

                # CRITICAL: Preserve confidence values from cached incidents
                # This prevents recalculation when incidents are re-enriched
                cached_incidents = {}
                if sid in SITE_CACHE:
                    for cached_inc in SITE_CACHE[sid].get("incidents", []):
                        event_id = cached_inc.get("event_id")
                        if event_id:
                            cached_incidents[event_id] = {
                                "confidence_evolution_daily": cached_inc.get(
                                    "confidence_evolution_daily", []
                                ),
                                "confidence": cached_inc.get(
                                    "confidence", None
                                ),  # Main gauge confidence
                            }

                # enrich - but INJECT cached confidence values BEFORE enriching
                enriched = []
                for i in incs:
                    event_id = i.get("event_id")
                    # INJECT cached values into incident dict BEFORE enrichment
                    if event_id in cached_incidents:
                        cached = cached_incidents[event_id]
                        if cached.get("confidence_evolution_daily"):
                            i["confidence_evolution_daily"] = cached[
                                "confidence_evolution_daily"
                            ]
                        # PRESERVE the main confidence score from before
                        if cached.get("confidence") is not None:
                            i["_cached_confidence"] = cached["confidence"]

                    enriched_inc = _enrich_incident(
                        i, det, df_site, site_id_override=sid
                    )

                    # RESTORE cached confidence if it was preserved
                    if (
                        "_cached_confidence" in i
                        and i.get("_cached_confidence") is not None
                    ):
                        enriched_inc["confidence"] = i["_cached_confidence"]
                        # Clean up temp field
                        if "_cached_confidence" in enriched_inc:
                            del enriched_inc["_cached_confidence"]

                    enriched.append(enriched_inc)

                incs = normalize_incidents(enriched)
                if sid in SITE_CACHE:
                    SITE_CACHE[sid]["incidents"] = incs
                incidents.extend(incs)

                flt = cfg.get("events_tab_filters", {})
                allowed = set(flt.get("allowed_statuses", [])) or {
                    "WATCH",
                    "INVESTIGATE",
                    "CALL",
                }
                min_ls = float(flt.get("min_leak_score", 0) or 0)
                min_kL = float(flt.get("min_volume_kL", 0) or 0)

                for inc in incs:
                    ad = pd.to_datetime(
                        inc.get("alert_date", inc["last_day"])
                    ).normalize()
                    if (
                        start <= ad <= end
                        and (inc.get("status") in allowed)
                        and float(
                            inc.get("peak_leak_score") or inc.get("leak_score") or 0
                        )
                        >= min_ls
                        and float(
                            inc.get("volume_lost_kL")
                            or inc.get("ui_total_volume_kL")
                            or 0
                        )
                        >= min_kL
                    ):
                        session_incidents.append(inc)

                if isinstance(cdf, pd.DataFrame) and not cdf.empty:
                    cdf = cdf.copy()
                    if "site_id" not in cdf.columns:
                        cdf["site_id"] = sid
                    all_confirmed.append(cdf)

            confirmed_df = (
                pd.concat(all_confirmed, ignore_index=True)
                if all_confirmed
                else pd.DataFrame()
            )

            # pause on today's alerts
            todays = []
            for inc in incidents:
                ad = pd.to_datetime(inc.get("alert_date", inc["last_day"])).normalize()
                status = inc.get("status")
                eid = inc.get("event_id")
                inc_site = inc.get("site_id")

                rep_key = f"{inc_site}::{eid}"
                already_reported = (eid in reported) or (rep_key in reported)

                log_step(
                    f"Check {eid} | site={inc_site} | start={inc.get('start_day')} | "
                    f"last={inc.get('last_day')} | alert={ad.date()} | status={status} | "
                    f"current={current.date()} | already_reported={already_reported}"
                )

                if (
                    (current >= ad)
                    and (status in ("INVESTIGATE", "CALL"))
                    and (not already_reported)
                ):
                    log_step(
                        f"➡️ Triggering pause for {eid} | site={inc_site} | alert={ad.date()} "
                        f"| status={status} | current={current.date()}"
                    )
                    todays.append(inc)

            if todays:
                # mark as reported
                for inc in todays:
                    inc_site = inc.get("site_id")
                    rep_key = f"{inc_site}::{inc.get('event_id')}"
                    reported.add(rep_key)

                # cards for left panel
                left_panel_children = [
                    make_incident_card(
                        inc.get("site_id"), inc, sid_to_det.get(inc.get("site_id"))
                    )
                    for inc in todays
                ]
                selected_event_id = todays[0].get("event_id")
                state["reported"] = list(reported)

                sel_label = (
                    "ALL_SITES (Portfolio)"
                    if set(selected_sites) == set(ALL_SITES)
                    else f"{len(selected_sites)} site(s)"
                )
                log_step(
                    f"⏸️ Pausing on {len(todays)} incident(s) at {current.date()} ({sel_label})"
                )

                if pause_on_incident or trig == "btn-step":
                    left_panel_children = [
                        make_incident_card(
                            inc.get("site_id"), inc, sid_to_det.get(inc.get("site_id"))
                        )
                        for inc in todays
                    ]
                    selected_event_id = todays[0].get("event_id")
                    state["reported"] = list(reported)
                    halted = True
                    break

            # advance one day
            current = current + pd.Timedelta(days=1)
            state["current"] = current.strftime("%Y-%m-%d")

        if not halted:
            # build cards for all incidents seen in the full run window
            left_panel_children = [
                make_incident_card(
                    inc.get("site_id"), inc, None
                )  # or sid_to_det.get(inc.get("site_id"))
                for inc in session_incidents
            ]
            # If you want a stable order:
            session_incidents.sort(
                key=lambda i: (
                    pd.to_datetime(i.get("alert_date", i["last_day"])).value,
                    i.get("site_id", ""),
                )
            )
            # Optionally pick a default selection (e.g., most recent)
            selected_event_id = (
                session_incidents[-1]["event_id"] if session_incidents else None
            )

        # ---------- outputs ----------
        sel_label = (
            "ALL_SITES"
            if set(selected_sites) == set(ALL_SITES)
            else f"{len(selected_sites)} site(s)"
        )
        toast = dbc.Alert(
            ("⏸️ Paused on incident" if halted else "✅ Replay complete")
            + f" | selection={sel_label}",
            color=("warning" if halted else "success"),
            duration=4000,
            is_open=True,
        )

        cat_opts = (
            [
                {"label": c, "value": c}
                for c in confirmed_df["category"].dropna().unique()
            ]
            if isinstance(confirmed_df, pd.DataFrame)
            and (not confirmed_df.empty)
            and ("category" in confirmed_df)
            else []
        )

        if not left_panel_children:
            left_panel_children = [
                dbc.Alert(
                    "No new confirmed incidents on this step. Use Resume/Step to continue.",
                    color="dark",
                )
            ]

        t1 = time.perf_counter()
        log_step(f"Replay {'paused' if halted else 'ended'} in {t1 - t0:.2f}s")

        return (
            toast,
            normalize_incidents(
                confirmed_df.to_dict("records")
                if isinstance(confirmed_df, pd.DataFrame) and not confirmed_df.empty
                else []
            ),
            cat_opts,
            left_panel_children,
            state,  # <-- store-replay.data
            selected_event_id,  # <-- store-selected-event.data
        )

    except Exception as e:
        # Defensive: NEVER return None for multi-output callbacks
        log_step(f"❌ run_replay crashed: {e}", level="error")
        toast = dbc.Alert(f"Error: {e}", color="danger", duration=6000, is_open=True)
        safe_state = replay_state or {
            "current": None,
            "start": None,
            "end": None,
            "reported": [],
        }
        return (
            toast,
            [],
            [],
            "\n".join(log_lines),
            safe_state,
            None,
        )


# ========= Improved OVERVIEW tab =========


def _no_data_fig(title="No data"):
    fig = go.Figure()
    fig.add_annotation(
        text="No data for selected filters",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14),
    )
    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis=dict(visible=False),  # was True
        yaxis=dict(visible=False),  # was True
        margin=dict(l=30, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
    )
    return fig


def _to_df(confirmed_data):
    import pandas as pd

    if not confirmed_data:
        return pd.DataFrame()
    df = pd.DataFrame(confirmed_data).copy()
    # Best-effort parsing
    for c in ("start_day", "last_day", "alert_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    # Ensure site_id exists for portfolio charts
    if "site_id" not in df.columns:
        df["site_id"] = None
    # Duration (hours)
    if {"start_day", "last_day"}.issubset(df.columns):
        dur_days = (
            df["last_day"].dt.normalize() - df["start_day"].dt.normalize()
        ).dt.days + 1
        df["duration_hours"] = (dur_days.clip(lower=1).fillna(0) * 24).astype(float)
    else:
        df["duration_hours"] = None
    # Volume (kL): prefer model field, fall back to UI field
    if "volume_lost_kL" not in df.columns and "ui_total_volume_kL" in df.columns:
        df["volume_lost_kL"] = df["ui_total_volume_kL"]
    if "volume_lost_kL" not in df.columns:
        df["volume_lost_kL"] = 0.0
    # Leak score for bubble sizing (optional)
    if "leak_score_ui" not in df.columns and "leak_score" in df.columns:
        df["leak_score_ui"] = df["leak_score"]
    return df


@app.callback(
    [
        Output("kpi-total-leaks", "figure"),
        Output("kpi-volume", "figure"),
        Output("kpi-duration", "figure"),
        Output("kpi-mnf", "figure"),
        Output("ov-scatter", "figure"),
        Output("ov-bar", "figure"),
    ],
    [
        Input("store-confirmed", "data"),
        Input("category-filter", "value"),
        Input("overview-range", "start_date"),
        Input("overview-range", "end_date"),
    ],
)
def update_overview(confirmed_records, selected_categories, start_date, end_date):
    import numpy as np  # needed in helpers below

    # ---------- KPI tile ----------
    def kpi_fig(title, value, icon):
        is_num = isinstance(value, (int, float, np.floating))
        valid = (value is not None) and (not (is_num and not np.isfinite(value)))
        val = float(value) if valid else 0.0
        valueformat = ",.0f" if float(val).is_integer() else ",.1f"

        fig = go.Figure(
            go.Indicator(
                mode="number",
                value=val,
                title={
                    "text": f"{icon} {title}",
                    "font": {"size": 28, "color": "#999"},
                },
                number={
                    "valueformat": valueformat,
                    "font": {
                        "size": 42,
                        "color": "#fff",
                        "family": "Arial, sans-serif",
                    },
                },
                domain={"x": [0, 1], "y": [0, 1]},
            )
        )
        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=140,
        )
        return fig

    # ---------- Placeholder (no axes) ----------
    def no_data(title):
        f = go.Figure()
        f.add_annotation(
            text="No data for selected filters",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        f.update_layout(
            template="plotly_dark",
            title=title,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=30, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=220,
        )
        return f

    # ---------- Derive Avg MNF (L/h) if not explicitly present ----------
    def _derive_avg_mnf(df):
        # 1) Try explicit MNF-like columns if present
        mnf_candidates = [
            "avg_mnf_Lph",
            "mnf_Lh",
            "mnf",
            "mnf_at_confirm_Lph",
            "nightflow_Lph",
            "night_flow_Lh",
        ]
        for c in mnf_candidates:
            if c in df.columns:
                s = pd.to_numeric(df[c], errors="coerce").dropna()
                if not s.empty:
                    return float(s.mean())

        # 2) Fallback: average leak rate (L/h) from total volume & duration
        if "volume_kL" in df.columns and "duration_hours" in df.columns:
            v = pd.to_numeric(df["volume_kL"], errors="coerce") * 1000.0  # kL → L
            h = pd.to_numeric(df["duration_hours"], errors="coerce")
            rate = (v / h).replace([np.inf, -np.inf], np.nan).dropna()
            if not rate.empty:
                return float(rate.mean())

        # 3) Nothing available
        return None

    # ---------- Build DataFrame (robust) ----------
    df = pd.DataFrame(confirmed_records or [])

    # Define placeholders up-front so returns never reference undefined vars
    summary_fig = no_data("📋 Event Summary")
    bar = no_data("🏷️ Count by Category")

    if df.empty:
        return (
            kpi_fig("Total Leaks", 0, "🚰"),
            kpi_fig("Volume Lost (kL)", 0, "🛢️"),
            kpi_fig("Avg Duration (hrs)", None, "⏱️"),
            kpi_fig("Avg MNF (L/h)", None, "🌙"),
            summary_fig,
            bar,
        )

    # Parse dates if present
    for c in ("start_day", "last_day", "alert_date", "start_time", "end_time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Duration (hours) fallback
    if "duration_hours" not in df.columns and {"start_day", "last_day"}.issubset(
        df.columns
    ):
        dur_days = (
            df["last_day"].dt.normalize() - df["start_day"].dt.normalize()
        ).dt.days + 1
        df["duration_hours"] = (dur_days.clip(lower=1).fillna(0) * 24).astype(float)

    # Unified volume_kL (use best available, safely)
    if "volume_lost_kL" in df.columns:
        df["volume_kL"] = pd.to_numeric(df["volume_lost_kL"], errors="coerce").fillna(
            0.0
        )
    elif "ui_total_volume_kL" in df.columns:
        df["volume_kL"] = pd.to_numeric(
            df["ui_total_volume_kL"], errors="coerce"
        ).fillna(0.0)
    elif "total_volume_L" in df.columns:
        df["volume_kL"] = (
            pd.to_numeric(df["total_volume_L"], errors="coerce").fillna(0.0) / 1000.0
        )
    else:
        df["volume_kL"] = 0.0

    # Category filter (never call .fillna on None)
    cat_series = (
        df["category"]
        if "category" in df.columns
        else pd.Series([None] * len(df), index=df.index)
    ).fillna("Unlabeled")
    if selected_categories:
        cats = (
            selected_categories
            if isinstance(selected_categories, list)
            else [selected_categories]
        )
        keep = cat_series.isin(cats)
        df = df.loc[keep]
        cat_series = cat_series.loc[df.index]

    # Date window (coalesce alert_date → last_day → start_day)
    def coalesce_datetime(cols):
        for c in cols:
            if c in df.columns:
                s = pd.to_datetime(df[c], errors="coerce")
                if not s.isna().all():
                    return s
        return pd.Series([pd.NaT] * len(df), index=df.index)

    ref = coalesce_datetime(["alert_date", "last_day", "start_day"])
    mask = pd.Series(True, index=df.index)
    if start_date:
        mask &= ref >= pd.to_datetime(start_date)
    if end_date:
        mask &= ref <= (pd.to_datetime(end_date) + pd.Timedelta(days=1))
    df = df.loc[mask]
    cat_series = cat_series.loc[df.index]
    ref = ref.loc[df.index]

    if df.empty:
        return (
            kpi_fig("Total Leaks", 0, "🚰"),
            kpi_fig("Volume Lost (kL)", 0, "🛢️"),
            kpi_fig("Avg Duration (hrs)", None, "⏱️"),
            kpi_fig("Avg MNF (L/h)", None, "🌙"),
            summary_fig,
            bar,
        )

    # ---------- KPIs ----------
    total_leaks = int(len(df))
    total_vol = float(pd.to_numeric(df["volume_kL"], errors="coerce").sum())
    avg_dur = (
        float(pd.to_numeric(df.get("duration_hours"), errors="coerce").dropna().mean())
        if "duration_hours" in df
        else None
    )
    avg_mnf = _derive_avg_mnf(df)

    kpi1 = kpi_fig("Total Leaks", total_leaks, "🚰")
    kpi2 = kpi_fig("Volume Lost (kL)", total_vol, "🛢️")
    kpi3 = kpi_fig("Avg Duration (hrs)", avg_dur, "⏱️")
    kpi4 = kpi_fig("Avg MNF (L/h)", avg_mnf, "🌙")

    # ---------- Event Summary table (one row per property) ----------
    df["site_id"] = df.get("site_id", pd.Series(["—"] * len(df), index=df.index))

    # Build aggregation dynamically (only add columns that exist)
    key_col = "event_id" if "event_id" in df.columns else "start_day"
    agg = {key_col: "nunique"}
    if "volume_kL" in df.columns:
        agg["volume_kL"] = "sum"
    if "duration_hours" in df.columns:
        agg["duration_hours"] = "mean"

    tbl = (
        df.groupby(["site_id"], dropna=False)
        .agg(agg)
        .rename(
            columns={
                key_col: "Events",
                "volume_kL": "Total Volume (kL)",
                "duration_hours": "Avg Duration (hrs)",
            }
        )
        .reset_index()
    )

    # Round display columns if present
    if "Total Volume (kL)" in tbl.columns:
        tbl["Total Volume (kL)"] = tbl["Total Volume (kL)"].astype(float).round(1)
    if "Avg Duration (hrs)" in tbl.columns:
        tbl["Avg Duration (hrs)"] = tbl["Avg Duration (hrs)"].astype(float).round(1)

    # Build table header/cells based on available columns
    header_vals = ["Property", "Events"]
    cell_vals = [tbl["site_id"], tbl["Events"]]
    if "Total Volume (kL)" in tbl.columns:
        header_vals.append("Total Volume (kL)")
        cell_vals.append(tbl["Total Volume (kL)"])
    if "Avg Duration (hrs)" in tbl.columns:
        header_vals.append("Avg Duration (hrs)")
        cell_vals.append(tbl["Avg Duration (hrs)"])

    summary_fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=header_vals,
                    fill_color="#2b2b2b",
                    font=dict(color="white", size=24),
                    align="left",
                    height=60,
                ),
                cells=dict(
                    values=cell_vals,
                    align="left",
                    font=dict(size=20),
                    height=50,
                ),
                columnwidth=[360, 160, 320, 320],
            )
        ]
    )
    summary_fig.update_layout(
        template="plotly_dark",
        title=dict(text="📋 Event Summary", font=dict(size=28)),
        margin=dict(l=30, r=20, t=50, b=20),
    )

    # ---------- Count by Category ----------
    by_cat = (
        cat_series.value_counts(dropna=False)
        .rename_axis("category")
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    bar = go.Figure(
        go.Bar(
            x=by_cat["category"],
            y=by_cat["count"],
            text=by_cat["count"],
            textposition="auto",
        )
    )
    bar.update_layout(
        template="plotly_dark",
        title="🏷️ Count by Category",
        xaxis_title="Category",
        yaxis_title="Count",
    )

    return kpi1, kpi2, kpi3, kpi4, summary_fig, bar


# ---- Export Summary CSV
@app.callback(
    Output("download-summary-csv", "data"),
    Input("btn-export-summary", "n_clicks"),
    State("store-confirmed", "data"),
    prevent_initial_call=True,
)
def export_summary(n_clicks, confirmed_data):
    if not confirmed_data:
        raise dash.exceptions.PreventUpdate
    df = pd.DataFrame(confirmed_data)
    if df.empty:
        raise dash.exceptions.PreventUpdate

    # Select key columns for export
    cols = [
        "site_id",
        "event_id",
        "start_day",
        "last_day",
        "category",
        "confidence",
        "severity_max",
        "volume_lost_kL",
        "leak_score",
    ]
    export_df = df[[c for c in cols if c in df.columns]]

    return dcc.send_data_frame(export_df.to_csv, "leak_summary.csv", index=False)


# ---- Incident selection


@app.callback(
    Output("store-selected-event", "data", allow_duplicate=True),
    Input({"type": "evt-select", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_event(n_clicks):

    if not ctx.triggered_id:
        raise dash.exceptions.PreventUpdate
    trig = ctx.triggered_id  # {'type':'evt-select','index': <event_id>}
    return trig["index"]


@app.callback(
    Output("event-detail-header", "children"),
    Output("gauge-confidence", "figure"),
    Output("chart-confidence-evolution", "figure"),
    Output("detail-subscores", "children"),
    Input("store-selected-event", "data"),
    State("site-dd", "value"),
)
def render_event_header(event_id, site_sel):
    """Render event header, gauges, confidence evolution, and subscores"""
    import plotly.graph_objects as go
    import dash_bootstrap_components as dbc

    def _safe_indicator(title, val, suffix="%", vmax=100):
        try:
            v = float(val if val is not None else 0)
        except Exception:
            v = 0.0
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=max(0, min(v, vmax)),
                number={"suffix": suffix},
                gauge={"axis": {"range": [0, vmax]}},
                title={"text": title},
            )
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20), height=200, template="plotly_dark"
        )
        return fig

    # Find incident
    inc, site_for_event = None, None
    if event_id:
        sites = (
            site_sel
            if isinstance(site_sel, (list, tuple))
            else ([site_sel] if site_sel else [])
        )
        candidates = ALL_SITES if "ALL_SITES" in sites else sites

        for sid in candidates:
            sc = SITE_CACHE.get(sid, {})
            for i in sc.get("incidents", []):
                if i.get("event_id") == event_id:
                    inc, site_for_event = i, sid
                    break
            if inc:
                break

    # Default empty state
    if not inc:
        empty_evo_fig = go.Figure()
        empty_evo_fig.update_layout(
            template="plotly_dark",
            height=200,
            margin=dict(l=20, r=20, t=40, b=20),
            title="Confidence Evolution",
        )
        return (
            dbc.Alert(
                "Select an incident to view details", color="info", className="mb-2"
            ),
            _safe_indicator("Confidence", 0),
            empty_evo_fig,
            [],
        )

    # Header
    hdr = html.Div(
        [
            html.H5(
                f"Event {inc.get('event_id', 'Unknown')} — {site_for_event}",
                style={"fontSize": "1.15rem", "marginBottom": "0.4rem"},
            ),
            html.Div(
                f"{pd.to_datetime(inc.get('start_day')).date()} → {pd.to_datetime(inc.get('last_day')).date()} | "
                f"Status: {inc.get('status', '?')} | Severity: {inc.get('severity_max', '?')}",
                className="text-muted",
                style={"fontSize": "0.9rem"},
            ),
        ]
    )

    # Gauges
    conf = inc.get("confidence", 0)
    fig_conf = _safe_indicator("Confidence", conf)

    # Confidence Evolution Chart
    # Get detector for the site to generate evolution chart
    detector = SITE_CACHE.get(site_for_event, {}).get("detector")
    if detector:
        try:
            fig_conf_evo = detector.create_confidence_evolution_mini(inc)
        except Exception as e:
            logging.warning(f"Failed to create confidence evolution chart: {e}")
            fig_conf_evo = go.Figure()
            fig_conf_evo.update_layout(
                template="plotly_dark",
                height=200,
                margin=dict(l=20, r=20, t=40, b=20),
                title="Confidence Evolution",
            )
    else:
        fig_conf_evo = go.Figure()
        fig_conf_evo.update_layout(
            template="plotly_dark",
            height=200,
            margin=dict(l=20, r=20, t=40, b=20),
            title="Confidence Evolution",
        )

    # Subscores
    subs = inc.get("subscores_ui", {})
    chips = [
        dbc.Badge(f"{k}: {int(v*100)}%", color="secondary", className="me-1 mb-1")
        for k, v in subs.items()
    ]

    return hdr, fig_conf, fig_conf_evo, chips


@app.callback(
    Output("detail-tabs-content", "children"),
    [Input("detail-tabs", "active_tab"), Input("store-selected-event", "data")],
    State("site-dd", "value"),
)
def render_tab_content(active_tab, event_id, site_sel):
    """Render content for the active drill-down tab"""
    import plotly.graph_objects as go
    import dash_bootstrap_components as dbc

    def _empty_fig(title):
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(
            title=title,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            height=500,
        )
        return fig

    # Find incident and detector
    inc, detector, site_for_event = None, None, None
    if event_id:
        sites = (
            site_sel
            if isinstance(site_sel, (list, tuple))
            else ([site_sel] if site_sel else [])
        )
        candidates = ALL_SITES if "ALL_SITES" in sites else sites

        for sid in candidates:
            sc = SITE_CACHE.get(sid, {})
            for i in sc.get("incidents", []):
                if i.get("event_id") == event_id:
                    inc = i
                    detector = sc.get("detector")
                    site_for_event = sid
                    break
            if inc:
                break

    if not inc or not detector:
        return dbc.Alert(
            "No incident selected or detector unavailable", color="warning"
        )

    # Get figures from detector
    try:
        log.info(f"=" * 60)
        log.info(f"Generating figures for event {event_id}")
        log.info(f"Detector type: {type(detector)}")
        log.info(f"Has to_plotly_figs: {hasattr(detector, 'to_plotly_figs')}")
        log.info(
            f"Detector has df: {hasattr(detector, 'df')}, df length: {len(detector.df) if hasattr(detector, 'df') else 0}"
        )
        log.info(
            f"Detector has daily: {hasattr(detector, 'daily')}, daily length: {len(detector.daily) if hasattr(detector, 'daily') else 0}"
        )
        log.info(f"Incident keys: {list(inc.keys())}")
        log.info(
            f"Incident start_day: {inc.get('start_day')}, last_day: {inc.get('last_day')}"
        )
        log.info(f"Incident event_id: {inc.get('event_id')}")

        if hasattr(detector, "to_plotly_figs"):
            log.info("Calling detector.to_plotly_figs()...")
            figs = detector.to_plotly_figs(inc)
            log.info(f"✓ Figures generated successfully!")
            log.info(f"Figures type: {type(figs)}, length: {len(figs) if figs else 0}")
        else:
            log.warning("Detector does not have to_plotly_figs method")
            figs = None

        if not figs or len(figs) < 4:
            log.warning(
                f"Insufficient figures returned (got {len(figs) if figs else 0}), using empty figs"
            )
            figs = (
                _empty_fig("Flow"),
                _empty_fig("Night Flow"),
                _empty_fig("After Hours"),
                _empty_fig("Heatmap"),
            )

        flow_fig, nf_fig, ah_fig, heatmap_fig = figs[:4]
        log.info("Figures assigned successfully")

    except Exception as e:
        import traceback

        log.error(f"Error generating figures: {e}")
        log.error(traceback.format_exc())
        figs = (
            _empty_fig("Flow"),
            _empty_fig("Night Flow"),
            _empty_fig("After Hours"),
            _empty_fig("Heatmap"),
        )
        flow_fig, nf_fig, ah_fig, heatmap_fig = figs

    # TAB 1: Timeline View
    if active_tab == "tab-timeline":
        log.info(f"Rendering Timeline tab, flow_fig type: {type(flow_fig)}")
        return html.Div(
            [
                dcc.Graph(
                    id="timeline-graph",
                    figure=flow_fig,
                    config={
                        "displayModeBar": True,
                        "toImageButtonOptions": {"format": "png"},
                    },
                ),
            ]
        )

    # TAB 2: Statistical Analysis
    elif active_tab == "tab-statistical":
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(figure=nf_fig, config={"displayModeBar": True}),
                            md=6,
                        ),
                        dbc.Col(
                            dcc.Graph(figure=ah_fig, config={"displayModeBar": True}),
                            md=6,
                        ),
                    ]
                ),
            ]
        )

    # TAB 3: Pattern Analysis
    elif active_tab == "tab-pattern":
        return html.Div(
            [
                dcc.Graph(figure=heatmap_fig, config={"displayModeBar": True}),
            ]
        )

    # TAB 4: Impact Assessment
    elif active_tab == "tab-impact":
        # Calculate impact metrics
        vol_kL = inc.get("volume_lost_kL", inc.get("ui_total_volume_kL", 0))
        max_delta = inc.get("max_deltaNF", 0)
        duration = inc.get("days_persisted", 0)
        cost_estimate = vol_kL * 2.0  # $2/kL

        severity = inc.get("severity_max", "S1")
        category = inc.get("category", "Unknown")

        # Create impact cards
        impact_content = html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H3(
                                                    f"{vol_kL:.1f}",
                                                    className="text-danger",
                                                ),
                                                html.P(
                                                    "kL Lost",
                                                    className="text-muted mb-0",
                                                ),
                                            ]
                                        )
                                    ],
                                    className="text-center shadow-sm",
                                )
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H3(
                                                    f"${cost_estimate:.2f}",
                                                    className="text-warning",
                                                ),
                                                html.P(
                                                    "Est. Cost",
                                                    className="text-muted mb-0",
                                                ),
                                            ]
                                        )
                                    ],
                                    className="text-center shadow-sm",
                                )
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H3(
                                                    f"{duration}", className="text-info"
                                                ),
                                                html.P(
                                                    "Days", className="text-muted mb-0"
                                                ),
                                            ]
                                        )
                                    ],
                                    className="text-center shadow-sm",
                                )
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H3(
                                                    f"{max_delta:.0f}",
                                                    className="text-primary",
                                                ),
                                                html.P(
                                                    "L/h Peak",
                                                    className="text-muted mb-0",
                                                ),
                                            ]
                                        )
                                    ],
                                    className="text-center shadow-sm",
                                )
                            ],
                            md=3,
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H6("Classification", className="mb-0")
                                        ),
                                        dbc.CardBody(
                                            [
                                                html.Div(
                                                    [
                                                        html.Strong("Severity: "),
                                                        dbc.Badge(
                                                            severity,
                                                            color="danger",
                                                            className="ms-2",
                                                        ),
                                                    ],
                                                    className="mb-2",
                                                ),
                                                html.Div(
                                                    [
                                                        html.Strong("Category: "),
                                                        dbc.Badge(
                                                            category,
                                                            color="info",
                                                            className="ms-2",
                                                        ),
                                                    ],
                                                    className="mb-2",
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="shadow-sm",
                                )
                            ],
                            md=6,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H6(
                                                "Financial Impact", className="mb-0"
                                            )
                                        ),
                                        dbc.CardBody(
                                            [
                                                html.Div(
                                                    [
                                                        html.Strong(
                                                            "Total Water Cost: "
                                                        ),
                                                        html.Span(
                                                            f"${cost_estimate:.2f}",
                                                            className="text-danger ms-2",
                                                        ),
                                                    ],
                                                    className="mb-2",
                                                ),
                                                html.Div(
                                                    [
                                                        html.Strong("Daily Cost: "),
                                                        html.Span(
                                                            f"${cost_estimate/max(duration,1):.2f}",
                                                            className="text-warning ms-2",
                                                        ),
                                                    ],
                                                    className="mb-2",
                                                ),
                                                html.Small(
                                                    "Rate: $2.00 per kL",
                                                    className="text-muted d-block mt-3",
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="shadow-sm",
                                )
                            ],
                            md=6,
                        ),
                    ],
                    className="mb-4",
                ),
                # Recommendation panel
                dbc.Card(
                    [
                        dbc.CardHeader(
                            [html.H6("💡 Recommended Actions", className="mb-0")]
                        ),
                        dbc.CardBody(
                            [
                                html.Ul(
                                    [
                                        html.Li(
                                            f"Priority: {'High' if severity in ['S4','S5'] else 'Medium' if severity in ['S2','S3'] else 'Low'}"
                                        ),
                                        html.Li(f"Likely cause: {category}"),
                                        html.Li(
                                            "Inspect: "
                                            + (
                                                "Toilets and fixtures"
                                                if "Fixture" in category
                                                else (
                                                    "Underground pipes"
                                                    if "Pipe" in category
                                                    else (
                                                        "Appliances and cycling equipment"
                                                        if "Appliance" in category
                                                        else (
                                                            "Main supply lines - major break"
                                                            if "Burst" in category
                                                            else "General inspection"
                                                        )
                                                    )
                                                )
                                            )
                                        ),
                                        html.Li(
                                            f"Estimated repair urgency: {'Immediate' if severity in ['S4','S5'] else 'Within 48 hours' if severity == 'S3' else 'Within 1 week'}"
                                        ),
                                    ]
                                )
                            ]
                        ),
                    ],
                    className="shadow-sm",
                    color="dark",
                    outline=True,
                ),
            ]
        )

        return impact_content

    return dbc.Alert("Unknown tab", color="danger")


# ---- Site Analytics
@app.callback(
    [
        Output("action-table", "data"),
        Output("action-log-refresh", "children"),
    ],
    [
        Input({"type": "evt-btn", "index": ALL}, "n_clicks"),
        Input("tabs", "active_tab"),
    ],
    [
        State("site-dd", "value"),
        State("store-selected-event", "data"),
        State("store-cutoff-date", "data"),
    ],
    prevent_initial_call=True,
)
def unified_action_table(
    n_clicks_list, active_tab, site_id, selected_event_id, cutoff_date_str
):
    triggered = ctx.triggered_id

    # Tab click triggered it → just refresh
    if isinstance(triggered, str) and triggered == "tab-log":
        df = safe_read_actions()
        return (
            df.sort_values("timestamp", ascending=False).to_dict("records"),
            dash.no_update,
        )

    # Button click triggered it → log the action
    if not isinstance(triggered, dict):
        raise dash.exceptions.PreventUpdate

    btn_index = triggered["index"]
    try:
        site_from_id, event_id, action = btn_index.split("||")
    except ValueError:
        raise dash.exceptions.PreventUpdate

    inc = None
    if site_from_id in SITE_CACHE:
        for _inc in SITE_CACHE[site_from_id].get("incidents", []):
            if _inc["event_id"] == event_id:
                inc = _inc
                break

    start_day = inc["start_day"] if inc else None
    end_day = inc["last_day"] if inc else None
    status = inc.get("status", "") if inc else ""

    df = append_action_row(site_from_id, event_id, start_day, end_day, status, action)
    data = df.sort_values("timestamp", ascending=False).to_dict("records")
    return data, html.Span(f"Logged {action} on {event_id}", className="text-muted")


# -------------------------
# Action Button and Modal Callbacks
# -------------------------


# Button click opens corresponding modal and stores context
@app.callback(
    [
        Output("modal-acknowledge", "is_open"),
        Output("modal-watch", "is_open"),
        Output("modal-escalate", "is_open"),
        Output("modal-resolved", "is_open"),
        Output("modal-ignore", "is_open"),
        Output("store-action-context", "data"),
        Output("modal-ack-details", "children"),
        Output("modal-watch-details", "children"),
        Output("modal-escalate-details", "children"),
        Output("modal-resolved-details", "children"),
        Output("modal-ignore-details", "children"),
    ],
    [Input({"type": "evt-btn", "index": ALL}, "n_clicks")],
    [State({"type": "evt-btn", "index": ALL}, "id")],
    prevent_initial_call=True,
)
def open_action_modal(n_clicks_list, button_ids):
    """Open the appropriate modal when action button is clicked"""
    if not ctx.triggered_id or not any(n_clicks_list):
        raise dash.exceptions.PreventUpdate

    # Find which button was clicked
    triggered_idx = ctx.triggered_id["index"]
    site_id, event_id, action = triggered_idx.split("||")

    # Get incident details from cache
    inc = None
    if site_id in SITE_CACHE:
        for _inc in SITE_CACHE[site_id].get("incidents", []):
            if _inc["event_id"] == event_id:
                inc = _inc
                break

    if not inc:
        raise dash.exceptions.PreventUpdate

    # Store context for later use
    context = {
        "site_id": site_id,
        "event_id": event_id,
        "action": action,
        "start_day": str(inc.get("start_day", "")),
        "end_day": str(inc.get("last_day", "")),
        "status": inc.get("status", ""),
        "max_deltaNF": float(inc.get("max_deltaNF", 0)),
        "confidence": float(inc.get("confidence", 0)),
        "volume_lost_kL": float(inc.get("volume_lost_kL", 0)),
        "severity": inc.get("severity_max", "S1"),
    }

    # Create incident summary for modal
    details = dbc.Alert(
        [
            html.Strong(f"Property: {site_id}"),
            html.Br(),
            html.Strong(f"Event: {event_id}"),
            html.Br(),
            f"Dates: {pd.to_datetime(inc['start_day']).date()} → {pd.to_datetime(inc['last_day']).date()}",
            html.Br(),
            f"Severity: {inc.get('severity_max', 'S1')} | ΔNF: {inc.get('max_deltaNF', 0):.0f} L/h",
            html.Br(),
            f"Confidence: {inc.get('confidence', 0):.0f}% | Volume Lost: {inc.get('volume_lost_kL', 0):.1f} kL",
        ],
        color="info",
        className="mb-0",
    )

    # Open the appropriate modal
    modals = {
        "Acknowledge": [
            True,
            False,
            False,
            False,
            False,
            context,
            details,
            "",
            "",
            "",
            "",
        ],
        "Watch": [False, True, False, False, False, context, "", details, "", "", ""],
        "Escalate": [
            False,
            False,
            True,
            False,
            False,
            context,
            "",
            "",
            details,
            "",
            "",
        ],
        "Resolved": [
            False,
            False,
            False,
            True,
            False,
            context,
            "",
            "",
            "",
            details,
            "",
        ],
        "Ignore": [False, False, False, False, True, context, "", "", "", "", details],
    }

    return modals.get(action, [False] * 5 + [None] + [""] * 5)


# Acknowledge modal confirmation
@app.callback(
    [
        Output("modal-acknowledge", "is_open", allow_duplicate=True),
        Output("action-toast", "children"),
        Output("action-log-refresh", "children", allow_duplicate=True),
    ],
    [
        Input("modal-ack-confirm", "n_clicks"),
        Input("modal-ack-cancel", "n_clicks"),
    ],
    [
        State("store-action-context", "data"),
        State("modal-ack-notes", "value"),
        State("modal-ack-user", "value"),
    ],
    prevent_initial_call=True,
)
def confirm_acknowledge(n_confirm, n_cancel, context, notes, user):
    """Handle acknowledge modal confirmation"""
    if not ctx.triggered_id:
        raise dash.exceptions.PreventUpdate

    # Cancel button
    if ctx.triggered_id == "modal-ack-cancel":
        return False, dash.no_update, dash.no_update

    # Confirm button
    if not context:
        raise dash.exceptions.PreventUpdate

    # Log the action
    df = append_action_row(
        site_id=context["site_id"],
        event_id=context["event_id"],
        start_day=context["start_day"],
        end_day=context["end_day"],
        status=context["status"],
        action="Acknowledge",
        notes=notes or "",
        user=user or "Unknown",
    )

    toast = dbc.Toast(
        [
            html.P(f"✓ Acknowledged leak at {context['site_id']}", className="mb-1"),
            html.Small(f"Event: {context['event_id']}", className="text-muted"),
        ],
        header="Action Recorded",
        icon="success",
        duration=4000,
        is_open=True,
        style={
            "position": "fixed",
            "top": 66,
            "right": 10,
            "width": 350,
            "zIndex": 9999,
        },
    )

    return False, toast, html.Span(f"✓ Acknowledged", className="text-success")


# Watch modal confirmation
@app.callback(
    [
        Output("modal-watch", "is_open", allow_duplicate=True),
        Output("action-toast", "children", allow_duplicate=True),
        Output("action-log-refresh", "children", allow_duplicate=True),
    ],
    [
        Input("modal-watch-confirm", "n_clicks"),
        Input("modal-watch-cancel", "n_clicks"),
    ],
    [
        State("store-action-context", "data"),
        State("modal-watch-reason", "value"),
        State("modal-watch-days", "value"),
        State("modal-watch-notes", "value"),
    ],
    prevent_initial_call=True,
)
def confirm_watch(n_confirm, n_cancel, context, reason, days, notes):
    """Handle watch modal confirmation"""
    if not ctx.triggered_id:
        raise dash.exceptions.PreventUpdate

    if ctx.triggered_id == "modal-watch-cancel":
        return False, dash.no_update, dash.no_update

    if not context:
        raise dash.exceptions.PreventUpdate

    # Calculate reminder date
    reminder_date = pd.Timestamp.now() + pd.Timedelta(days=int(days))

    # Map reason codes to readable text
    reason_text = {
        "waiting_data": "Waiting for more data",
        "check_maintenance": "Checking for maintenance activity",
        "pool_fill": "Pool fill suspected",
        "low_confidence": "Low confidence - need confirmation",
        "other": "Other",
    }.get(reason, reason)

    combined_notes = f"Reason: {reason_text}. Review in {days} days. {notes or ''}"

    df = append_action_row(
        site_id=context["site_id"],
        event_id=context["event_id"],
        start_day=context["start_day"],
        end_day=context["end_day"],
        status=context["status"],
        action="Watch",
        notes=combined_notes,
        reason=reason_text,
        reminder_date=str(reminder_date.date()),
    )

    toast = dbc.Toast(
        [
            html.P(f"👁️ Watching leak at {context['site_id']}", className="mb-1"),
            html.Small(f"Will review in {days} days", className="text-muted"),
        ],
        header="Watch Set",
        icon="info",
        duration=4000,
        is_open=True,
        style={
            "position": "fixed",
            "top": 66,
            "right": 10,
            "width": 350,
            "zIndex": 9999,
        },
    )

    return False, toast, html.Span(f"👁️ Watching", className="text-info")


# Escalate modal confirmation
@app.callback(
    [
        Output("modal-escalate", "is_open", allow_duplicate=True),
        Output("action-toast", "children", allow_duplicate=True),
        Output("action-log-refresh", "children", allow_duplicate=True),
    ],
    [
        Input("modal-escalate-confirm", "n_clicks"),
        Input("modal-escalate-cancel", "n_clicks"),
    ],
    [
        State("store-action-context", "data"),
        State("modal-escalate-to", "value"),
        State("modal-escalate-urgency", "value"),
        State("modal-escalate-notes", "value"),
    ],
    prevent_initial_call=True,
)
def confirm_escalate(n_confirm, n_cancel, context, escalate_to, urgency, notes):
    """Handle escalate modal confirmation"""
    if not ctx.triggered_id:
        raise dash.exceptions.PreventUpdate

    if ctx.triggered_id == "modal-escalate-cancel":
        return False, dash.no_update, dash.no_update

    if not context:
        raise dash.exceptions.PreventUpdate

    # Map escalation targets
    target_map = {
        "facilities": "Facilities Manager",
        "regional": "Regional Manager",
        "plumber": "Emergency Plumber",
        "property": "Property Manager",
    }
    escalated_to_text = ", ".join([target_map.get(t, t) for t in escalate_to])

    urgency_map = {
        "standard": "Standard",
        "urgent": "Urgent",
        "emergency": "Emergency",
    }
    urgency_text = urgency_map.get(urgency, urgency)

    combined_notes = (
        f"Escalated to: {escalated_to_text}. Urgency: {urgency_text}. {notes or ''}"
    )

    df = append_action_row(
        site_id=context["site_id"],
        event_id=context["event_id"],
        start_day=context["start_day"],
        end_day=context["end_day"],
        status=context["status"],
        action="Escalate",
        notes=combined_notes,
        escalated_to=escalated_to_text,
        urgency=urgency_text,
    )

    toast = dbc.Toast(
        [
            html.P(f"🚨 Escalated leak at {context['site_id']}", className="mb-1"),
            html.Small(
                f"To: {escalated_to_text} ({urgency_text})", className="text-muted"
            ),
        ],
        header="Escalated",
        icon="danger",
        duration=4000,
        is_open=True,
        style={
            "position": "fixed",
            "top": 66,
            "right": 10,
            "width": 350,
            "zIndex": 9999,
        },
    )

    return False, toast, html.Span(f"🚨 Escalated", className="text-danger")


# Resolved modal confirmation
@app.callback(
    [
        Output("modal-resolved", "is_open", allow_duplicate=True),
        Output("action-toast", "children", allow_duplicate=True),
        Output("action-log-refresh", "children", allow_duplicate=True),
    ],
    [
        Input("modal-resolved-confirm", "n_clicks"),
        Input("modal-resolved-cancel", "n_clicks"),
    ],
    [
        State("store-action-context", "data"),
        State("modal-resolved-type", "value"),
        State("modal-resolved-cause", "value"),
        State("modal-resolved-by", "value"),
        State("modal-resolved-cost", "value"),
        State("modal-resolved-notes", "value"),
    ],
    prevent_initial_call=True,
)
def confirm_resolved(
    n_confirm, n_cancel, context, res_type, cause, resolved_by, cost, notes
):
    """Handle resolved modal confirmation"""
    if not ctx.triggered_id:
        raise dash.exceptions.PreventUpdate

    if ctx.triggered_id == "modal-resolved-cancel":
        return False, dash.no_update, dash.no_update

    if not context:
        raise dash.exceptions.PreventUpdate

    # Map values to readable text
    type_map = {
        "fixed": "Leak fixed",
        "false_alarm": "False alarm confirmed",
        "not_found": "No leak found",
        "other": "Other",
    }
    cause_map = {
        "toilet": "Running toilet",
        "pipe": "Pipe leak",
        "irrigation": "Irrigation valve",
        "pool": "Pool fill",
        "tap": "Tap left open",
        "none": "No leak found",
        "other": "Other",
    }
    by_map = {
        "maintenance": "Maintenance team",
        "plumber": "Plumber",
        "self": "Self-resolved",
        "other": "Other",
    }

    resolution_text = type_map.get(res_type, res_type)
    cause_text = cause_map.get(cause, cause)
    by_text = by_map.get(resolved_by, resolved_by)

    cost_text = f" Cost: ${float(cost):.2f}." if cost else ""
    combined_notes = f"{resolution_text}. Found: {cause_text}. By: {by_text}.{cost_text} {notes or ''}"

    df = append_action_row(
        site_id=context["site_id"],
        event_id=context["event_id"],
        start_day=context["start_day"],
        end_day=context["end_day"],
        status=context["status"],
        action="Resolved",
        notes=combined_notes,
        resolution_type=resolution_text,
        resolution_cause=cause_text,
        resolved_by=by_text,
        cost=cost or "",
    )

    toast = dbc.Toast(
        [
            html.P(f"✅ Resolved leak at {context['site_id']}", className="mb-1"),
            html.Small(f"Found: {cause_text} | By: {by_text}", className="text-muted"),
        ],
        header="Leak Resolved",
        icon="success",
        duration=4000,
        is_open=True,
        style={
            "position": "fixed",
            "top": 66,
            "right": 10,
            "width": 350,
            "zIndex": 9999,
        },
    )

    return False, toast, html.Span(f"✅ Resolved", className="text-success")


# Ignore modal confirmation
@app.callback(
    [
        Output("modal-ignore", "is_open", allow_duplicate=True),
        Output("action-toast", "children", allow_duplicate=True),
        Output("action-log-refresh", "children", allow_duplicate=True),
    ],
    [
        Input("modal-ignore-confirm", "n_clicks"),
        Input("modal-ignore-cancel", "n_clicks"),
    ],
    [
        State("store-action-context", "data"),
        State("modal-ignore-reason", "value"),
        State("modal-ignore-notes", "value"),
    ],
    prevent_initial_call=True,
)
def confirm_ignore(n_confirm, n_cancel, context, reason, notes):
    """Handle ignore modal confirmation"""
    if not ctx.triggered_id:
        raise dash.exceptions.PreventUpdate

    if ctx.triggered_id == "modal-ignore-cancel":
        return False, dash.no_update, dash.no_update

    if not context or not notes:  # Notes required for ignore
        return (
            dash.no_update,
            dbc.Toast(
                "Please provide an explanation for ignoring this leak.",
                header="Missing Information",
                icon="warning",
                duration=3000,
                is_open=True,
                style={
                    "position": "fixed",
                    "top": 66,
                    "right": 10,
                    "width": 350,
                    "zIndex": 9999,
                },
            ),
            dash.no_update,
        )

    # Map reason codes
    reason_map = {
        "false_alarm": "False alarm",
        "pool_fill": "Pool fill",
        "fire_test": "Fire system test",
        "maintenance": "Planned maintenance",
        "data_error": "Data error/sensor issue",
        "temp_usage": "Known temporary usage",
        "other": "Other",
    }
    reason_text = reason_map.get(reason, reason)

    combined_notes = f"IGNORED - Reason: {reason_text}. {notes}"

    df = append_action_row(
        site_id=context["site_id"],
        event_id=context["event_id"],
        start_day=context["start_day"],
        end_day=context["end_day"],
        status=context["status"],
        action="Ignore",
        notes=combined_notes,
        reason=reason_text,
    )

    toast = dbc.Toast(
        [
            html.P(f"🚫 Ignored leak at {context['site_id']}", className="mb-1"),
            html.Small(f"Reason: {reason_text}", className="text-muted"),
            html.Br(),
            html.Small(
                "⚠️ Dates will be excluded from future analysis",
                className="text-warning",
            ),
        ],
        header="Leak Ignored",
        icon="warning",
        duration=5000,
        is_open=True,
        style={
            "position": "fixed",
            "top": 66,
            "right": 10,
            "width": 350,
            "zIndex": 9999,
        },
    )

    return False, toast, html.Span(f"🚫 Ignored", className="text-secondary")


# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=True, threaded=True)


# %%
