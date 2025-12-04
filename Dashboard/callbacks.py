# callbacks.py
# -*- coding: utf-8 -*-
"""
Dashboard callbacks - all interactive functionality.

This module contains the complete business logic for:
- Day-by-day replay simulation
- Overview KPI generation
- Event detail rendering
- Action logging and modals
"""

import time
import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
from dash import ctx, html, dcc, ALL
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from config import DEFAULT_CFG
from data import SITE_CACHE, ALL_SITES, normalize_incidents
from processing import compute_or_refresh_site
from utils import (
    fig_placeholder,
    gauge_figure,
    mini_progress,
    safe_read_actions,
    append_action_row,
    get_confidence_interpretation,
)
from components import make_incident_card

log = logging.getLogger(__name__)


def register_callbacks(app):
    """Register all dashboard callbacks"""

    # -------------------------
    # Clientside callback: Update replay controls state
    # -------------------------
    app.clientside_callback(
        """
        function(replayState) {
            if (!replayState || !replayState.current) {
                return [true, true, 'Not initialized'];
            }
            const current = new Date(replayState.current);
            const end = new Date(replayState.end);
            const isRunning = current < end;
            return [!isRunning, isRunning, current.toDateString()];
        }
        """,
        [
            Output("btn-step", "disabled"),
            Output("btn-resume", "disabled"),
            Output("analysis-status", "children"),
        ],
        Input("store-replay", "data"),
    )

    # -------------------------
    # Helper Functions for Replay
    # -------------------------
    def _resolve_sites(sel):
        """Resolve site selection to list of site IDs"""
        if not sel:
            return ALL_SITES
        if isinstance(sel, str):
            return [sel] if sel in ALL_SITES else ALL_SITES
        return [s for s in sel if s in ALL_SITES] or ALL_SITES

    def _peak_score_and_delta_volume(inc, df_site):
        """Calculate peak leak score and delta volume across incident date range"""
        if df_site is None or df_site.empty:
            return inc

        start = pd.to_datetime(inc.get("start_day"))
        last = pd.to_datetime(inc.get("last_day"))
        if pd.isna(start) or pd.isna(last):
            return inc

        # Handle different DataFrame structures:
        # 1. Time as index (from detector.df after resample)
        # 2. 'time' column (original data)
        # 3. 'Timestamp' column (legacy)
        try:
            if isinstance(df_site.index, pd.DatetimeIndex):
                # Time is in the index
                mask = (df_site.index >= start) & (
                    df_site.index <= last + pd.Timedelta(days=1)
                )
            elif "time" in df_site.columns:
                mask = (df_site["time"] >= start) & (
                    df_site["time"] <= last + pd.Timedelta(days=1)
                )
            elif "Timestamp" in df_site.columns:
                mask = (df_site["Timestamp"] >= start) & (
                    df_site["Timestamp"] <= last + pd.Timedelta(days=1)
                )
            else:
                # No time column found, skip enrichment
                return inc
            slice_df = df_site.loc[mask].copy()
        except Exception:
            return inc

        if slice_df.empty:
            return inc

        if "leak_score" in slice_df.columns:
            peak = slice_df["leak_score"].max()
            if pd.notna(peak):
                inc["peak_leak_score"] = float(peak)

        if "delta_NF" in slice_df.columns:
            max_delta = slice_df["delta_NF"].max()
            if pd.notna(max_delta):
                inc["max_deltaNF"] = float(max_delta)

        return inc

    def _enrich_incident(inc, detector, df_site, site_id_override=None):
        """Add UI-friendly fields to incident for display"""
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

        # Peak leak score + delta volume calculation
        peak_ls = 0.0
        vol_kL_from_delta = 0.0

        # Calculate peak leak score and delta volume (match original implementation)
        if detector:
            for d0 in pd.date_range(
                pd.to_datetime(i["start_day"]),
                pd.to_datetime(i["last_day"]),
                freq="D",
            ):
                try:
                    _, ls, deltaNF, _ = detector.signals_and_score(pd.to_datetime(d0))
                    peak_ls = max(peak_ls, float(ls or 0.0))
                    vol_kL_from_delta += max(float(deltaNF), 0.0) * 24.0 / 1000.0
                except Exception:
                    continue

        ls_val = float(peak_ls or 0.0)
        i["leak_score_ui"] = ls_val
        i["leak_score"] = ls_val

        # Get max subscores across ALL days in the event (match original implementation)
        try:
            if detector:
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
            else:
                subs = {
                    "MNF": 0,
                    "RESIDUAL": 0,
                    "CUSUM": 0,
                    "AFTERHRS": 0,
                    "BURSTBF": 0,
                }
        except Exception:
            subs = {
                "MNF": 0,
                "RESIDUAL": 0,
                "CUSUM": 0,
                "AFTERHRS": 0,
                "BURSTBF": 0,
            }

        i["subscores_ui"] = subs

        # Debug log for subscores
        log.info(f"[ENRICH] Event {event_id}: subscores_ui = {subs}")

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
                if df_site is not None and not df_site.empty:
                    if isinstance(df_site.index, pd.DatetimeIndex):
                        m = (df_site.index.normalize() >= st) & (
                            df_site.index.normalize() <= en
                        )
                        total_L = float(df_site.loc[m, "flow"].sum())
                    elif "time" in df_site.columns and "flow" in df_site.columns:
                        m = (df_site["time"].dt.normalize() >= st) & (
                            df_site["time"].dt.normalize() <= en
                        )
                        total_L = float(df_site.loc[m, "flow"].sum())
                    else:
                        total_L = 0.0
                else:
                    total_L = 0.0
                i["ui_total_volume_kL"] = round(total_L / 1000.0, 2)
            except Exception:
                i["ui_total_volume_kL"] = 0.0

        # Days persisted and duration
        start = pd.to_datetime(i.get("start_day"))
        last = pd.to_datetime(i.get("last_day"))
        if pd.notna(start) and pd.notna(last):
            days = (last.normalize() - start.normalize()).days + 1
            i["days_persisted"] = max(1, days)
            i["duration_hours"] = i["days_persisted"] * 24
        else:
            i["days_persisted"] = 1
            i["duration_hours"] = 24

        # Transfer signal_components_by_date from incident to enriched output
        # This is the PRIMARY mechanism for preserving confidence calculations
        if "signal_components_by_date" in i and i["signal_components_by_date"]:
            pass  # Keep the signal_components_by_date as-is

        # LEGACY: Calculate and store daily confidence evolution
        try:
            if detector and "signal_components_by_date" not in i:
                start_dt = pd.to_datetime(i["start_day"])
                end_dt = pd.to_datetime(i["last_day"])

                # Load existing stored confidences to preserve them
                existing_confidences = {}
                if (
                    "confidence_evolution_daily" in i
                    and i["confidence_evolution_daily"]
                ):
                    for entry in i["confidence_evolution_daily"]:
                        existing_confidences[entry["date"]] = entry["confidence"]

                daily_confidences = []
                for d in pd.date_range(start_dt, end_dt, freq="D"):
                    d_str = d.strftime("%Y-%m-%d")

                    # PRESERVE existing values - don't recalculate
                    if d_str in existing_confidences:
                        daily_confidences.append(
                            {"date": d_str, "confidence": existing_confidences[d_str]}
                        )
                        continue

                    if hasattr(detector, "daily") and d not in detector.daily.index:
                        continue

                    try:
                        sub_scores, _, deltaNF, NF_MAD = detector.signals_and_score(d)
                        persistence_days = (d - start_dt).days + 1
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

    # -------------------------
    # Main Replay Callback
    # -------------------------
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
            Input("btn-step", "n_clicks"),
            Input("btn-resume", "n_clicks"),
        ],
        [
            State("store-replay", "data"),
            State("site-dd", "value"),
            State("overview-range", "start_date"),
            State("overview-range", "end_date"),
            State("pause-toggle", "value"),
        ],
        prevent_initial_call=True,
    )
    def run_replay(
        replay_clicks,
        step_clicks,
        resume_clicks,
        replay_state,
        selected_sites,
        start_date_str,
        end_date_str,
        pause_toggle_value,
    ):
        """Simulate day-by-day leak detection replay"""
        try:
            t0 = time.perf_counter()
            trig = ctx.triggered_id
            log_lines = []

            def log_step(msg, level="info"):
                getattr(log, level)(msg)
                log_lines.append(msg)

            log_step(f"run_replay triggered by: {trig}")

            # Initialize or restore state
            # btn-replay: always start fresh
            # btn-step: advance one day from current position (or initialize if no state)
            # btn-resume: continue running from current position
            if not replay_state or trig == "btn-replay":
                start = pd.to_datetime(start_date_str or "2024-01-01").normalize()
                end = pd.to_datetime(end_date_str or pd.Timestamp.now()).normalize()
                state = {
                    "current": start.strftime("%Y-%m-%d"),
                    "start": start.strftime("%Y-%m-%d"),
                    "end": end.strftime("%Y-%m-%d"),
                    "reported": [],
                }
                log_step(f"Initialized replay: {start.date()} → {end.date()}")
            else:
                state = replay_state.copy()

            current = pd.to_datetime(state["current"]).normalize()
            start = pd.to_datetime(state["start"]).normalize()
            end = pd.to_datetime(state["end"]).normalize()
            reported = set(state.get("reported", []))

            selected_sites = _resolve_sites(selected_sites)
            halted = False
            incidents = []
            session_incidents = []
            all_confirmed = []
            sid_to_det = {}

            # Simulation loop
            while current <= end:
                sim_ts = current + pd.Timedelta(hours=6)

                for sid in selected_sites:
                    try:
                        sc = compute_or_refresh_site(sid, sim_ts)
                    except Exception as e:
                        log_step(
                            f"⚠️ {sid}: compute_or_refresh_site failed: {e}",
                            level="warning",
                        )
                        continue

                    det = sc.get("detector")
                    df_site = sc.get("df", pd.DataFrame())
                    cdf = sc.get("confirmed", pd.DataFrame())
                    incs = sc.get("incidents", []) or []

                    sid_to_det[sid] = det

                    # CRITICAL: Preserve confidence values from cached incidents
                    cached_incidents = {}
                    if sid in SITE_CACHE:
                        for cached_inc in SITE_CACHE[sid].get("incidents", []):
                            event_id = cached_inc.get("event_id")
                            if event_id:
                                cached_incidents[event_id] = {
                                    "confidence_evolution_daily": cached_inc.get(
                                        "confidence_evolution_daily", []
                                    ),
                                    "confidence": cached_inc.get("confidence", None),
                                }

                    # Enrich - but INJECT cached confidence values BEFORE enriching
                    enriched = []
                    for i in incs:
                        event_id = i.get("event_id")
                        if event_id in cached_incidents:
                            cached = cached_incidents[event_id]
                            if cached.get("confidence_evolution_daily"):
                                i["confidence_evolution_daily"] = cached[
                                    "confidence_evolution_daily"
                                ]
                            if cached.get("confidence") is not None:
                                i["_cached_confidence"] = cached["confidence"]

                        enriched_inc = _enrich_incident(
                            i, det, df_site, site_id_override=sid
                        )

                        # RESTORE cached confidence
                        if (
                            "_cached_confidence" in i
                            and i.get("_cached_confidence") is not None
                        ):
                            enriched_inc["confidence"] = i["_cached_confidence"]
                            if "_cached_confidence" in enriched_inc:
                                del enriched_inc["_cached_confidence"]

                        enriched.append(enriched_inc)

                    incs = normalize_incidents(enriched)
                    if sid in SITE_CACHE:
                        SITE_CACHE[sid]["incidents"] = incs
                    incidents.extend(incs)

                    # Filter incidents for session
                    flt = DEFAULT_CFG.get("events_tab_filters", {})
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

                # Pause on today's alerts
                todays = []
                for inc in incidents:
                    ad = pd.to_datetime(
                        inc.get("alert_date", inc["last_day"])
                    ).normalize()
                    status = inc.get("status")
                    eid = inc.get("event_id")
                    inc_site = inc.get("site_id")

                    rep_key = f"{inc_site}::{eid}"
                    already_reported = (eid in reported) or (rep_key in reported)

                    log_step(
                        f"Check {eid} | site={inc_site} | alert={ad.date()} | status={status} | "
                        f"current={current.date()} | already_reported={already_reported}"
                    )

                    if (
                        (current >= ad)
                        and (status in ("INVESTIGATE", "CALL"))
                        and (not already_reported)
                    ):
                        log_step(
                            f"➡️ Triggering pause for {eid} | site={inc_site} | alert={ad.date()}"
                        )
                        todays.append(inc)

                if todays:
                    for inc in todays:
                        inc_site = inc.get("site_id")
                        rep_key = f"{inc_site}::{inc.get('event_id')}"
                        reported.add(rep_key)

                    left_panel_children = [
                        make_incident_card(
                            inc.get("site_id"),
                            inc,
                            sid_to_det.get(inc.get("site_id")),
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

                    # pause_toggle_value is a list like ["pause"] or []
                    pause_on_incident = (
                        pause_toggle_value and "pause" in pause_toggle_value
                    )
                    if pause_on_incident or trig == "btn-step":
                        halted = True
                        break

                current = current + pd.Timedelta(days=1)
                state["current"] = current.strftime("%Y-%m-%d")

            if not halted:
                left_panel_children = [
                    make_incident_card(inc.get("site_id"), inc, None)
                    for inc in session_incidents
                ]
                session_incidents.sort(
                    key=lambda i: (
                        pd.to_datetime(i.get("alert_date", i["last_day"])).value,
                        i.get("site_id", ""),
                    )
                )
                selected_event_id = (
                    session_incidents[-1]["event_id"] if session_incidents else None
                )

            # Outputs
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
                state,
                selected_event_id,
            )

        except Exception as e:
            log.error(f"❌ run_replay crashed: {e}", exc_info=True)
            toast = dbc.Alert(
                f"Error: {e}", color="danger", duration=6000, is_open=True
            )
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
                [dbc.Alert(f"Error: {e}", color="danger")],
                safe_state,
                None,
            )

    # -------------------------
    # Overview Tab Callback
    # -------------------------
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
                        "font": {"size": 18, "color": "#999"},
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
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=100,
            )
            return fig

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

        def _derive_avg_mnf(df):
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

            if "volume_kL" in df.columns and "duration_hours" in df.columns:
                v = pd.to_numeric(df["volume_kL"], errors="coerce") * 1000.0
                h = pd.to_numeric(df["duration_hours"], errors="coerce")
                rate = (v / h).replace([np.inf, -np.inf], np.nan).dropna()
                if not rate.empty:
                    return float(rate.mean())

            return None

        df = pd.DataFrame(confirmed_records or [])
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

        for c in ("start_day", "last_day", "alert_date", "start_time", "end_time"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")

        if "duration_hours" not in df.columns and {"start_day", "last_day"}.issubset(
            df.columns
        ):
            dur_days = (
                df["last_day"].dt.normalize() - df["start_day"].dt.normalize()
            ).dt.days + 1
            df["duration_hours"] = (dur_days.clip(lower=1).fillna(0) * 24).astype(float)

        if "volume_lost_kL" in df.columns:
            df["volume_kL"] = pd.to_numeric(
                df["volume_lost_kL"], errors="coerce"
            ).fillna(0.0)
        elif "ui_total_volume_kL" in df.columns:
            df["volume_kL"] = pd.to_numeric(
                df["ui_total_volume_kL"], errors="coerce"
            ).fillna(0.0)
        elif "total_volume_L" in df.columns:
            df["volume_kL"] = (
                pd.to_numeric(df["total_volume_L"], errors="coerce").fillna(0.0)
                / 1000.0
            )
        else:
            df["volume_kL"] = 0.0

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

        total_leaks = int(len(df))
        total_vol = float(pd.to_numeric(df["volume_kL"], errors="coerce").sum())
        avg_dur = (
            float(
                pd.to_numeric(df.get("duration_hours"), errors="coerce").dropna().mean()
            )
            if "duration_hours" in df
            else None
        )
        avg_mnf = _derive_avg_mnf(df)

        kpi1 = kpi_fig("Total Leaks", total_leaks, "🚰")
        kpi2 = kpi_fig("Volume Lost (kL)", total_vol, "🛢️")
        kpi3 = kpi_fig("Avg Duration (hrs)", avg_dur, "⏱️")
        kpi4 = kpi_fig("Avg MNF (L/h)", avg_mnf, "🌙")

        df["site_id"] = df.get("site_id", pd.Series(["—"] * len(df), index=df.index))

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

        if "Total Volume (kL)" in tbl.columns:
            tbl["Total Volume (kL)"] = tbl["Total Volume (kL)"].astype(float).round(1)
        if "Avg Duration (hrs)" in tbl.columns:
            tbl["Avg Duration (hrs)"] = tbl["Avg Duration (hrs)"].astype(float).round(1)

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

    # -------------------------
    # Export Summary CSV
    # -------------------------
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

        cols = [
            "site_id",
            "event_id",
            "start_day",
            "last_day",
            "status",
            "category",
            "volume_lost_kL",
            "confidence",
        ]
        export_cols = [c for c in cols if c in df.columns]
        export_df = df[export_cols]

        return dcc.send_data_frame(
            export_df.to_csv, "leak_events_summary.csv", index=False
        )

    # -------------------------
    # Event Detail Rendering
    # -------------------------
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
        if detector and hasattr(detector, "create_confidence_evolution_mini"):
            try:
                fig_conf_evo = detector.create_confidence_evolution_mini(inc)
            except Exception as e:
                log.warning(f"Failed to create confidence evolution chart: {e}")
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

        # Subscores - render as badges
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
        """Render drill-down tab content for selected event"""

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

        if not inc:
            return dbc.Alert("No event selected", color="secondary")

        def _empty_fig(title):
            return fig_placeholder(f"{title}")

        # Generate charts from detector
        flow_fig = _empty_fig("Flow Timeline")
        nf_fig = _empty_fig("Night Flow")
        ah_fig = _empty_fig("After Hours")
        heatmap_fig = _empty_fig("Heatmap")

        if detector and hasattr(detector, "to_plotly_figs"):
            try:
                log.info(f"Generating figures for event {event_id}")
                figs = detector.to_plotly_figs(inc)
                if figs and len(figs) >= 4:
                    flow_fig, nf_fig, ah_fig, heatmap_fig = figs[:4]
                    log.info("Figures generated successfully")
            except Exception as e:
                log.error(f"Error generating figures: {e}")
                import traceback

                log.error(traceback.format_exc())

        if active_tab == "tab-timeline":
            return html.Div([dcc.Graph(id="timeline-graph", figure=flow_fig)])

        elif active_tab == "tab-statistical":
            return html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=nf_fig), md=6),
                            dbc.Col(dcc.Graph(figure=ah_fig), md=6),
                        ]
                    )
                ]
            )

        elif active_tab == "tab-pattern":
            return html.Div([dcc.Graph(figure=heatmap_fig)])

        elif active_tab == "tab-impact":
            vol_kL = inc.get("volume_lost_kL", inc.get("ui_total_volume_kL", 0))
            max_delta = inc.get("max_deltaNF", 0)
            duration = inc.get("days_persisted", 0)
            cost_estimate = vol_kL * 2.0

            severity = inc.get("severity_max", "S1")
            category = inc.get("category", "Unknown")

            return html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H3(
                                                f"{vol_kL:.1f}", className="text-danger"
                                            ),
                                            html.P(
                                                "kL Lost", className="text-muted mb-0"
                                            ),
                                        ]
                                    ),
                                    className="text-center shadow-sm",
                                ),
                                md=3,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H3(
                                                f"${cost_estimate:.2f}",
                                                className="text-warning",
                                            ),
                                            html.P(
                                                "Est. Cost", className="text-muted mb-0"
                                            ),
                                        ]
                                    ),
                                    className="text-center shadow-sm",
                                ),
                                md=3,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H3(
                                                f"{duration}", className="text-info"
                                            ),
                                            html.P("Days", className="text-muted mb-0"),
                                        ]
                                    ),
                                    className="text-center shadow-sm",
                                ),
                                md=3,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H3(
                                                f"{max_delta:.0f}",
                                                className="text-primary",
                                            ),
                                            html.P(
                                                "L/h Peak", className="text-muted mb-0"
                                            ),
                                        ]
                                    ),
                                    className="text-center shadow-sm",
                                ),
                                md=3,
                            ),
                        ],
                        className="mb-4",
                    ),
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                html.H6("💡 Recommended Actions", className="mb-0")
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

        return dbc.Alert("Unknown tab", color="danger")

    # -------------------------
    # Action Log Update
    # -------------------------
    @app.callback(
        [Output("action-table", "data"), Output("action-log-refresh", "children")],
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

        if isinstance(triggered, str) and triggered == "tab-log":
            df = safe_read_actions()
            return (
                df.sort_values("timestamp", ascending=False).to_dict("records"),
                dash.no_update,
            )

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

        df = append_action_row(
            site_from_id, event_id, start_day, end_day, status, action
        )
        data = df.sort_values("timestamp", ascending=False).to_dict("records")
        return data, html.Span(f"Logged {action} on {event_id}", className="text-muted")

    # -------------------------
    # Modal Callbacks
    # -------------------------
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
        if not ctx.triggered_id or not any(n_clicks_list):
            raise dash.exceptions.PreventUpdate

        triggered_idx = ctx.triggered_id["index"]
        site_id, event_id, action = triggered_idx.split("||")

        inc = None
        if site_id in SITE_CACHE:
            for _inc in SITE_CACHE[site_id].get("incidents", []):
                if _inc["event_id"] == event_id:
                    inc = _inc
                    break

        if not inc:
            raise dash.exceptions.PreventUpdate

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
            "Watch": [
                False,
                True,
                False,
                False,
                False,
                context,
                "",
                details,
                "",
                "",
                "",
            ],
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
            "Ignore": [
                False,
                False,
                False,
                False,
                True,
                context,
                "",
                "",
                "",
                "",
                details,
            ],
        }

        return modals.get(action, [False] * 5 + [None] + [""] * 5)

    # Individual modal confirmation callbacks
    @app.callback(
        [
            Output("modal-acknowledge", "is_open", allow_duplicate=True),
            Output("action-toast", "children"),
            Output("action-log-refresh", "children", allow_duplicate=True),
        ],
        [Input("modal-ack-confirm", "n_clicks"), Input("modal-ack-cancel", "n_clicks")],
        [
            State("store-action-context", "data"),
            State("modal-ack-notes", "value"),
            State("modal-ack-user", "value"),
        ],
        prevent_initial_call=True,
    )
    def confirm_acknowledge(n_confirm, n_cancel, context, notes, user):
        if not ctx.triggered_id:
            raise dash.exceptions.PreventUpdate

        if ctx.triggered_id == "modal-ack-cancel":
            return False, dash.no_update, dash.no_update

        # Log action
        df = append_action_row(
            context["site_id"],
            context["event_id"],
            context["start_day"],
            context["end_day"],
            context["status"],
            "Acknowledge",
            notes=notes,
            user=user,
        )

        toast = dbc.Alert(
            f"✓ Acknowledged {context['event_id']}",
            color="success",
            duration=3000,
            is_open=True,
        )

        return False, toast, "Refreshed"

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
        [State("store-action-context", "data"), State("modal-watch-notes", "value")],
        prevent_initial_call=True,
    )
    def confirm_watch(n_confirm, n_cancel, context, notes):
        if not ctx.triggered_id:
            raise dash.exceptions.PreventUpdate

        if ctx.triggered_id == "modal-watch-cancel":
            return False, dash.no_update, dash.no_update

        df = append_action_row(
            context["site_id"],
            context["event_id"],
            context["start_day"],
            context["end_day"],
            context["status"],
            "Watch",
            notes=notes,
        )

        toast = dbc.Alert(
            f"👁️ Watching {context['event_id']}",
            color="info",
            duration=3000,
            is_open=True,
        )

        return False, toast, "Refreshed"

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
        [State("store-action-context", "data"), State("modal-escalate-notes", "value")],
        prevent_initial_call=True,
    )
    def confirm_escalate(n_confirm, n_cancel, context, notes):
        if not ctx.triggered_id:
            raise dash.exceptions.PreventUpdate

        if ctx.triggered_id == "modal-escalate-cancel":
            return False, dash.no_update, dash.no_update

        df = append_action_row(
            context["site_id"],
            context["event_id"],
            context["start_day"],
            context["end_day"],
            context["status"],
            "Escalate",
            notes=notes,
        )

        toast = dbc.Alert(
            f"🚨 Escalated {context['event_id']}",
            color="danger",
            duration=3000,
            is_open=True,
        )

        return False, toast, "Refreshed"

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
        [State("store-action-context", "data"), State("modal-resolved-notes", "value")],
        prevent_initial_call=True,
    )
    def confirm_resolved(n_confirm, n_cancel, context, notes):
        if not ctx.triggered_id:
            raise dash.exceptions.PreventUpdate

        if ctx.triggered_id == "modal-resolved-cancel":
            return False, dash.no_update, dash.no_update

        df = append_action_row(
            context["site_id"],
            context["event_id"],
            context["start_day"],
            context["end_day"],
            context["status"],
            "Resolved",
            notes=notes,
        )

        toast = dbc.Alert(
            f"✅ Resolved {context['event_id']}",
            color="primary",
            duration=3000,
            is_open=True,
        )

        return False, toast, "Refreshed"

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
        [State("store-action-context", "data"), State("modal-ignore-notes", "value")],
        prevent_initial_call=True,
    )
    def confirm_ignore(n_confirm, n_cancel, context, notes):
        if not ctx.triggered_id:
            raise dash.exceptions.PreventUpdate

        if ctx.triggered_id == "modal-ignore-cancel":
            return False, dash.no_update, dash.no_update

        df = append_action_row(
            context["site_id"],
            context["event_id"],
            context["start_day"],
            context["end_day"],
            context["status"],
            "Ignore",
            notes=notes,
        )

        toast = dbc.Alert(
            f"🚫 Ignored {context['event_id']}",
            color="dark",
            duration=3000,
            is_open=True,
        )

        return False, toast, "Refreshed"
