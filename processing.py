# processing.py
# -*- coding: utf-8 -*-
"""
Site processing and incident enrichment logic.
"""

import pandas as pd
import numpy as np
from config import cfg, log
from data import SCHOOL_DFS, SITE_CACHE, make_json_safe, dedupe_by_event_id, ENGINE_OK
from engine_fallback import SchoolLeakDetector

# Import pattern matching
try:
    from false_alarm_patterns import (
        match_incident_to_patterns,
        check_should_suppress,
        log_pattern_match,
    )

    HAS_PATTERN_MATCHING = True
except ImportError:
    HAS_PATTERN_MATCHING = False
    log.warning("Pattern matching module not available")


def build_subscores_from_signal_components(incident: dict) -> dict:
    """
    Build subscores_ui dict from signal_components_by_date.
    This allows pattern matching to work before full enrichment.
    Returns max of each subscore across all days in the incident.
    """
    max_subs = {
        "MNF": 0,
        "RESIDUAL": 0,
        "CUSUM": 0,
        "AFTERHRS": 0,
        "BURSTBF": 0,
    }

    signal_components = incident.get("signal_components_by_date", {})
    if not signal_components:
        return max_subs

    for date_key, components in signal_components.items():
        sub_scores = components.get("sub_scores", {})
        for key in max_subs:
            max_subs[key] = max(max_subs[key], float(sub_scores.get(key, 0)))

    return max_subs


# Try to import real engine
try:
    from Model_1_realtime_simulation import process_site as engine_process_site

    HAS_ENGINE = True
except:
    HAS_ENGINE = False


def compute_or_refresh_site(site_id, up_to_date, start_date=None, warmup_days=None):
    """
    Build/refresh the cached detector for a site using a historical slice.

    Parameters
    ----------
    site_id : str
    up_to_date : datetime|str
        Replay cutoff (e.g., 06:00 of the current replay day).
    start_date : datetime|str|None
        If provided, the analysis window starts here.
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

    # Resolve bounds
    up_to = pd.to_datetime(up_to_date)

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
        log.info(f"{site_id}: Warmup of {wd} days; lower bound = {lb.date()}")
    else:
        lb = pd.to_datetime(df_full["time"].min())
        wd = 0
        log.info(f"{site_id}: No start_date; using full history from {lb.date()}")

    if lb > up_to:
        lb = up_to

    mask = (df_full["time"] >= lb) & (df_full["time"] <= up_to)
    df_slice = df_full.loc[mask].copy()
    log.info(
        f"{site_id}: Sliced {len(df_slice)} rows from {lb.date()} to {up_to.date()}"
    )

    # Preserve signal components and frozen confidence from previous detector
    prev_signal_components = {}
    prev_confidence_by_date = {}
    if site_id in SITE_CACHE:
        prev_detector = SITE_CACHE[site_id].get("detector")
        if prev_detector:
            if hasattr(prev_detector, "signal_components_by_date"):
                prev_signal_components = prev_detector.signal_components_by_date.copy()
            if hasattr(prev_detector, "confidence_by_date"):
                prev_confidence_by_date = prev_detector.confidence_by_date.copy()

        # Extract from cached incidents
        cached_incidents = SITE_CACHE[site_id].get("incidents", [])
        for inc in cached_incidents:
            if "signal_components_by_date" in inc:
                for date_key, comp in inc["signal_components_by_date"].items():
                    if "confidence" in comp:
                        prev_confidence_by_date[date_key] = comp["confidence"]

    # Run engine
    detector, confirmed_df = None, pd.DataFrame()
    try:
        if HAS_ENGINE and ENGINE_OK:
            log.info(f"{site_id}: Using ENGINE path")
            _, detector, confirmed_df = engine_process_site(
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
            # Restore preserved values
            if detector:
                if prev_signal_components and hasattr(
                    detector, "signal_components_by_date"
                ):
                    for date_key, components in prev_signal_components.items():
                        if date_key not in detector.signal_components_by_date:
                            detector.signal_components_by_date[date_key] = components
                if prev_confidence_by_date and hasattr(detector, "confidence_by_date"):
                    for date_key, conf_val in prev_confidence_by_date.items():
                        if date_key not in detector.confidence_by_date:
                            detector.confidence_by_date[date_key] = conf_val
        else:
            # Fallback path
            detector = SchoolLeakDetector(df_slice, site_id, cfg, up_to_date=up_to)
            if prev_signal_components and hasattr(
                detector, "signal_components_by_date"
            ):
                detector.signal_components_by_date = prev_signal_components.copy()
            if prev_confidence_by_date and hasattr(detector, "confidence_by_date"):
                detector.confidence_by_date = prev_confidence_by_date.copy()

            if hasattr(detector, "preprocess"):
                detector.preprocess()
            if hasattr(detector, "state_machine"):
                detector.state_machine()

            confirmed_df = pd.DataFrame()
    except Exception as e:
        log.error(f"process_site failed for {site_id}: {e}")
        detector, confirmed_df = None, pd.DataFrame()

    # Normalize confirmed incidents
    if isinstance(confirmed_df, pd.DataFrame) and not confirmed_df.empty:
        if ("start_day" in confirmed_df.columns) and (
            "start_time" not in confirmed_df.columns
        ):
            confirmed_df["start_time"] = confirmed_df["start_day"]
        if ("last_day" in confirmed_df.columns) and (
            "end_time" not in confirmed_df.columns
        ):
            confirmed_df["end_time"] = confirmed_df["last_day"]

    # Collect incidents
    incidents = []
    if detector and hasattr(detector, "incidents"):
        log.info(
            f"[PATTERN_DEBUG] {site_id}: Processing {len(detector.incidents)} incidents for pattern matching"
        )
        for inc in detector.incidents:
            log.info(f"[PATTERN_DEBUG] Processing incident: {inc.get('event_id')}")
            if not inc.get("category"):
                try:
                    if detector and hasattr(detector, "categorize_leak"):
                        cat, _ = detector.categorize_leak(inc)
                        inc["category"] = cat or "Unlabelled"
                except Exception:
                    inc["category"] = "Unlabelled"

            # Pattern matching - check if this incident matches known false alarm patterns
            if HAS_PATTERN_MATCHING:
                try:
                    # Build subscores_ui from signal_components_by_date BEFORE pattern matching
                    # This is needed because full enrichment happens later in callbacks.py
                    if not inc.get("subscores_ui") and inc.get(
                        "signal_components_by_date"
                    ):
                        inc["subscores_ui"] = build_subscores_from_signal_components(
                            inc
                        )
                        log.info(
                            f"[PATTERN_DEBUG] Built subscores_ui from signal_components: {inc['subscores_ui']}"
                        )

                    log.info(
                        f"[PATTERN_DEBUG] Checking patterns for {inc.get('event_id')}"
                    )
                    should_suppress, matching_pattern = check_should_suppress(
                        inc, site_id
                    )
                    log.info(
                        f"[PATTERN_DEBUG] should_suppress={should_suppress}, matching_pattern={matching_pattern}"
                    )
                    if should_suppress and matching_pattern:
                        # Auto-suppress this incident
                        inc["status"] = "Suppressed"
                        inc["suppressed_by_pattern"] = matching_pattern.get(
                            "pattern_id"
                        )
                        inc["pattern_match_score"] = matching_pattern.get("final_score")
                        inc["pattern_category"] = matching_pattern.get("category")
                        log.info(
                            f"Auto-suppressed incident {inc.get('event_id')} - "
                            f"matches pattern {matching_pattern.get('pattern_id')} "
                            f"(score: {matching_pattern.get('final_score'):.2f})"
                        )
                        # Log the match
                        log_pattern_match(
                            incident_id=inc.get("event_id"),
                            pattern_id=matching_pattern.get("pattern_id"),
                            match_score=matching_pattern.get("final_score"),
                            action_taken="suppressed",
                            site_id=site_id,
                        )
                    else:
                        # Check for non-auto-suppress matches (flag for review)
                        log.info(
                            f"[PATTERN_DEBUG] Calling match_incident_to_patterns for {inc.get('event_id')}"
                        )
                        log.info(f"[PATTERN_DEBUG] Incident keys: {list(inc.keys())}")
                        log.info(
                            f"[PATTERN_DEBUG] subscores_ui: {inc.get('subscores_ui')}"
                        )
                        log.info(f"[PATTERN_DEBUG] subscores: {inc.get('subscores')}")
                        matches = match_incident_to_patterns(inc, site_id)
                        log.info(f"[PATTERN_DEBUG] Got {len(matches)} matches")
                        strong_matches = [
                            m for m in matches if m.get("is_strong_match")
                        ]
                        log.info(
                            f"[PATTERN_DEBUG] {len(strong_matches)} strong matches"
                        )
                        if strong_matches:
                            inc["pattern_matches"] = strong_matches[:3]  # Top 3 matches
                            inc["has_pattern_match"] = True
                            log.info(
                                f"🧠 Pattern match found! Incident {inc.get('event_id')} has {len(strong_matches)} "
                                f"match(es), top score: {strong_matches[0].get('final_score'):.0%}"
                            )
                except Exception as e:
                    log.warning(
                        f"Pattern matching failed for {inc.get('event_id')}: {e}"
                    )

            incidents.append(make_json_safe(inc))

    # Deduplicate
    before = len(incidents)
    incidents = dedupe_by_event_id(incidents)
    removed = before - len(incidents)
    if removed > 0:
        log.info(f"{site_id}: de-duplicated {removed} incidents")

    # Propagate categories to confirmed_df
    if isinstance(confirmed_df, pd.DataFrame) and not confirmed_df.empty:
        confirmed_df = confirmed_df.copy()

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

    # Cache & return
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
