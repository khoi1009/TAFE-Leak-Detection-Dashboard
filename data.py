# data.py
# -*- coding: utf-8 -*-
"""
Data loading, caching, and site processing.
"""

import os
import pandas as pd
import numpy as np
from config import cfg, log

# -------------------------
# Engine Import (with fallback)
# -------------------------

ENGINE_OK = True
try:
    from Model_1_realtime_simulation import (
        load_tafe_data,
        validate_config,
        process_site,
        SchoolLeakDetector,
    )

    log.info("Model_1_realtime_simulation loaded successfully.")
except Exception as e:
    ENGINE_OK = False
    log.warning(f"Model engine not available: {e}")

    # Fallback implementations
    from engine_fallback import (
        load_tafe_data,
        validate_config,
        process_site,
        SchoolLeakDetector,
    )

# -------------------------
# Data Loading
# -------------------------


def safe_load_sites():
    """Load site data from file or generate demo data."""
    try:
        if ENGINE_OK and os.path.exists(cfg["data_path"]):
            school_dfs = load_tafe_data(cfg["data_path"])
            log.info(f"Loaded {len(school_dfs)} sites from data_path.")
            return school_dfs
        else:
            demo = load_tafe_data(cfg["data_path"])
            log.info("Using demo data (engine missing or data_path not found).")
            return demo
    except Exception as e:
        log.error(f"Failed to load data: {e}. Generating small demo.")
        return load_tafe_data(cfg["data_path"])


SCHOOL_DFS = safe_load_sites()
ALL_SITES = sorted(list(SCHOOL_DFS.keys()))


def global_date_bounds():
    """Get the min/max date range across all sites."""
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
DEFAULT_CUTOFF_IDX = max(0, len(DATE_INDEX) - 7)

# -------------------------
# In-Memory Cache
# -------------------------

SITE_CACHE = {}


def make_json_safe(inc):
    """Convert incident dict to JSON-serializable form."""
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
    """Apply make_json_safe() to a list of incidents."""
    return [make_json_safe(inc) for inc in (incidents or [])]


def dedupe_by_event_id(inc_list):
    """Deduplicate incidents by event_id, keeping the best one."""
    best = {}
    for inc in inc_list or []:
        eid = inc.get("event_id")
        if not eid:
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


# Re-export for easy import
__all__ = [
    "SCHOOL_DFS",
    "ALL_SITES",
    "DATE_INDEX",
    "DATE_MARKS",
    "DEFAULT_CUTOFF_IDX",
    "SITE_CACHE",
    "make_json_safe",
    "normalize_incidents",
    "dedupe_by_event_id",
    "ENGINE_OK",
]
