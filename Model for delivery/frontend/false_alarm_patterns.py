# false_alarm_patterns.py
# -*- coding: utf-8 -*-
"""
False Alarm Pattern Recording and Matching System

This module provides functionality to:
1. Record false alarm patterns when users mark events as "Ignore"
2. Store patterns with their fingerprints (signals, timing, recurrence)
3. Match new incidents against known patterns
4. Auto-suppress or flag incidents that match known false alarm patterns

Pattern Fingerprint includes:
- Site ID
- Category (pool_fill, fire_test, maintenance, etc.)
- Day of week patterns
- Time window
- Detection signal fingerprint
- Recurrence rules
"""

import os
import json
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from config import log

# ============================================
# CONFIGURATION
# ============================================

PATTERNS_FILE = "False_Alarm_Patterns.csv"
PATTERN_MATCHES_LOG = "Pattern_Matches_Log.csv"

# Matching thresholds
SIGNAL_MATCH_THRESHOLD = 0.7  # 70% signal similarity required
TIME_WINDOW_TOLERANCE_HOURS = 2  # ±2 hours for time matching
CONFIDENCE_DECAY_DAYS = (
    90  # Pattern confidence decays after 90 days without confirmation
)

# Signal weights for fingerprint matching
SIGNAL_WEIGHTS = {
    "mnf": 0.25,
    "cusum": 0.20,
    "afterhrs": 0.15,
    "vol_spike": 0.15,
    "duration": 0.10,
    "weekday": 0.10,
    "time_of_day": 0.05,
}


# ============================================
# PATTERN DATA STRUCTURES
# ============================================


def create_pattern_id(site_id: str, category: str, fingerprint: Dict) -> str:
    """Generate a unique pattern ID based on site, category, and fingerprint."""
    data = f"{site_id}_{category}_{json.dumps(fingerprint, sort_keys=True)}"
    return hashlib.md5(data.encode()).hexdigest()[:12].upper()


def create_signal_fingerprint(incident: Dict) -> Dict:
    """
    Extract the signal fingerprint from an incident.

    This captures which detection signals fired and their relative strengths,
    PLUS critical flow rate metrics for accurate pattern matching.
    """
    fingerprint = {
        "signals_active": [],
        "signal_scores": {},
        "mnf_range": None,
        "mnf_value_Lph": None,  # Actual MNF value in Liters per hour
        "avg_flow_rate_Lph": None,  # Average flow rate during incident
        "peak_flow_rate_Lph": None,  # Peak flow rate observed
        "volume_range": None,
        "volume_kL": None,  # Actual volume in kL
        "duration_range": None,
        "duration_hours": None,  # Actual duration
    }

    # Extract subscores if available
    subscores = incident.get("subscores_ui", {}) or incident.get("subscores", {})
    if subscores:
        for signal, score in subscores.items():
            if score and float(score) > 0.1:  # Signal is active if score > 10%
                fingerprint["signals_active"].append(signal)
                fingerprint["signal_scores"][signal] = round(float(score), 2)

    # Extract MNF (Minimum Night Flow) - CRITICAL for flow rate matching
    # Try multiple possible field names
    mnf = None
    for mnf_field in ["mnf_at_confirm_Lph", "avg_mnf_Lph", "mnf_Lph", "mnf", "MNF"]:
        if mnf_field in incident and incident.get(mnf_field):
            try:
                mnf = float(incident[mnf_field])
                break
            except (ValueError, TypeError):
                continue

    if mnf:
        fingerprint["mnf_value_Lph"] = round(mnf, 2)
        fingerprint["mnf_range"] = [round(mnf * 0.7, 2), round(mnf * 1.3, 2)]  # ±30%

    # Extract average flow rate during the incident
    for flow_field in ["avg_flow_Lph", "avg_flow_rate", "mean_flow", "flow_rate"]:
        if flow_field in incident and incident.get(flow_field):
            try:
                fingerprint["avg_flow_rate_Lph"] = round(float(incident[flow_field]), 2)
                break
            except (ValueError, TypeError):
                continue

    # Extract peak flow rate
    for peak_field in ["peak_flow_Lph", "max_flow", "peak_flow"]:
        if peak_field in incident and incident.get(peak_field):
            try:
                fingerprint["peak_flow_rate_Lph"] = round(
                    float(incident[peak_field]), 2
                )
                break
            except (ValueError, TypeError):
                continue

    # Extract volume
    if "volume_kL" in incident:
        vol = incident.get("volume_kL", 0)
        if vol:
            fingerprint["volume_kL"] = round(float(vol), 2)
            fingerprint["volume_range"] = [
                round(float(vol) * 0.5, 2),
                round(float(vol) * 1.5, 2),
            ]  # ±50%

    # Extract duration
    if "duration_hours" in incident:
        dur = incident.get("duration_hours", 0)
        if dur:
            fingerprint["duration_hours"] = round(float(dur), 2)
            fingerprint["duration_range"] = [
                max(0, round(float(dur) - 6, 2)),
                round(float(dur) + 6, 2),
            ]  # ±6 hours

    return fingerprint


def create_time_fingerprint(incident: Dict) -> Dict:
    """
    Extract time-based fingerprint from an incident.

    Captures day of week, time of day patterns for recurring events.
    """
    fingerprint = {
        "days_of_week": [],
        "time_window_start": None,
        "time_window_end": None,
        "typical_duration_hours": None,
    }

    # Parse start date
    start_day = incident.get("start_day")
    if start_day:
        try:
            dt = pd.to_datetime(start_day)
            fingerprint["days_of_week"] = [dt.dayofweek]  # 0=Monday, 6=Sunday
        except Exception:
            pass

    # If we have hourly data, extract time window
    # For now, use defaults that can be overridden by user

    return fingerprint


# ============================================
# PATTERN STORAGE
# ============================================


def get_patterns_df() -> pd.DataFrame:
    """Load patterns from CSV file."""
    if os.path.exists(PATTERNS_FILE):
        try:
            df = pd.read_csv(PATTERNS_FILE)
            # Parse JSON columns
            for col in ["signal_fingerprint", "time_fingerprint", "recurrence_rule"]:
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda x: json.loads(x) if pd.notna(x) and x else {}
                    )
            return df
        except Exception as e:
            log.error(f"Error loading patterns file: {e}")

    # Return empty DataFrame with schema
    return pd.DataFrame(
        columns=[
            "pattern_id",
            "site_id",
            "category",
            "description",
            "signal_fingerprint",
            "time_fingerprint",
            "recurrence_rule",
            "auto_suppress",
            "confidence",
            "times_matched",
            "times_confirmed_false",
            "times_was_real_leak",
            "created_at",
            "created_by",
            "last_matched_at",
            "last_updated_at",
            "is_active",
            "notes",
        ]
    )


def save_patterns_df(df: pd.DataFrame) -> None:
    """Save patterns DataFrame to CSV."""
    df_to_save = df.copy()

    # Convert dict columns to JSON strings
    for col in ["signal_fingerprint", "time_fingerprint", "recurrence_rule"]:
        if col in df_to_save.columns:
            df_to_save[col] = df_to_save[col].apply(
                lambda x: json.dumps(x) if isinstance(x, dict) else x
            )

    df_to_save.to_csv(PATTERNS_FILE, index=False)
    log.info(f"Saved {len(df_to_save)} patterns to {PATTERNS_FILE}")


def record_pattern(
    site_id: str,
    event_id: str,
    incident: Dict,
    category: str,
    description: str,
    is_recurring: bool = False,
    recurrence_type: str = None,  # daily, weekly, monthly, yearly
    recurrence_days: List[int] = None,  # [0,1,2...] for days of week
    time_window_start: str = None,  # "06:00"
    time_window_end: str = None,  # "08:00"
    auto_suppress: bool = False,
    notes: str = "",
    user: str = "",
) -> Dict:
    """
    Record a new false alarm pattern.

    Args:
        site_id: Property ID
        event_id: Original event ID that was marked as false alarm
        incident: Full incident data dict
        category: Category of false alarm (pool_fill, fire_test, etc.)
        description: Human-readable description
        is_recurring: Whether this is expected to recur
        recurrence_type: Type of recurrence (daily, weekly, monthly, yearly)
        recurrence_days: Days of week for weekly recurrence
        time_window_start: Expected start time (HH:MM)
        time_window_end: Expected end time (HH:MM)
        auto_suppress: Whether to automatically suppress future matches
        notes: Additional notes
        user: User who recorded the pattern

    Returns:
        Dict with pattern_id and success status
    """
    # Create fingerprints
    signal_fp = create_signal_fingerprint(incident)
    time_fp = create_time_fingerprint(incident)

    # Override time fingerprint with user-provided values
    if recurrence_days:
        time_fp["days_of_week"] = recurrence_days
    if time_window_start:
        time_fp["time_window_start"] = time_window_start
    if time_window_end:
        time_fp["time_window_end"] = time_window_end

    # Create recurrence rule
    recurrence_rule = {
        "is_recurring": is_recurring,
        "type": recurrence_type,
        "days_of_week": recurrence_days or time_fp.get("days_of_week", []),
    }

    # Generate pattern ID
    pattern_id = create_pattern_id(site_id, category, signal_fp)

    # Check if similar pattern already exists
    df = get_patterns_df()
    existing = df[(df["site_id"] == site_id) & (df["category"] == category)]

    if not existing.empty:
        # Check signal similarity
        for _, row in existing.iterrows():
            similarity = calculate_signal_similarity(
                signal_fp, row["signal_fingerprint"]
            )
            if similarity > 0.8:  # Very similar pattern exists
                # Update existing pattern instead
                log.info(
                    f"Updating existing pattern {row['pattern_id']} (similarity: {similarity:.2f})"
                )
                df.loc[
                    df["pattern_id"] == row["pattern_id"], "times_confirmed_false"
                ] += 1
                df.loc[df["pattern_id"] == row["pattern_id"], "confidence"] = min(
                    1.0, row["confidence"] + 0.05
                )
                df.loc[df["pattern_id"] == row["pattern_id"], "last_updated_at"] = (
                    datetime.now().isoformat()
                )
                df.loc[df["pattern_id"] == row["pattern_id"], "auto_suppress"] = (
                    auto_suppress
                )
                save_patterns_df(df)
                return {
                    "success": True,
                    "pattern_id": row["pattern_id"],
                    "action": "updated",
                    "message": f"Updated existing pattern {row['pattern_id']}",
                }

    # Create new pattern
    new_pattern = {
        "pattern_id": pattern_id,
        "site_id": site_id,
        "category": category,
        "description": description,
        "signal_fingerprint": signal_fp,
        "time_fingerprint": time_fp,
        "recurrence_rule": recurrence_rule,
        "auto_suppress": auto_suppress,
        "confidence": 0.6,  # Initial confidence
        "times_matched": 0,
        "times_confirmed_false": 1,
        "times_was_real_leak": 0,
        "created_at": datetime.now().isoformat(),
        "created_by": user,
        "last_matched_at": None,
        "last_updated_at": datetime.now().isoformat(),
        "is_active": True,
        "notes": notes,
    }

    df = pd.concat([df, pd.DataFrame([new_pattern])], ignore_index=True)
    save_patterns_df(df)

    log.info(f"Recorded new false alarm pattern: {pattern_id} for site {site_id}")

    return {
        "success": True,
        "pattern_id": pattern_id,
        "action": "created",
        "message": f"Created new pattern {pattern_id}",
    }


# ============================================
# PATTERN MATCHING
# ============================================


def calculate_signal_similarity(fp1: Dict, fp2: Dict) -> float:
    """
    Calculate similarity between two signal fingerprints using weighted feature matching.

    NOTE: This is RULE-BASED similarity matching, NOT machine learning.
    It uses:
    - Jaccard similarity for signal sets
    - Range overlap calculations for numeric features
    - Weighted aggregation of all similarity scores

    Returns value between 0.0 (no match) and 1.0 (perfect match).
    """
    if not fp1 or not fp2:
        return 0.0

    score = 0.0
    weights_used = 0.0

    # Compare active signals (Jaccard similarity)
    signals1 = set(fp1.get("signals_active", []))
    signals2 = set(fp2.get("signals_active", []))

    if signals1 or signals2:
        intersection = len(signals1 & signals2)
        union = len(signals1 | signals2)
        if union > 0:
            signal_sim = intersection / union
            score += signal_sim * 0.30  # 30% weight for signal type match
            weights_used += 0.30

    # Compare signal scores (intensity matching)
    scores1 = fp1.get("signal_scores", {})
    scores2 = fp2.get("signal_scores", {})

    if scores1 and scores2:
        common_signals = set(scores1.keys()) & set(scores2.keys())
        if common_signals:
            score_diffs = [abs(scores1[s] - scores2[s]) for s in common_signals]
            avg_diff = sum(score_diffs) / len(score_diffs)
            score_sim = max(0, 1 - avg_diff)  # 0 diff = 1.0, 1.0 diff = 0.0
            score += score_sim * 0.20  # 20% weight for signal intensity match
            weights_used += 0.20

    # Compare MNF (flow rate) - CRITICAL for accurate matching
    mnf1 = fp1.get("mnf_range")
    mnf2 = fp2.get("mnf_range")
    if mnf1 and mnf2:
        # Check if ranges overlap
        overlap = max(0, min(mnf1[1], mnf2[1]) - max(mnf1[0], mnf2[0]))
        total_range = max(mnf1[1], mnf2[1]) - min(mnf1[0], mnf2[0])
        if total_range > 0:
            mnf_sim = overlap / total_range
            score += mnf_sim * 0.20  # 20% weight for flow rate match
            weights_used += 0.20

    # Compare volume range
    vol1 = fp1.get("volume_range")
    vol2 = fp2.get("volume_range")
    if vol1 and vol2:
        overlap = max(0, min(vol1[1], vol2[1]) - max(vol1[0], vol2[0]))
        total_range = max(vol1[1], vol2[1]) - min(vol1[0], vol2[0])
        if total_range > 0:
            vol_sim = overlap / total_range
            score += vol_sim * 0.15  # 15% weight for volume match
            weights_used += 0.15

    # Compare duration range
    dur1 = fp1.get("duration_range")
    dur2 = fp2.get("duration_range")
    if dur1 and dur2:
        overlap = max(0, min(dur1[1], dur2[1]) - max(dur1[0], dur2[0]))
        total_range = max(dur1[1], dur2[1]) - min(dur1[0], dur2[0])
        if total_range > 0:
            dur_sim = overlap / total_range
            score += dur_sim * 0.15  # 15% weight for duration match
            weights_used += 0.15

    # Normalize by weights used
    if weights_used > 0:
        return score / weights_used

    return 0.0


def calculate_time_similarity(incident: Dict, pattern: Dict) -> float:
    """
    Calculate time-based similarity between an incident and a pattern.

    Returns value between 0.0 (no match) and 1.0 (perfect match).
    """
    time_fp = pattern.get("time_fingerprint", {})
    recurrence = pattern.get("recurrence_rule", {})

    if not time_fp and not recurrence:
        return 0.5  # Neutral if no time pattern defined

    score = 0.0
    weights_used = 0.0

    # Check day of week match
    incident_start = incident.get("start_day")
    pattern_days = time_fp.get("days_of_week", []) or recurrence.get("days_of_week", [])

    if incident_start and pattern_days:
        try:
            incident_dow = pd.to_datetime(incident_start).dayofweek
            if incident_dow in pattern_days:
                score += 0.5
            weights_used += 0.5
        except Exception:
            pass

    # Check time window match (if we have time data)
    time_start = time_fp.get("time_window_start")
    time_end = time_fp.get("time_window_end")

    if time_start and time_end:
        # For now, give partial credit if time window is defined
        # In future, compare actual incident time
        score += 0.25
        weights_used += 0.5

    if weights_used > 0:
        return score / weights_used

    return 0.5


def match_incident_to_patterns(
    incident: Dict,
    site_id: str = None,
) -> List[Dict]:
    """
    Match an incident against all known false alarm patterns.

    Args:
        incident: Incident data dict
        site_id: Optional site ID to filter patterns

    Returns:
        List of matching patterns with match scores, sorted by relevance
    """
    df = get_patterns_df()

    if df.empty:
        return []

    # Filter by site if provided
    if site_id:
        # Include site-specific patterns and any "global" patterns
        df = df[(df["site_id"] == site_id) | (df["site_id"] == "ALL")]

    # Only consider active patterns
    df = df[df["is_active"] == True]

    if df.empty:
        return []

    # Create fingerprint for the incident
    incident_fp = create_signal_fingerprint(incident)

    matches = []

    for _, pattern in df.iterrows():
        # Calculate signal similarity
        signal_sim = calculate_signal_similarity(
            incident_fp, pattern["signal_fingerprint"]
        )

        # Calculate time similarity
        time_sim = calculate_time_similarity(incident, pattern)

        # Combined score (weighted) - this is the RAW match quality
        combined_score = (signal_sim * 0.7) + (time_sim * 0.3)

        # Final score uses combined_score directly for matching
        # Confidence is used separately for auto-suppress decisions
        # This ensures high-quality matches are flagged even for new patterns
        final_score = combined_score

        # A match is "strong" if the combined similarity is above threshold
        # Pattern confidence affects auto-suppress, not the match flagging
        is_strong = combined_score >= SIGNAL_MATCH_THRESHOLD

        if (
            combined_score >= SIGNAL_MATCH_THRESHOLD * 0.5
        ):  # Lower threshold for returning matches
            matches.append(
                {
                    "pattern_id": pattern["pattern_id"],
                    "site_id": pattern["site_id"],
                    "category": pattern["category"],
                    "description": pattern["description"],
                    "signal_similarity": round(signal_sim, 3),
                    "time_similarity": round(time_sim, 3),
                    "combined_score": round(combined_score, 3),
                    "pattern_confidence": round(pattern["confidence"], 3),
                    "final_score": round(final_score, 3),
                    "auto_suppress": pattern["auto_suppress"],
                    "times_matched": pattern["times_matched"],
                    "is_strong_match": is_strong,
                }
            )

    # Sort by final score descending
    matches.sort(key=lambda x: x["final_score"], reverse=True)

    return matches


def check_should_suppress(incident: Dict, site_id: str) -> Tuple[bool, Optional[Dict]]:
    """
    Check if an incident should be auto-suppressed based on matching patterns.

    Args:
        incident: Incident data dict
        site_id: Site ID

    Returns:
        Tuple of (should_suppress, matching_pattern or None)
    """
    matches = match_incident_to_patterns(incident, site_id)

    for match in matches:
        if match["is_strong_match"] and match["auto_suppress"]:
            # Update pattern match count
            update_pattern_match(match["pattern_id"])
            return True, match

    return False, None


def update_pattern_match(pattern_id: str) -> None:
    """Update pattern statistics when a match is found."""
    df = get_patterns_df()

    if pattern_id in df["pattern_id"].values:
        df.loc[df["pattern_id"] == pattern_id, "times_matched"] += 1
        df.loc[df["pattern_id"] == pattern_id, "last_matched_at"] = (
            datetime.now().isoformat()
        )
        save_patterns_df(df)


def confirm_pattern_was_false(pattern_id: str) -> None:
    """User confirms that a suppressed/flagged incident was indeed a false alarm."""
    df = get_patterns_df()

    if pattern_id in df["pattern_id"].values:
        df.loc[df["pattern_id"] == pattern_id, "times_confirmed_false"] += 1
        # Increase confidence
        current_conf = df.loc[df["pattern_id"] == pattern_id, "confidence"].values[0]
        df.loc[df["pattern_id"] == pattern_id, "confidence"] = min(
            1.0, current_conf + 0.05
        )
        df.loc[df["pattern_id"] == pattern_id, "last_updated_at"] = (
            datetime.now().isoformat()
        )
        save_patterns_df(df)


def report_pattern_was_real_leak(pattern_id: str) -> None:
    """User reports that a suppressed/flagged incident was actually a real leak."""
    df = get_patterns_df()

    if pattern_id in df["pattern_id"].values:
        df.loc[df["pattern_id"] == pattern_id, "times_was_real_leak"] += 1
        # Decrease confidence significantly
        current_conf = df.loc[df["pattern_id"] == pattern_id, "confidence"].values[0]
        df.loc[df["pattern_id"] == pattern_id, "confidence"] = max(
            0.1, current_conf - 0.2
        )

        # If too many false negatives, deactivate pattern
        was_real = df.loc[df["pattern_id"] == pattern_id, "times_was_real_leak"].values[
            0
        ]
        was_false = df.loc[
            df["pattern_id"] == pattern_id, "times_confirmed_false"
        ].values[0]

        if was_real > 2 and was_real / (was_real + was_false) > 0.3:
            df.loc[df["pattern_id"] == pattern_id, "is_active"] = False
            log.warning(
                f"Deactivated pattern {pattern_id} due to high false negative rate"
            )

        df.loc[df["pattern_id"] == pattern_id, "last_updated_at"] = (
            datetime.now().isoformat()
        )
        save_patterns_df(df)


# ============================================
# PATTERN MANAGEMENT
# ============================================


def get_patterns_for_site(site_id: str) -> List[Dict]:
    """Get all patterns for a specific site."""
    df = get_patterns_df()
    site_patterns = df[(df["site_id"] == site_id) | (df["site_id"] == "ALL")]
    return site_patterns.to_dict("records")


def get_all_patterns() -> List[Dict]:
    """Get all patterns."""
    df = get_patterns_df()
    return df.to_dict("records")


def delete_pattern(pattern_id: str) -> bool:
    """Delete a pattern by ID."""
    df = get_patterns_df()

    if pattern_id in df["pattern_id"].values:
        df = df[df["pattern_id"] != pattern_id]
        save_patterns_df(df)
        log.info(f"Deleted pattern {pattern_id}")
        return True

    return False


def toggle_pattern_active(pattern_id: str) -> bool:
    """Toggle a pattern's active status."""
    df = get_patterns_df()

    if pattern_id in df["pattern_id"].values:
        current = df.loc[df["pattern_id"] == pattern_id, "is_active"].values[0]
        df.loc[df["pattern_id"] == pattern_id, "is_active"] = not current
        df.loc[df["pattern_id"] == pattern_id, "last_updated_at"] = (
            datetime.now().isoformat()
        )
        save_patterns_df(df)
        return True

    return False


def toggle_pattern_auto_suppress(pattern_id: str) -> bool:
    """Toggle a pattern's auto-suppress setting."""
    df = get_patterns_df()

    if pattern_id in df["pattern_id"].values:
        current = df.loc[df["pattern_id"] == pattern_id, "auto_suppress"].values[0]
        df.loc[df["pattern_id"] == pattern_id, "auto_suppress"] = not current
        df.loc[df["pattern_id"] == pattern_id, "last_updated_at"] = (
            datetime.now().isoformat()
        )
        save_patterns_df(df)
        return True

    return False


# ============================================
# UTILITY FUNCTIONS
# ============================================


def get_category_display_name(category: str) -> str:
    """Get human-readable category name."""
    category_names = {
        "false_alarm": "False Alarm",
        "pool_fill": "Pool Fill",
        "fire_test": "Fire System Test",
        "maintenance": "Planned Maintenance",
        "data_error": "Data Error / Sensor Issue",
        "temp_usage": "Known Temporary Usage",
        "irrigation": "Irrigation Schedule",
        "hvac": "HVAC System",
        "cleaning": "Cleaning Schedule",
        "event": "Scheduled Event",
        "other": "Other",
    }
    return category_names.get(category, category.replace("_", " ").title())


def get_pattern_summary(pattern: Dict) -> str:
    """Generate a human-readable summary of a pattern."""
    parts = []

    parts.append(
        f"Category: {get_category_display_name(pattern.get('category', 'unknown'))}"
    )

    recurrence = pattern.get("recurrence_rule", {})
    if recurrence.get("is_recurring"):
        rec_type = recurrence.get("type", "unknown")
        days = recurrence.get("days_of_week", [])
        if days:
            day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            day_str = ", ".join([day_names[d] for d in days if d < 7])
            parts.append(f"Recurs: {rec_type} ({day_str})")
        else:
            parts.append(f"Recurs: {rec_type}")

    time_fp = pattern.get("time_fingerprint", {})
    if time_fp.get("time_window_start") and time_fp.get("time_window_end"):
        parts.append(
            f"Time: {time_fp['time_window_start']} - {time_fp['time_window_end']}"
        )

    parts.append(f"Confidence: {pattern.get('confidence', 0):.0%}")

    if pattern.get("auto_suppress"):
        parts.append("🚫 Auto-suppress ON")

    return " | ".join(parts)


# ============================================
# LOG PATTERN MATCHES
# ============================================


def log_pattern_match(
    incident_id: str,
    pattern_id: str,
    match_score: float,
    action_taken: str,  # "suppressed", "flagged", "ignored"
    site_id: str,
) -> None:
    """Log when a pattern match occurs for auditing."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "incident_id": incident_id,
        "pattern_id": pattern_id,
        "match_score": match_score,
        "action_taken": action_taken,
        "site_id": site_id,
    }

    if os.path.exists(PATTERN_MATCHES_LOG):
        df = pd.read_csv(PATTERN_MATCHES_LOG)
    else:
        df = pd.DataFrame(columns=list(log_entry.keys()))

    df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
    df.to_csv(PATTERN_MATCHES_LOG, index=False)
