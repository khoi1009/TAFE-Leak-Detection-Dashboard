# config.py
# -*- coding: utf-8 -*-
"""
Configuration, constants, and initialization.
"""

import os
import sys
import io
import yaml
import logging

# -------------------------
# Global Constants
# -------------------------

APP_TITLE = "💦 Smart Leak Detection — TAFE and Schools"
CONFIG_FILE = "config_leak_detection.yml"
ACTION_LOG = "Action_Log.csv"

# -------------------------
# Logging Setup
# -------------------------

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
        logging.FileHandler("logs/dashboard.log", encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
)

log = logging.getLogger("dashboard")

# -------------------------
# Default Configuration
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
        "min_leak_score": 30.0,
        "min_volume_kl": 0.10,
        "allowed_statuses": ["INVESTIGATE", "CALL"],
    },
    "replay": {
        "analyze_hour": 6,
        "pause_on_incident_default": True,
        "max_days_per_run": 90,
    },
}

# -------------------------
# Load Configuration
# -------------------------

if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        cfg = {**DEFAULT_CFG, **yaml.safe_load(f)}
else:
    cfg = DEFAULT_CFG.copy()
    log.warning("config_leak_detection.yml not found. Using safe defaults.")

# Create necessary directories
os.makedirs(cfg.get("export_folder", "export"), exist_ok=True)
os.makedirs(cfg.get("save_dir", "plots"), exist_ok=True)
