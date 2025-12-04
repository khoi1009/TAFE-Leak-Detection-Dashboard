# Leak Detection Dashboard - Modular Version

Modularized version of `leak_detection_dashboard_realtime_simulation_all_schools.py`.

## Structure

```
Dashboard_Realtime_Modular/
├── app.py                    # Main entry point - runs the Dash server
├── config.py                 # Configuration and constants
├── data.py                   # Data loading and caching
├── engine_fallback.py        # Fallback engine / Demo mode
├── processing.py             # Site processing and incident enrichment
├── utils.py                  # Helper functions (charts, UI, logging)
├── components.py             # UI components (incident cards)
├── layout.py                 # Dashboard layout (tabs, controls, modals)
├── callbacks.py              # All Dash callbacks (interactive logic)
├── false_alarm_patterns.py   # Pattern matching system
└── README.md                 # This file
```

## Architecture

### Separation of Concerns

1.  **Configuration Layer** (`config.py`): Global settings, logging, and YAML config loading.
2.  **Data Layer** (`data.py`): Site data loading, in-memory caching (`SITE_CACHE`), and normalization.
3.  **Engine Layer**:
    *   `processing.py`: Main entry point for site computation. Integrates detection logic and pattern matching.
    *   `Model_1_realtime_simulation.py`: (External) Primary detection engine.
    *   `engine_fallback.py`: Demo mode used when the primary engine is unavailable.
4.  **Presentation Layer** (`layout.py`, `components.py`): Page structure, tabs, controls, and reusable UI components.
5.  **Business Logic Layer** (`callbacks.py`): Handles all interactive functionality, including the replay simulation, chart updates, and action logging.
6.  **Pattern Matching** (`false_alarm_patterns.py`): Advanced system for recording and suppressing false alarms based on signal fingerprints.

## Key Features

*   **Modular Design**: Clean separation of concerns for maintainability and scalability.
*   **Full Replay Simulation**: Day-by-day simulation of leak detection with "Pause on Incident" functionality.
*   **Pattern Matching**: Intelligent system to identify and suppress recurring false alarms (e.g., pool fills, fire tests).
*   **Robust Fallback**: Automatically switches to demo mode if the real detection engine is missing.
*   **State Preservation**: Critical fixes implemented to preserve signal components and confidence values during simulation steps.

## Running the Dashboard

```bash
cd "Dashboard_Realtime_Modular"
python app.py
```

Then open http://127.0.0.1:8050 in your browser.
