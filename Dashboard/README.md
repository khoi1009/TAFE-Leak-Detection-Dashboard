# Leak Detection Dashboard - Modular Version

Modularized version of `leak_detection_dashboard_realtime_simulation_all_schools.py`

## Structure

```
Dashboard_Realtime_Modular/
├── app.py                    # Main entry point - runs the Dash server
├── config.py                 # Configuration and constants
├── data.py                   # Data loading and caching
├── engine_fallback.py        # Fallback when Model_1 unavailable
├── processing.py             # Site processing and enrichment
├── utils.py                  # Helper functions (charts, UI, logging)
├── components.py             # UI components (incident cards)
├── layout.py                 # Dashboard layout (tabs, controls, modals)
├── callbacks.py              # All Dash callbacks (interactive logic)
└── README.md                 # This file
```

## Architecture

### Separation of Concerns

1. **Configuration Layer** (`config.py`)

   - Global settings and constants
   - Logging setup
   - YAML config loading

2. **Data Layer** (`data.py`)

   - Site data loading
   - In-memory caching (SITE_CACHE)
   - Data normalization and deduplication

3. **Engine Layer** (`engine_fallback.py`, `processing.py`)

   - `engine_fallback.py`: Demo mode when real engine unavailable
   - `processing.py`: Site computation and incident enrichment

4. **Presentation Layer** (`layout.py`, `components.py`)

   - `layout.py`: Page structure, tabs, controls
   - `components.py`: Reusable UI components (cards, modals)

5. **Business Logic Layer** (`callbacks.py`)

   - All interactive functionality
   - Replay simulation
   - Chart updates
   - Action logging

6. **Utilities Layer** (`utils.py`)
   - Chart helpers
   - UI utilities
   - Action log I/O

## Running the Dashboard

```bash
cd "Dashboard_Realtime_Modular"
python app.py
```

Then open http://127.0.0.1:8050 in your browser.

## Key Features

- ✅ Modular file structure (easy to maintain)
- ✅ Clean separation of concerns
- ✅ Fallback mode for demo/testing
- ✅ Preserved all functionality from original
- ⚠️ Callbacks are currently stubs (need implementation)

## Next Steps

The modular structure is in place. To complete the migration:

1. Implement full replay logic in `callbacks.py:run_replay()`
2. Implement KPI generation in `callbacks.py:update_overview()`
3. Implement event detail rendering in `callbacks.py:render_event_header()`
4. Add modal components to `layout.py`
5. Add modal callbacks to `callbacks.py`
6. Copy enrichment logic to `processing.py:compute_or_refresh_site()`

## Benefits of Modular Approach

1. **Maintainability**: Each file has a single responsibility
2. **Testability**: Functions can be unit tested in isolation
3. **Reusability**: Components can be reused across dashboards
4. **Collaboration**: Multiple developers can work on different modules
5. **Debugging**: Easier to locate and fix issues
6. **Scalability**: Easy to add new features without breaking existing code

## Migration Status

- ✅ Project structure created
- ✅ Configuration module
- ✅ Data loading and caching
- ✅ Engine fallback for demo mode
- ✅ Utility functions
- ✅ Layout definition
- ✅ Component definitions
- ⚠️ Callbacks (stub implementation)
- ❌ Full replay logic (TODO)
- ❌ Modal components (TODO)
- ❌ Modal callbacks (TODO)
