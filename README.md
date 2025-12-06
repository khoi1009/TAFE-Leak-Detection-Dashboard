# TAFE NSW Leak Detection Dashboard

**UI UX Pro Max Edition** - A professional-grade water leak detection dashboard for NSW Schools.

## 🌟 Overview

This dashboard provides real-time leak detection and monitoring for NSW school water infrastructure. It uses machine learning algorithms to detect anomalies in water consumption patterns and alerts facility managers to potential leaks.

**GitHub Repository:** https://github.com/khoi1009/TAFE-Leak-Detection-Dashboard

## 🚀 Quick Start

```bash
cd "Model for Delivery - Modular"
python app.py
```

Open http://127.0.0.1:8050 in your browser.

## 📁 Project Structure

```
Model for Delivery - Modular/
├── app.py                        # Main entry point - Dash server
├── config.py                     # Configuration and constants
├── config_leak_detection.yml     # YAML configuration
├── data.py                       # Data loading and caching
├── processing.py                 # Site processing and enrichment
├── utils.py                      # Helper functions (charts, UI)
├── components.py                 # UI components (incident cards)
├── components_map.py             # GIS Map component with Leaflet
├── layout.py                     # Dashboard layout (tabs, controls)
├── callbacks.py                  # Dash callbacks (interactive logic)
├── false_alarm_patterns.py       # Pattern matching for false alarms
├── Model_1_realtime_simulation.py # Core leak detection engine
├── engine_fallback.py            # Demo mode fallback
│
├── # Data Scripts
├── create_demo_data.py           # Generate demo data (fast loading)
├── create_school_mapping.py      # Map properties to NSW schools
│
├── # Data Files
├── demo_data.xlsx                # Demo water data (5 properties)
├── demo_school_mapping.csv       # Demo property-school mapping
├── demo_schools_gis.json         # 50 schools for GIS map
├── property_school_mapping.csv   # Full 85 property mapping
├── Action_Log.csv                # User action history
├── False_Alarm_Patterns.csv      # Learned false alarm patterns
│
├── assets/
│   ├── design-system.css         # UI UX Pro Max styling
│   └── responsive.css            # Mobile responsiveness
│
└── logs/                         # Application logs
```

## ✨ Key Features

### 🔍 Leak Detection Engine

- **Multi-signal analysis**: MNF, RESIDUAL, CUSUM, AFTERHRS, BURSTBF
- **Confidence scoring**: 0-100% leak probability with daily evolution
- **Day-by-day replay**: Simulate detection over historical data
- **Pattern matching**: Identify and suppress recurring false alarms

### 🗺️ GIS Map Integration (NEW)

- **Interactive NSW map** with 50 demo schools (or 2,216 full)
- **Toggle views**: "All Schools" vs "Leak Alerts Only"
- **Live leak markers**: See detected leaks on actual school locations
- **Property-school mapping**: Links anonymous properties to real schools
- **Color-coded status**: Normal (green), Warning (amber), Leak (orange), Critical (red)

### 📊 Dashboard Tabs

1. **Overview Tab**: KPIs, summary charts, volume analysis
2. **Events Tab**: Incident cards with drill-down details
3. **Log Tab**: Action history and audit trail
4. **GIS Map Tab**: Interactive map with leak visualization

### 🎯 Action Workflows

- **Acknowledge**: Assign incidents to team members
- **Watch**: Monitor developing situations
- **Escalate**: Urgent escalation with priority levels
- **Resolved**: Record fixes with cost tracking
- **Ignore**: Mark false alarms and record patterns

### ⚡ Performance Optimization

- **Demo mode**: 5 properties, 50 schools (~6 second load time)
- **Full mode**: 85 properties, 2,216 schools (~2 minute load time)
- **In-memory caching**: Prevents redundant calculations
- **Confidence freezing**: Stable values across simulation steps

## 🛠️ Configuration

### Switch Between Demo and Full Data

Edit `config_leak_detection.yml`:

```yaml
# Demo mode (fast - 5 properties)
data_path: "demo_data.xlsx"

# Full mode (complete - 85 properties)
data_path: "data_with_schools.xlsx"
```

### Regenerate Demo Data

```bash
python create_demo_data.py
```

### Create School Mapping (Full Dataset)

```bash
python create_school_mapping.py
```

## 🗂️ Data Pipeline

```
Original Excel Data (31.6 MB, 85 properties)
        │
        ▼
create_school_mapping.py ──► property_school_mapping.csv
        │                    data_with_schools.xlsx
        ▼
create_demo_data.py ──────► demo_data.xlsx (1.6 MB, 5 properties)
                            demo_school_mapping.csv
                            demo_schools_gis.json (50 schools)
```

## 🔧 Architecture

### Separation of Concerns

```
Config Layer ─► Data Layer ─► Processing Layer ─► Presentation Layer ─► Business Logic
(config.py)    (data.py)     (processing.py)    (layout.py)           (callbacks.py)
                                                 (components.py)
                                                 (components_map.py)
```

### Key Components

| Module                           | Purpose                       |
| -------------------------------- | ----------------------------- |
| `Model_1_realtime_simulation.py` | Core leak detection algorithm |
| `components_map.py`              | GIS map with school locations |
| `false_alarm_patterns.py`        | ML-ready pattern learning     |
| `callbacks.py`                   | All interactive functionality |

## 📈 Leak Detection Signals

| Signal       | Description                  | Weight |
| ------------ | ---------------------------- | ------ |
| **MNF**      | Minimum Night Flow anomaly   | 0-1.0  |
| **RESIDUAL** | Residual from expected usage | 0-1.0  |
| **CUSUM**    | Cumulative sum of deviations | 0-1.0  |
| **AFTERHRS** | After-hours usage pattern    | 0-1.0  |
| **BURSTBF**  | Burst before/after detection | 0-1.0  |

## 🎨 UI Design System

**Theme:** Dark mode with glass morphism effects  
**Colors:**

- Background: `#09090B` (dark)
- Cards: `#1C1C1F` with gradient
- Accent: `#3B82F6` (blue)
- Success: `#22C55E` (green)
- Warning: `#F59E0B` (amber)
- Danger: `#EF4444` (red)

## 🔮 Future Roadmap

- [ ] Real-time WebSocket updates
- [ ] Email/SMS notifications
- [ ] PDF report generation
- [ ] User authentication
- [ ] Mobile app companion
- [ ] Predictive leak forecasting
- [ ] Integration with building management systems

## 📝 Recent Updates (December 2025)

### Version 2.0 - GIS Map & Demo Data

- ✅ Interactive GIS Map with school locations
- ✅ Toggle between "All Schools" and "Leak Alerts Only"
- ✅ Property-to-school mapping (85 properties)
- ✅ Demo data optimization (6 sec vs 2 min load)
- ✅ Fixed confidence display (74% not 7400%)

### Version 1.0 - Core Dashboard

- ✅ Modular architecture from monolithic 4,506-line file
- ✅ UI UX Pro Max design system
- ✅ False alarm pattern learning
- ✅ Day-by-day replay simulation
- ✅ Action workflow system

## 👥 Contributors

- **TAFE NSW** - Project sponsor
- **Griffith University** - Research partnership
- **GitHub Copilot** - Development assistance

## 📄 License

Proprietary - TAFE NSW / Griffith University

---

**Last Updated:** December 6, 2025  
**Version:** 2.0.0
