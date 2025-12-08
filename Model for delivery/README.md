# TAFE NSW Leak Detection Dashboard

## Complete Edition v2.0.0

A comprehensive water leak detection system for NSW schools, featuring real-time monitoring, GIS mapping, and intelligent pattern recognition.

---

## 🚀 Quick Start (Recommended)

### One-Command Start

Simply run the login portal which automatically starts the dashboard:

```bash
cd frontend
python login_app.py
```

Then open your browser to: **http://127.0.0.1:8050**

### Demo Credentials

- **Admin**: `admin` / `admin123`
- **Operator**: `operator` / `operator123`

---

## 📋 System Requirements

### Software Requirements

- **Python**: 3.9 or higher (tested with 3.10)
- **Operating System**: Windows 10/11 (also compatible with macOS/Linux)
- **Browser**: Chrome, Firefox, Edge (modern versions)

### Python Packages

All required packages are listed in `frontend/requirements.txt`

---

## 📦 Installation

### Step 1: Install Python

Download and install Python 3.10+ from [python.org](https://www.python.org/downloads/)

Make sure to check "Add Python to PATH" during installation.

### Step 2: Install Dependencies

Open a terminal/command prompt and navigate to the frontend folder:

```bash
cd "Model for delivery/frontend"
pip install -r requirements.txt
```

### Step 3: Run the Application

```bash
python login_app.py
```

---

## 🎯 Features

### Dashboard Features

- **Real-time Leak Detection**: Monitors water usage patterns to detect anomalies
- **Confidence Scoring**: Calculates leak probability (0-100%)
- **Pattern Matching**: Compares incidents with known leak patterns
- **Alert System**: Automatic escalation from WATCH → CALL → EMERGENCY

### GIS Map Features

- **Interactive Map**: View all 50 demo schools on an interactive map
- **Leak Alerts View**: Toggle to see only schools with active leak alerts
- **School Details**: Click on markers to see detailed information
- **Auto-zoom**: Map automatically fits to show all relevant locations

### Analysis Features

- **Time-series Analysis**: Visual charts showing water consumption patterns
- **Signal Components**: Breakdown of leak indicators (MNF, CUSUM, Residual, etc.)
- **Volume Estimation**: Estimated water loss in kiloliters
- **Historical Playback**: Simulate historical data to test detection

---

## 📁 Folder Structure

```
Model for delivery/
├── frontend/                    # Main dashboard application
│   ├── login_app.py            # Login portal (START HERE)
│   ├── app.py                  # Main dashboard
│   ├── callbacks.py            # Dash callbacks
│   ├── components.py           # UI components
│   ├── components_map.py       # GIS map component
│   ├── layout.py               # Dashboard layout
│   ├── data.py                 # Data loading
│   ├── processing.py           # Data processing
│   ├── config.py               # Configuration
│   ├── Model_1_realtime_simulation.py  # Core leak detection algorithm
│   ├── false_alarm_patterns.py # Pattern matching logic
│   ├── requirements.txt        # Python dependencies
│   ├── data/                   # Demo data files
│   │   ├── demo_data.xlsx      # Water usage data (5 properties)
│   │   ├── demo_schools_gis.json    # School locations (50 schools)
│   │   └── demo_school_mapping.csv  # Property-to-school mapping
│   └── assets/                 # CSS stylesheets
│       ├── design-system.css
│       └── responsive.css
├── backend/                    # API backend (optional)
│   └── app/                    # FastAPI application
├── start_login.bat             # Windows: Start login portal
├── start_demo.bat              # Windows: Start in demo mode
└── README.md                   # This file
```

---

## 🖥️ Usage Guide

### Starting the Application

#### Method 1: Using Python (Recommended)

```bash
cd frontend
python login_app.py
```

#### Method 2: Using Batch Files (Windows)

Double-click `start_login.bat`

### Using the Dashboard

1. **Login**: Enter credentials at http://127.0.0.1:8050
2. **Dashboard**: You'll be redirected to http://127.0.0.1:8051
3. **Analysis Tab**:
   - Select properties from the dropdown
   - Choose date range
   - Click "▶ Replay" to start simulation
4. **GIS Map Tab**:
   - View all schools or filter by leak alerts
   - Click on markers for details

### Running a Leak Detection Simulation

1. Go to the **Analysis** tab
2. Select one or more properties from the "Select Properties" dropdown
3. Set the date range (default: 2024-11-30 to 2025-04-30)
4. Click the **▶ Replay** button
5. Watch as the system processes each day and detects leaks
6. When a leak is detected, the simulation pauses showing:
   - Confidence score
   - Signal components
   - Volume lost estimate
   - Pattern matches
7. Click **Acknowledge** or **Dismiss** to continue

---

## ⚙️ Configuration

### config_leak_detection.yml

Main configuration file for the leak detection algorithm:

```yaml
# Key parameters
delta_NF_threshold: 100 # Minimum flow difference (L/h)
min_confidence_call: 50 # Minimum confidence for CALL status
min_confidence_emergency: 80 # Minimum confidence for EMERGENCY
persistence_days_call: 3 # Days before escalation to CALL
```

### Environment Variables (Optional)

```bash
DEMO_MODE=true                 # Enable demo mode (no backend required)
LOGIN_PORT=8050               # Login portal port
DASHBOARD_PORT=8051           # Dashboard port
```

---

## 🔧 Troubleshooting

### Common Issues

#### "ModuleNotFoundError"

Install missing packages:

```bash
pip install -r requirements.txt
```

#### Port Already in Use

Stop existing Python processes:

```bash
# Windows PowerShell
Get-Process -Name python | Stop-Process -Force
```

#### Dashboard Not Loading

1. Check terminal for errors
2. Ensure you're running from the `frontend` folder
3. Try running with demo mode: `set DEMO_MODE=true` then `python app.py`

### Contact Support

For technical issues, please contact the development team.

---

## 📊 Demo Data

The package includes demo data for testing:

| File                      | Description                             |
| ------------------------- | --------------------------------------- |
| `demo_data.xlsx`          | Water usage data for 5 properties       |
| `demo_schools_gis.json`   | GIS coordinates for 50 NSW schools      |
| `demo_school_mapping.csv` | Maps property IDs to school information |

### Demo Properties

1. Property 11127 - Coffs Harbour Public School
2. Property 1000 - Kingscliff Public School
3. Property 2000 - Ballina Coast High School
4. Property 3000 - Lismore High School
5. Property 4000 - Byron Bay Public School

---

## 📜 Version History

### v2.0.0 (December 2024) - Complete Edition

- ✅ Unified login portal with auto-start dashboard
- ✅ Beautiful GIS map with circle markers
- ✅ Automatic zoom to leak alerts
- ✅ Pattern matching for false alarm detection
- ✅ Demo mode for easy testing
- ✅ Responsive UI design

---

## 📝 License

This software is proprietary and confidential.
© 2024 TAFE NSW. All rights reserved.
