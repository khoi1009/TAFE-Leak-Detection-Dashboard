# frontend/login_app.py
"""
TAFE Leak Detection - Login Portal with Authentication
Authenticates against FastAPI backend and redirects to dashboard.
Automatically starts the dashboard subprocess.
"""
import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import requests
import os
import subprocess
import sys
import time
import threading
import atexit
from flask import session, redirect

# Configuration
DEBUG = os.environ.get("DEBUG", "true").lower() == "true"
PORT = int(os.environ.get("LOGIN_PORT", 8050))
API_URL = os.environ.get("API_URL", "http://localhost:8000/api/v1")
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://localhost:8051")
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", 8051))

# Global variable to track dashboard process
dashboard_process = None

# Create Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css",
    ],
    suppress_callback_exceptions=True,
    title="TAFE Leak Detection - Login",
    assets_folder="assets",
)
server = app.server
server.secret_key = os.environ.get("SECRET_KEY", "tafe-leak-detection-secret-key-2025")


# Custom CSS for login page
login_css = """
<style>
    body {
        background: linear-gradient(135deg, #09090B 0%, #18181B 50%, #09090B 100%);
        min-height: 100vh;
    }
    .login-container {
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .login-card {
        background: linear-gradient(145deg, rgba(28, 28, 31, 0.95) 0%, rgba(24, 24, 27, 0.9) 100%);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 16px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(10px);
        padding: 2.5rem;
        max-width: 420px;
        width: 100%;
    }
    .login-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .login-icon {
        font-size: 4rem;
        color: #3B82F6;
        margin-bottom: 1rem;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .login-title {
        color: #E2E8F0;
        font-weight: 700;
        font-size: 1.75rem;
        margin-bottom: 0.5rem;
    }
    .login-subtitle {
        color: #71717A;
        font-size: 0.9rem;
    }
    .form-input {
        background-color: rgba(24, 24, 27, 0.8) !important;
        border: 1px solid rgba(63, 63, 70, 0.8) !important;
        color: #E2E8F0 !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    .form-input:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
    }
    .form-input::placeholder {
        color: #52525B !important;
    }
    .login-btn {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.875rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .login-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -5px rgba(59, 130, 246, 0.4) !important;
    }
    .version-badge {
        background: rgba(59, 130, 246, 0.1);
        color: #3B82F6;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .demo-credentials {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1.5rem;
    }
    .demo-credentials h6 {
        color: #22C55E;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }
    .demo-credentials code {
        background: rgba(24, 24, 27, 0.8);
        color: #E2E8F0;
        padding: 0.125rem 0.375rem;
        border-radius: 4px;
        font-size: 0.8rem;
    }
</style>
"""

# Main layout
app.layout = html.Div(
    [
        html.Div(login_css, style={"display": "none"}),
        dcc.Store(id="auth-store", storage_type="session"),
        dcc.Location(id="url", refresh=True),
        html.Div(
            [
                html.Div(
                    [
                        # Header
                        html.Div(
                            [
                                html.I(className="bi bi-droplet-fill login-icon"),
                                html.H1("TAFE Leak Detection", className="login-title"),
                                html.P(
                                    "Water Management Dashboard",
                                    className="login-subtitle",
                                ),
                                html.Span(
                                    "v2.0.0 - Complete Edition",
                                    className="version-badge",
                                ),
                            ],
                            className="login-header",
                        ),
                        # Login Form
                        html.Div(
                            [
                                # Error Alert
                                dbc.Alert(
                                    id="login-error",
                                    color="danger",
                                    is_open=False,
                                    dismissable=True,
                                    className="mb-3",
                                ),
                                # Success Alert
                                dbc.Alert(
                                    id="login-success",
                                    color="success",
                                    is_open=False,
                                    className="mb-3",
                                ),
                                # Username
                                html.Div(
                                    [
                                        html.Label(
                                            "Username",
                                            className="text-muted small mb-1",
                                        ),
                                        dbc.Input(
                                            id="login-username",
                                            placeholder="Enter your username",
                                            type="text",
                                            className="form-input mb-3",
                                        ),
                                    ]
                                ),
                                # Password
                                html.Div(
                                    [
                                        html.Label(
                                            "Password",
                                            className="text-muted small mb-1",
                                        ),
                                        dbc.Input(
                                            id="login-password",
                                            placeholder="Enter your password",
                                            type="password",
                                            className="form-input mb-4",
                                        ),
                                    ]
                                ),
                                # Login Button
                                dbc.Button(
                                    [
                                        html.I(
                                            className="bi bi-box-arrow-in-right me-2"
                                        ),
                                        "Sign In",
                                    ],
                                    id="login-button",
                                    className="login-btn w-100",
                                    size="lg",
                                    n_clicks=0,
                                ),
                                # Demo Credentials Info
                                html.Div(
                                    [
                                        html.H6(
                                            [
                                                html.I(
                                                    className="bi bi-info-circle me-1"
                                                ),
                                                "Demo Credentials",
                                            ]
                                        ),
                                        html.P(
                                            [
                                                "Admin: ",
                                                html.Code("admin"),
                                                " / ",
                                                html.Code("admin123"),
                                                html.Br(),
                                                "Operator: ",
                                                html.Code("operator"),
                                                " / ",
                                                html.Code("operator123"),
                                            ],
                                            className="mb-0 small text-muted",
                                        ),
                                    ],
                                    className="demo-credentials",
                                ),
                            ]
                        ),
                    ],
                    className="login-card",
                ),
            ],
            className="login-container",
        ),
    ],
    style={"minHeight": "100vh"},
)


@callback(
    [
        Output("login-error", "children"),
        Output("login-error", "is_open"),
        Output("login-success", "children"),
        Output("login-success", "is_open"),
        Output("auth-store", "data"),
        Output("url", "href"),
    ],
    Input("login-button", "n_clicks"),
    [
        State("login-username", "value"),
        State("login-password", "value"),
    ],
    prevent_initial_call=True,
)
def handle_login(n_clicks, username, password):
    """Handle login form submission."""
    if not n_clicks:
        return "", False, "", False, None, dash.no_update

    if not username or not password:
        return (
            "Please enter both username and password.",
            True,
            "",
            False,
            None,
            dash.no_update,
        )

    try:
        # Call FastAPI backend
        response = requests.post(
            f"{API_URL}/auth/login",
            data={"username": username, "password": password},
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()
            auth_data = {
                "access_token": data["access_token"],
                "refresh_token": data["refresh_token"],
                "user": data["user"],
            }
            # Redirect to dashboard with token
            dashboard_url = f"{DASHBOARD_URL}?token={data['access_token']}"
            return (
                "",
                False,
                f"Welcome, {data['user']['full_name'] or username}! Redirecting...",
                True,
                auth_data,
                dashboard_url,
            )

        elif response.status_code == 401:
            return (
                "Invalid username or password.",
                True,
                "",
                False,
                None,
                dash.no_update,
            )

        else:
            return (
                f"Login failed: {response.text}",
                True,
                "",
                False,
                None,
                dash.no_update,
            )

    except requests.exceptions.ConnectionError:
        # API is down - allow bypass for demo
        return (
            "⚠️ API server not running. Starting dashboard in demo mode...",
            False,
            "Redirecting to dashboard (demo mode)...",
            True,
            {"demo": True},
            f"{DASHBOARD_URL}?demo=true",
        )

    except Exception as e:
        return f"Error: {str(e)}", True, "", False, None, dash.no_update


def start_dashboard():
    """Start the dashboard app as a subprocess."""
    global dashboard_process

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_script = os.path.join(script_dir, "app.py")

    # Set environment variables for the subprocess
    env = os.environ.copy()
    env["DEMO_MODE"] = "true"
    env["DASHBOARD_PORT"] = str(DASHBOARD_PORT)

    # Start the dashboard process
    # NOTE: Do NOT capture stdout/stderr with PIPE - it causes the dashboard to hang
    # when it tries to print output during replay operations
    try:
        dashboard_process = subprocess.Popen(
            [sys.executable, dashboard_script],
            cwd=script_dir,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            ),
        )
        print(f"✅ Dashboard started (PID: {dashboard_process.pid})")

        # Give the dashboard time to start
        time.sleep(3)

        # Check if process is still running
        if dashboard_process.poll() is None:
            print(f"📊 Dashboard is running at: {DASHBOARD_URL}")
            return True
        else:
            print("❌ Dashboard failed to start")
            return False

    except Exception as e:
        print(f"❌ Failed to start dashboard: {e}")
        return False


def stop_dashboard():
    """Stop the dashboard subprocess on exit."""
    global dashboard_process
    if dashboard_process and dashboard_process.poll() is None:
        print("\n🛑 Stopping dashboard...")
        try:
            if sys.platform == "win32":
                dashboard_process.terminate()
            else:
                dashboard_process.terminate()
            dashboard_process.wait(timeout=5)
            print("✅ Dashboard stopped")
        except Exception as e:
            print(f"⚠️ Error stopping dashboard: {e}")
            dashboard_process.kill()


# Register cleanup function
atexit.register(stop_dashboard)


def check_dashboard_running():
    """Check if dashboard is accessible."""
    try:
        response = requests.get(DASHBOARD_URL, timeout=2)
        return response.status_code == 200
    except:
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("🔐 TAFE Leak Detection - Login Portal")
    print("=" * 60)

    # Check if dashboard is already running
    if check_dashboard_running():
        print(f"📊 Dashboard already running at: {DASHBOARD_URL}")
    else:
        print("🚀 Starting dashboard...")
        start_dashboard()

    print("-" * 60)
    print(f"📍 Login Portal: http://127.0.0.1:{PORT}")
    print(f"📊 Dashboard:    {DASHBOARD_URL}")
    print(f"🔗 API Backend:  {API_URL}")
    print("-" * 60)
    print("💡 Use demo credentials: admin/admin123 or operator/operator123")
    print("💡 Press Ctrl+C to stop both servers")
    print("=" * 60)

    app.run(debug=DEBUG, host="127.0.0.1", port=PORT, use_reloader=False)
