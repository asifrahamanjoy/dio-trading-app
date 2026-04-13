"""
Dio Trading App — One-Click Launcher
======================================
Starts both the FastAPI backend and Streamlit frontend.

Usage:
    python start.py

This will:
    1. Start FastAPI on http://localhost:8000
    2. Start Streamlit on http://localhost:8501
    3. Open the browser automatically
"""

import subprocess
import sys
import time
import webbrowser
import socket
import os
from urllib import error, request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VENV_PYTHON = BASE_DIR.parent / ".venv" / "Scripts" / "python.exe"

if not VENV_PYTHON.exists():
    VENV_PYTHON = Path(sys.executable)


def _find_available_port(preferred: int) -> int:
    port = preferred
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1


def _wait_for_http_ready(url: str, timeout_seconds: int) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with request.urlopen(url, timeout=2):
                return True
        except (error.URLError, OSError):
            time.sleep(0.5)
    return False


def _start_backend(preferred_port: int) -> tuple[subprocess.Popen, int]:
    port = preferred_port
    while True:
        candidate_port = _find_available_port(port)
        api_health_url = f"http://127.0.0.1:{candidate_port}/"
        print(f"[1/2] Starting FastAPI backend on http://localhost:{candidate_port} ...")
        backend = subprocess.Popen(
            [str(VENV_PYTHON), "-m", "uvicorn", "backend.api.main:app",
             "--host", "0.0.0.0", "--port", str(candidate_port)],
            cwd=str(BASE_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if _wait_for_http_ready(api_health_url, timeout_seconds=12):
            return backend, candidate_port

        if backend.poll() is None:
            backend.terminate()
            backend.wait(timeout=5)

        print(f"  Backend on port {candidate_port} was unavailable or unresponsive. Retrying...")
        port = candidate_port + 1


def _start_frontend(preferred_port: int, api_base_url: str) -> tuple[subprocess.Popen, int]:
    port = _find_available_port(preferred_port)
    dashboard_url = f"http://127.0.0.1:{port}"
    print(f"[2/2] Starting Streamlit dashboard on http://localhost:{port} ...")

    frontend_env = os.environ.copy()
    frontend_env["DIO_API_BASE_URL"] = api_base_url
    frontend = subprocess.Popen(
        [str(VENV_PYTHON), "-m", "streamlit", "run", "frontend/app.py",
         "--server.port", str(port),
         "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        cwd=str(BASE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=frontend_env,
    )

    if not _wait_for_http_ready(dashboard_url, timeout_seconds=20):
        if frontend.poll() is None:
            frontend.terminate()
            frontend.wait(timeout=5)
        raise RuntimeError("Streamlit frontend did not become ready in time.")

    return frontend, port


def main():
    print("=" * 60)
    print("  DIO TRADING APP — STARTING")
    print("=" * 60)
    print()

    backend, backend_port = _start_backend(8000)
    api_base_url = f"http://127.0.0.1:{backend_port}"
    frontend, frontend_port = _start_frontend(8501, api_base_url)
    dashboard_url = f"http://127.0.0.1:{frontend_port}"

    # 3. Open browser
    print()
    print("=" * 60)
    print("  APP IS RUNNING!")
    print(f"  Dashboard:  {dashboard_url}")
    print(f"  API Docs:   {api_base_url}/docs")
    print("  Press Ctrl+C to stop both servers.")
    print("=" * 60)
    print()
    webbrowser.open(dashboard_url)

    try:
        backend.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        backend.terminate()
        frontend.terminate()
        backend.wait()
        frontend.wait()
        print("Stopped.")


if __name__ == "__main__":
    main()
