"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest
import urllib.request

BASE_URL = "http://127.0.0.1:8001"
PYTHON = sys.executable
PROJECT_ROOT = str(Path(__file__).parent.parent)


@pytest.fixture(scope="session")
def dashboard_url():
    """Start the FastAPI dashboard on port 8001 for the test session."""
    proc = subprocess.Popen(
        [PYTHON, "-m", "uvicorn", "src.app.main:app",
         "--host", "127.0.0.1", "--port", "8001"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=PROJECT_ROOT,
    )

    # Wait up to 60 s for the server to be ready (model warm-up can be slow).
    ready = False
    for _ in range(60):
        try:
            with urllib.request.urlopen(f"{BASE_URL}/", timeout=1) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            pass
        time.sleep(1)

    if not ready:
        proc.terminate()
        proc.wait()
        pytest.fail("Dashboard server did not start within 60 s")

    yield BASE_URL

    proc.terminate()
    proc.wait()
