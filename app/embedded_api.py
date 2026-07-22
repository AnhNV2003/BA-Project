"""Run the FastAPI scoring service in-process, for single-port deployments.

Streamlit Community Cloud (and any 'one process, one public port' host) runs
only `streamlit run …`. The API Tester page, however, scores through the live
FastAPI `/score` endpoint. Rather than drop that feature on deploy, we start
uvicorn in a background **thread** bound to 127.0.0.1:8000.

This is safe because `src/api_client.py` calls the API **server-side** (from the
Streamlit Python process), so it reaches the API over localhost *inside the
container* — the browser never needs port 8000 exposed. Locally, if you already
run `uvicorn api.main:app` yourself, we detect the open port and don't start a
second copy.
"""
from __future__ import annotations

import socket
import sys
import threading
import time

import streamlit as st

import app_common  # noqa: F401  — wires src/ onto sys.path

_HOST = "127.0.0.1"
_PORT = 8000


def _port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


@st.cache_resource(show_spinner=False)
def ensure_api_running(host: str = _HOST, port: int = _PORT) -> dict:
    """Start the embedded FastAPI server once per Streamlit process.

    Returns {"status": "external"} if something is already serving the port
    (e.g. a separately-launched uvicorn during local dev), or {"status":
    "embedded"} once the in-process server is accepting connections. Cached, so
    it runs at most once regardless of how many pages call it.
    """
    if _port_open(host, port):
        return {"status": "external", "base_url": f"http://{host}:{port}"}

    # Make `import api.main` resolve (repo root as a namespace package root).
    root = str(app_common.ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    import uvicorn
    from api.main import app as fastapi_app

    config = uvicorn.Config(fastapi_app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True, name="embedded-fastapi").start()

    # Give it a moment to bind so the first health check doesn't race the boot.
    for _ in range(50):  # up to ~5s
        if _port_open(host, port):
            break
        time.sleep(0.1)
    return {"status": "embedded", "base_url": f"http://{host}:{port}"}
