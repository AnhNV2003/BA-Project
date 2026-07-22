"""Tiny HTTP client for the scoring API (used by the Streamlit API Tester page).

Stdlib only (urllib). Every call returns (ok, result_or_error) and never raises,
so the UI can render errors instead of crashing when the API is down.
"""
from __future__ import annotations

import json
import os
import urllib.request

DEFAULT_BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")


def _valid(base: str) -> bool:
    return bool(base) and str(base).lower().startswith(("http://", "https://"))


def _request(url: str, payload: dict | None, timeout: float):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    method = "POST" if payload is not None else "GET"
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def health(base: str = DEFAULT_BASE, timeout: float = 3.0):
    if not _valid(base):
        return False, "Invalid API base URL (must start with http:// or https://)."
    try:
        return True, _request(base.rstrip("/") + "/", None, timeout)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def score(base: str, txn: dict, timeout: float = 5.0):
    if not _valid(base):
        return False, "Invalid API base URL (must start with http:// or https://)."
    try:
        return True, _request(base.rstrip("/") + "/score", txn, timeout)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
