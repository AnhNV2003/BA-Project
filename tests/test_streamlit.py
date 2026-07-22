"""Headless smoke tests for the Streamlit pages via AppTest.

Runs each page's script server-side (no browser) and asserts it renders without
raising. Skipped if the ensemble bundle or processed data is missing.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "monitoring"))

if not (ROOT / "models" / "fraud_ensemble.joblib").exists():
    pytest.skip("fraud_ensemble.joblib not built", allow_module_level=True)

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

_MONITOR = "import monitoring_view; monitoring_view.render()"
_LIVE = "import live_view; live_view.render()"


def _run(body: str):
    at = AppTest.from_string(
        "import sys\n"
        f"sys.path[:0] = [{str(APP)!r}, {str(ROOT / 'src')!r}, {str(ROOT / 'monitoring')!r}]\n"
        + body
    )
    at.run(timeout=60)
    return at


def test_monitoring_renders():
    at = _run(_MONITOR)
    assert not at.exception, at.exception


def test_monitoring_live_dashboard_renders():
    # Seed a baseline + PSI history so the Live Monitor dashboard path runs
    # (feature chart, prediction chart, bell with triggers, snapshot table).
    body = (
        "import streamlit as st\n"
        "from simulate import generate_pool\n"
        "from app_common import get_ensemble\n"
        "b = get_ensemble()\n"
        "pool = generate_pool(n=700, seed=5)\n"
        "st.session_state['mon_baseline'] = pool.iloc[:300].copy()\n"
        "st.session_state['mon_stream'] = pool.iloc[:600].reset_index(drop=True)\n"
        "st.session_state['mon_received'] = 600\n"
        "st.session_state['mon_psi_history'] = ["
        "{'n': 300, 'amount': 0.02, 'ip_billing_distance_km': 0.28, "
        "'PREDICTION_SCORE_logreg': 0.05, 'PREDICTION_SCORE_rf': 0.01, "
        "'PREDICTION_SCORE_xgb': 0.31}]\n"
        "st.session_state['mon_perf_history'] = ["
        "{'n': 300, 'precision': 0.4, 'recall': 0.9, 'f1': 0.55, 'flagged_rate': 0.02}]\n"
        "st.session_state['mon_triggered'] = ['ip_billing_distance_km', 'PREDICTION_SCORE_xgb']\n"
        "st.session_state['mon_test_set'] = pool.iloc[:250].copy()\n"
        "st.session_state['mon_bundle'] = b\n"
        "st.session_state['mon_versions'] = ["
        "{'version': 1, 'bundle': b, 'when': 'deployed', 'scenario': '—', 'rows': None, "
        "'fraud': None, 'triggers': [], 'metrics': None},"
        "{'version': 2, 'bundle': b, 'when': '12:00:00', 'scenario': 'Sudden spike', 'rows': 15000, "
        "'fraud': 22, 'triggers': ['x'], 'metrics': None}]\n"
        "import monitoring_view; monitoring_view.render()\n"
    )
    at = _run(body)
    assert not at.exception, at.exception


def test_live_feed_renders():
    at = _run(_LIVE)
    assert not at.exception, at.exception


def test_live_feed_with_data_renders_timeline():
    # Seed a scored stream so the full render path runs (cumulative chart,
    # review/block timeline bar_chart, and the table styler).
    body = (
        "import pandas as pd, streamlit as st\n"
        "from app_common import get_ensemble\n"
        "from simulate import generate_pool, score_stream\n"
        "b = get_ensemble()\n"
        "s = score_stream(generate_pool(n=120, seed=9).iloc[:60], b)\n"
        "s.insert(0, 'arrival', range(1, len(s) + 1))\n"
        "st.session_state['live_stream'] = s\n"
        "st.session_state['live_received'] = len(s)\n"
        "st.session_state['live_history'] = [(1, len(s))]\n"
        "import live_view; live_view.render()\n"
    )
    at = _run(body)
    assert not at.exception, at.exception


def test_api_tester_renders():
    at = _run("import api_tester_view; api_tester_view.render()")
    assert not at.exception, at.exception


def test_entrypoint_navigation_runs():
    # Runs the real entrypoint through st.navigation (both pages registered),
    # which catches wiring errors like duplicate page url_paths.
    at = AppTest.from_file(str(APP / "streamlit_app.py"))
    at.run(timeout=60)
    assert not at.exception, at.exception
