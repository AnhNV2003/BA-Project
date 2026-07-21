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
if not (ROOT / "data" / "processed" / "sample_preview.csv").exists():
    pytest.skip("sample_preview.csv not built", allow_module_level=True)

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

_REVIEW = "import review_view; review_view.render()"
_MONITOR = "import monitoring_view; monitoring_view.render()"


def _run(body: str):
    at = AppTest.from_string(
        "import sys\n"
        f"sys.path[:0] = [{str(APP)!r}, {str(ROOT / 'src')!r}, {str(ROOT / 'monitoring')!r}]\n"
        + body
    )
    at.run(timeout=60)
    return at


def test_review_queue_renders():
    at = _run(_REVIEW)
    assert not at.exception, at.exception
    # metrics row present
    assert len(at.metric) >= 4


def test_monitoring_renders():
    at = _run(_MONITOR)
    assert not at.exception, at.exception


def test_entrypoint_navigation_runs():
    # Runs the real entrypoint through st.navigation (both pages registered),
    # which catches wiring errors like duplicate page url_paths.
    at = AppTest.from_file(str(APP / "streamlit_app.py"))
    at.run(timeout=60)
    assert not at.exception, at.exception
