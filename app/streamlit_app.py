"""Fraud-detection application (Module 6) — multipage Streamlit.

Pages:
  🛡️ Review Queue — all 3 models score each transaction; triage by max-risk.
  📈 Monitoring   — drift dashboard (Reports + Live tabs), per-model prediction drift.

Run:
    streamlit run app/streamlit_app.py
"""
import streamlit as st

# app_common wires sys.path (src/, monitoring/) — import it BEFORE the views so
# their module-level `from ensemble/config/drift import ...` resolve.
import app_common  # noqa: F401,E402
import review_view  # noqa: E402
import monitoring_view  # noqa: E402
import live_view  # noqa: E402

st.set_page_config(page_title="Fraud Detection", layout="wide")

nav = st.navigation([
    # Explicit url_path is required: the page callables are all named `render`, so
    # Streamlit would otherwise infer the same pathname for each and reject it.
    st.Page(review_view.render, title="Review Queue", icon="🛡️", url_path="review", default=True),
    st.Page(live_view.render, title="Live Feed", icon="📡", url_path="live"),
    st.Page(monitoring_view.render, title="Monitoring", icon="📈", url_path="monitoring"),
])
nav.run()
