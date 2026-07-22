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
import embedded_api  # noqa: E402
import home_view  # noqa: E402
import review_view  # noqa: E402
import cost_view  # noqa: E402
import eval_view  # noqa: E402
import segment_view  # noqa: E402
import monitoring_view  # noqa: E402
import live_view  # noqa: E402
import api_tester_view  # noqa: E402

st.set_page_config(
    page_title="E-Commerce Payment Fraud Detection",
    page_icon=str(app_common.ROOT / "hust.png"),
    layout="wide",
)

# Start the FastAPI scoring service in-process (once) so the API Tester page
# works on single-port hosts like Streamlit Community Cloud. No-op if a uvicorn
# is already serving the port locally.
embedded_api.ensure_api_running()

# --- Left panel: school logo + group roster (shown on every page) ---
_LOGO = app_common.ROOT / "hust-full.png"
if _LOGO.exists():
    st.sidebar.image(str(_LOGO), use_container_width=True)

st.sidebar.markdown(
    """
#### E-Commerce Real-Time Payment Fraud Detection
*Business Analytics — Master's Capstone Project*
"""
)
st.sidebar.divider()

st.sidebar.markdown(
    """
### Group
| Member | Student ID |
| --- | --- |
| Lương Minh Dương | 20251038M |
| Trần Lê Phương Thảo | 20251186M |
| Hoàng Minh Hoàng | 20252561M |
| Ngô Việt Anh | 20252570M |
| Phạm Tuấn Kiệt | 20252751M |
| Nguyễn Như Thái | 20252270M |
"""
)
st.sidebar.divider()

nav = st.navigation([
    # Order tells the capstone story: pitch → problem → product → rigor → value →
    # real-time → operations → serving. Explicit url_path is required: the page
    # callables are all named `render`, so Streamlit would otherwise infer the
    # same pathname for each and reject it.
    st.Page(home_view.render, title="Overview", icon="🏠", url_path="overview", default=True),
    st.Page(segment_view.render, title="Segment Analytics", icon="🔎", url_path="segments"),
    st.Page(review_view.render, title="Review Queue", icon="🛡️", url_path="review"),
    st.Page(eval_view.render, title="Model Evaluation", icon="📊", url_path="evaluation"),
    st.Page(cost_view.render, title="Cost & ROI", icon="💰", url_path="cost"),
    st.Page(live_view.render, title="Live Feed", icon="📡", url_path="live"),
    st.Page(monitoring_view.render, title="Monitoring", icon="📈", url_path="monitoring"),
    st.Page(api_tester_view.render, title="API Tester", icon="🧪", url_path="api-tester"),
])
nav.run()
