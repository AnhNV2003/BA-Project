"""Monitoring page — drift dashboard with a Reports tab and a Live tab."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

import drift
from app_common import REPORTS, get_context_data, get_ensemble

_BAND_COLORS = {"stable": "#e8f5e9", "moderate": "#fff8e1", "SIGNIFICANT": "#ffebee"}
_REPORT_MD = REPORTS / "drift_report.md"
_REPORT_HTML = REPORTS / "evidently_drift.html"


def _mtime(path):
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if path.exists() else None


def _render_reports():
    if _REPORT_MD.exists():
        st.caption(f"`drift_report.md` · last written {_mtime(_REPORT_MD)}")
        st.markdown(_REPORT_MD.read_text(encoding="utf-8"))
    else:
        st.info("No `drift_report.md` yet. Run `python monitoring/drift.py` to generate it, "
                "or use the **Live** tab to recompute now.")

    st.divider()
    if _REPORT_HTML.exists():
        st.caption(f"Evidently report · last written {_mtime(_REPORT_HTML)}")
        components.html(_REPORT_HTML.read_text(encoding="utf-8"), height=800, scrolling=True)
    else:
        st.info("No `evidently_drift.html` yet — generated alongside the report by `drift.py`.")


def _color_band(val):
    return f"background-color: {_BAND_COLORS.get(val, '')}"


def _render_live():
    bundle = get_ensemble()
    df = get_context_data()
    ref, cur_natural, med = drift.split_reference_current(df)

    simulate = st.toggle("Simulate fraud campaign (inject drift)", value=False,
                         help="Applies higher amounts, farther IPs and more failed attempts to the current window.")
    cur = drift.inject_drift(cur_natural, np.random.default_rng(0)) if simulate else cur_natural

    table = drift.compute_drift(ref, cur, bundle)   # feature | psi | trigger
    triggered = table[table["psi"] >= drift.RETRAIN_PSI]["feature"].tolist()

    if triggered:
        st.error(f"🚨 **RETRAIN** — significant drift (PSI ≥ {drift.RETRAIN_PSI}) on: "
                 + ", ".join(triggered))
    else:
        st.success(f"✅ No retraining needed — all monitored PSI < {drift.RETRAIN_PSI}.")

    styler = table.style.format({"psi": "{:.3f}"}).map(_color_band, subset=["trigger"])
    st.dataframe(styler, use_container_width=True, hide_index=True)
    st.caption(
        f"Reference: day ≤ {int(med)} ({len(ref):,} rows) · "
        f"Current{' + campaign' if simulate else ''}: day > {int(med)} ({len(cur):,} rows). "
        "`PREDICTION_SCORE_<model>` rows show per-model output drift."
    )


def render():
    st.title("📈 Monitoring — Drift")
    tab_reports, tab_live = st.tabs(["📄 Reports", "🔴 Live"])
    with tab_reports:
        _render_reports()
    with tab_live:
        _render_live()
