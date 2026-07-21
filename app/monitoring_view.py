"""Monitoring page — drift dashboard.

Two tabs:
  📄 Reports      — the CLI-generated drift_report.md + Evidently HTML (historical).
  🔴 Live Monitor — a self-contained, scenario-driven live drift monitor. It
     generates transactions per tick (Normal / Fraud campaign / Sudden spike),
     freezes the first N as a reference baseline, and measures rolling PSI on
     each feature and each model's prediction score, with a top-right bell alert
     when retraining is needed.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import drift
from app_common import get_ensemble, model_keys
from simulate import SCENARIOS, apply_scenario, generate_pool, scenario_intensity, DEFAULT_POOL_SIZE

_BAND_COLORS = {"stable": "#e8f5e9", "moderate": "#fff8e1", "SIGNIFICANT": "#ffebee"}
_REPORT_MD = drift.REPORTS / "drift_report.md"
_REPORT_HTML = drift.REPORTS / "evidently_drift.html"

_SEED_BASE = 77000
BASELINE_N = 300     # transactions frozen as the reference baseline
WINDOW_N = 300       # rolling current window
RAMP = 300           # campaign ramp length (transactions)


# --------------------------------------------------------------------------- #
# Reports tab
# --------------------------------------------------------------------------- #
def _mtime(path):
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if path.exists() else None


def _render_reports():
    if _REPORT_MD.exists():
        st.caption(f"`drift_report.md` · last written {_mtime(_REPORT_MD)}")
        st.markdown(_REPORT_MD.read_text(encoding="utf-8"))
    else:
        st.info("No `drift_report.md` yet. Run `python monitoring/drift.py`, or use the "
                "**Live Monitor** tab.")
    st.divider()
    if _REPORT_HTML.exists():
        st.caption(f"Evidently report · last written {_mtime(_REPORT_HTML)}")
        components.html(_REPORT_HTML.read_text(encoding="utf-8"), height=800, scrolling=True)
    else:
        st.info("No `evidently_drift.html` yet — generated alongside the report by `drift.py`.")


# --------------------------------------------------------------------------- #
# Live Monitor tab
# --------------------------------------------------------------------------- #
def _ensure_mon_state():
    ss = st.session_state
    ss.setdefault("mon_stream", None)
    ss.setdefault("mon_baseline", None)
    ss.setdefault("mon_received", 0)
    ss.setdefault("mon_pool", None)
    ss.setdefault("mon_cursor", 0)
    ss.setdefault("mon_gen", 0)
    ss.setdefault("mon_psi_history", [])
    ss.setdefault("mon_triggered", [])


def _reset_mon():
    for k in ("mon_stream", "mon_baseline", "mon_received", "mon_pool",
              "mon_cursor", "mon_gen", "mon_psi_history", "mon_triggered"):
        st.session_state.pop(k, None)
    _ensure_mon_state()


def _refill_pool(k: int):
    ss = st.session_state
    if ss.mon_pool is None or ss.mon_cursor + k > len(ss.mon_pool):
        ss.mon_gen += 1
        ss.mon_pool = generate_pool(DEFAULT_POOL_SIZE, seed=_SEED_BASE + ss.mon_gen)
        ss.mon_cursor = 0


def _advance_mon(bundle: dict, k: int, scenario: str):
    ss = st.session_state
    _refill_pool(k)
    rows = ss.mon_pool.iloc[ss.mon_cursor:ss.mon_cursor + k].copy()
    ss.mon_cursor += k

    intensity = scenario_intensity(scenario, ss.mon_received, BASELINE_N, RAMP)
    rng = np.random.default_rng(_SEED_BASE + ss.mon_received)
    rows = apply_scenario(rows, scenario, intensity, rng)

    ss.mon_received += len(rows)
    ss.mon_stream = rows if ss.mon_stream is None else pd.concat([ss.mon_stream, rows], ignore_index=True)
    if len(ss.mon_stream) > WINDOW_N * 3:
        ss.mon_stream = ss.mon_stream.iloc[-WINDOW_N * 3:].reset_index(drop=True)

    if ss.mon_baseline is None and ss.mon_received >= BASELINE_N:
        ss.mon_baseline = ss.mon_stream.iloc[:BASELINE_N].copy()

    if ss.mon_baseline is not None:
        current = ss.mon_stream.iloc[-WINDOW_N:]
        psi = drift.psi_map(ss.mon_baseline, current, bundle)
        ss.mon_psi_history.append({"n": ss.mon_received, **{k2: round(v, 3) for k2, v in psi.items()}})
        if len(ss.mon_psi_history) > 400:
            ss.mon_psi_history = ss.mon_psi_history[-400:]

        now_triggered = {name for name, v in psi.items() if v >= drift.RETRAIN_PSI}
        new = now_triggered - set(ss.mon_triggered)
        if new:
            st.toast("⚠️ Retrain needed — drift on " + ", ".join(sorted(new)), icon="🚨")
        ss.mon_triggered = sorted(now_triggered)


def _render_bell(names: list[str]):
    count = len(names)
    if count:
        tip = "Retrain needed — drift on: " + ", ".join(names)
        html = f"""<div title="{tip}" style="position:fixed; top:3.4rem; right:1.2rem;
            z-index:1000; font-size:1.7rem; line-height:1;">🔔<span style="position:absolute;
            top:-6px; right:-10px; background:#E5484D; color:#fff; font-size:.7rem;
            font-weight:700; padding:1px 6px; border-radius:10px;">{count}</span></div>"""
    else:
        html = """<div title="No drift alerts" style="position:fixed; top:3.4rem; right:1.2rem;
            z-index:1000; font-size:1.7rem; line-height:1; opacity:.4;">🔔</div>"""
    st.markdown(html, unsafe_allow_html=True)


def _render_dashboard(bundle: dict):
    ss = st.session_state
    triggered = list(ss.mon_triggered)
    _render_bell(triggered)

    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions", f"{ss.mon_received:,}")
    c2.metric("Baseline", f"{BASELINE_N} (frozen)" if ss.mon_baseline is not None
              else f"building {min(ss.mon_received, BASELINE_N)}/{BASELINE_N}")
    c3.metric("Signals in drift", len(triggered),
              delta="RETRAIN" if triggered else None, delta_color="inverse")

    if not ss.mon_psi_history:
        st.info(f"Press **▶ Run**. Drift starts once the baseline ({BASELINE_N} transactions) "
                "is captured; pick **Fraud campaign** or **Sudden spike** to watch it trip.")
        return

    hist = pd.DataFrame(ss.mon_psi_history).set_index("n")
    hist["threshold"] = drift.RETRAIN_PSI
    feats = [f for f in drift.MONITORED if f in hist.columns]
    pred_cols = [c for c in hist.columns if c.startswith("PREDICTION_SCORE_")]
    hist["combined"] = hist[pred_cols].mean(axis=1)

    st.markdown("**Feature drift — PSI over transactions received**")
    st.line_chart(hist[feats + ["threshold"]], height=240)
    st.markdown("**Prediction-score drift — per model + combined**")
    st.line_chart(hist[pred_cols + ["combined", "threshold"]], height=240)

    latest = ss.mon_psi_history[-1]
    snap = pd.DataFrame([{"signal": k, "psi": v, "trigger": drift.band(v)}
                         for k, v in latest.items() if k != "n"])
    styler = snap.style.format({"psi": "{:.3f}"}).map(
        lambda v: f"background-color: {_BAND_COLORS.get(v, '')}", subset=["trigger"])
    st.dataframe(styler, use_container_width=True, hide_index=True)
    st.caption(f"Reference: first {BASELINE_N} txns (frozen) · Current: last {WINDOW_N} txns. "
               f"Retrain trigger at PSI ≥ {drift.RETRAIN_PSI} (dashed threshold line).")


def _render_live_monitor():
    bundle = get_ensemble()
    _ensure_mon_state()

    c1, c2, c3, c4, c5 = st.columns([1, 2, 2, 2, 1])
    running = c1.toggle("▶ Run", key="mon_running")
    c2.selectbox("Scenario", SCENARIOS, key="mon_scenario")
    interval = c3.select_slider("Interval (s)", options=[0.5, 1.0, 2.0, 3.0], value=1.0, key="mon_interval")
    c4.slider("Txns / tick", 10, 60, 30, step=10, key="mon_per_tick")
    if c5.button("Reset", key="mon_reset"):
        _reset_mon()
        st.rerun()

    run_every = interval if running else None

    @st.fragment(run_every=run_every)
    def _tick():
        if st.session_state.get("mon_running"):
            _advance_mon(bundle, int(st.session_state.mon_per_tick), st.session_state.mon_scenario)
        _render_dashboard(bundle)

    _tick()


def render():
    st.title("📈 Monitoring — Drift")
    tab_reports, tab_live = st.tabs(["📄 Reports", "🔴 Live Monitor"])
    with tab_reports:
        _render_reports()
    with tab_live:
        _render_live_monitor()
