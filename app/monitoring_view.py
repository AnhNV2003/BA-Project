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

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import drift
from app_common import get_ensemble, model_keys
from simulate import SCENARIOS, apply_scenario, generate_pool, score_stream, scenario_intensity, DEFAULT_POOL_SIZE
from ensemble import window_performance
from retrain import retrain_ensemble
from alerting import build_alert_payload, incident_report_md, send_webhook

_RETRAIN_SAMPLE_N = 15000   # labelled rows generated for an in-app retrain
_PERF_COLORS = ["#2E7D32", "#1565C0", "#8E24AA", "#EF6C00"]  # precision/recall/f1/flagged

_BAND_COLORS = {"stable": "#e8f5e9", "moderate": "#fff8e1", "SIGNIFICANT": "#ffebee"}
_REPORT_MD = drift.REPORTS / "drift_report.md"
_REPORT_HTML = drift.REPORTS / "evidently_drift.html"

_REF_COLOR = "#6C8EBF"   # blue  — reference baseline
_CUR_COLOR = "#F6A445"   # orange — current window

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
    ss.setdefault("mon_perf_history", [])
    ss.setdefault("mon_triggered", [])
    ss.setdefault("mon_bundle", None)     # in-session retrained bundle override


def _reset_mon():
    for k in ("mon_stream", "mon_baseline", "mon_received", "mon_pool", "mon_cursor",
              "mon_gen", "mon_psi_history", "mon_perf_history", "mon_triggered", "mon_bundle"):
        st.session_state.pop(k, None)
    _ensure_mon_state()


def _active_bundle() -> dict:
    """The in-session retrained bundle if present, else the deployed ensemble."""
    return st.session_state.get("mon_bundle") or get_ensemble()


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

        # live performance of the aggregate decision on the (labelled) window
        perf = window_performance(score_stream(current, bundle))
        if perf:
            ss.mon_perf_history.append({"n": ss.mon_received,
                                        **{m: perf[m] for m in ("precision", "recall", "f1", "flagged_rate")}})
            if len(ss.mon_perf_history) > 400:
                ss.mon_perf_history = ss.mon_perf_history[-400:]

        now_triggered = {name for name, v in psi.items() if v >= drift.RETRAIN_PSI}
        new = now_triggered - set(ss.mon_triggered)
        if new:
            st.toast("⚠️ Retrain needed — drift on " + ", ".join(sorted(new)), icon="🚨")
            _maybe_send_alert(sorted(new), psi)
        ss.mon_triggered = sorted(now_triggered)


def _maybe_send_alert(new_signals, psi):
    ss = st.session_state
    url = ss.get("mon_webhook_url", "")
    if ss.get("mon_auto_send") and url:
        payload = build_alert_payload(new_signals, psi, received=ss.mon_received,
                                      when=datetime.now().isoformat(timespec="seconds"))
        ok, msg = send_webhook(url, payload)
        st.toast(("📤 Alert sent" if ok else f"⚠️ Alert failed — {msg}"), icon="📤" if ok else "⚠️")


def _do_retrain(scenario: str):
    """Refit the models on freshly-generated current-distribution data, adopt the
    current window as the new baseline, and reset the charts so drift + performance
    recover under the (now-adapted) model."""
    ss = st.session_state
    base = _active_bundle()
    intensity = scenario_intensity(scenario, ss.mon_received, BASELINE_N, RAMP)
    with st.spinner(f"Retraining 3 models on {_RETRAIN_SAMPLE_N:,} current-distribution transactions…"):
        sample = generate_pool(_RETRAIN_SAMPLE_N, seed=_SEED_BASE + 999)
        sample = apply_scenario(sample, scenario, intensity, np.random.default_rng(_SEED_BASE + 999))
        ss.mon_bundle = retrain_ensemble(base, sample, seed=7)
    # adopt the current (drifted) window as the new normal so ongoing traffic matches
    if ss.mon_stream is not None and len(ss.mon_stream):
        ss.mon_baseline = ss.mon_stream.iloc[-WINDOW_N:].copy()
    ss.mon_psi_history, ss.mon_perf_history, ss.mon_triggered = [], [], []
    st.toast(f"✅ Retrained on {ss.mon_bundle['retrain_rows']:,} rows "
             f"({ss.mon_bundle['retrain_fraud']} fraud) — baseline reset.", icon="✅")


def _render_bell(names: list[str], latest_psi: dict | None = None):
    """Clickable alert bell: a popover showing the drifting signals. The label
    carries a count badge so the alert state is visible without opening it."""
    count = len(names)
    label = f"🔔 {count}" if count else "🔔"
    with st.popover(label, use_container_width=True, help="Drift / retrain alerts"):
        if count:
            st.markdown(f"**⚠️ Retrain recommended** — {count} signal(s) at PSI ≥ {drift.RETRAIN_PSI}:")
            for n in names:
                v = (latest_psi or {}).get(n)
                st.markdown(f"- `{n}`" + (f" — PSI **{v:.3f}**" if isinstance(v, (int, float)) else ""))
        else:
            st.success("No drift alerts — all signals below threshold.")


def _render_dashboard(bundle: dict):
    ss = st.session_state
    triggered = list(ss.mon_triggered)

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

    if ss.mon_perf_history:
        st.markdown("**Model performance on the live stream** (aggregate decision vs. ground truth)")
        perf = pd.DataFrame(ss.mon_perf_history).set_index("n")
        st.line_chart(perf, height=220, color=_PERF_COLORS[:perf.shape[1]])
        st.caption("Precision falls / flagged-rate rises as drift pushes legitimate traffic over the "
                   "threshold. Fraud is rare per window, so recall is noisier.")

    latest = ss.mon_psi_history[-1]
    snap = pd.DataFrame([{"signal": k, "psi": v, "trigger": drift.band(v)}
                         for k, v in latest.items() if k != "n"])
    styler = snap.style.format({"psi": "{:.3f}"}).map(
        lambda v: f"background-color: {_BAND_COLORS.get(v, '')}", subset=["trigger"])
    st.dataframe(styler, use_container_width=True, hide_index=True)
    st.caption(f"Reference: first {BASELINE_N} txns (frozen) · Current: last {WINDOW_N} txns. "
               f"Retrain trigger at PSI ≥ {drift.RETRAIN_PSI} (dashed threshold line).")

    # Reference vs current distribution per feature — explains each PSI value.
    st.markdown("**Reference vs current distribution per feature** "
                "(blue = reference baseline · orange = current window)")
    current = ss.mon_stream.iloc[-WINDOW_N:]
    feats = [f for f in drift.MONITORED if f in current.columns]
    per_row = 3
    for i in range(0, len(feats), per_row):
        row = feats[i:i + per_row]
        for col, feat in zip(st.columns(len(row)), row):
            with col:
                st.caption(f"{feat} · PSI {latest.get(feat, float('nan')):.3f}")
                dist = drift.distribution_frame(ss.mon_baseline[feat], current[feat])
                st.altair_chart(_distribution_chart(dist), use_container_width=True)


def _distribution_chart(dist: pd.DataFrame) -> "alt.Chart":
    """Grouped bar chart of reference vs current, bins kept in numeric order with
    rotated readable labels and a percentage y-axis."""
    long = dist.reset_index(names="bin").melt("bin", var_name="set", value_name="frac")
    long["order"] = long.groupby("set").cumcount()
    return (
        alt.Chart(long, height=210)
        .mark_bar()
        .encode(
            x=alt.X("bin:N", title=None,
                    sort=alt.EncodingSortField(field="order", op="min", order="ascending"),
                    axis=alt.Axis(labelAngle=-40, labelLimit=90)),
            xOffset=alt.XOffset("set:N", sort=["reference", "current"]),
            y=alt.Y("frac:Q", title="share", axis=alt.Axis(format="%")),
            color=alt.Color("set:N", title=None,
                            scale=alt.Scale(domain=["reference", "current"],
                                            range=[_REF_COLOR, _CUR_COLOR])),
            tooltip=[alt.Tooltip("bin:N", title="range"),
                     alt.Tooltip("set:N", title="window"),
                     alt.Tooltip("frac:Q", title="share", format=".1%")],
        )
    )


def _render_alerting_panel():
    ss = st.session_state
    with st.expander("🔔 Alerting (webhook / report)", expanded=False):
        st.text_input("Webhook URL (Slack / Discord / generic incoming webhook)",
                      key="mon_webhook_url", placeholder="https://hooks.slack.com/services/…")
        st.checkbox("Auto-send when a new signal trips", key="mon_auto_send", value=False,
                    help="Off by default. Nothing is sent unless a URL is set and this is on.")
        a, b = st.columns(2)
        if a.button("Send test alert"):
            payload = build_alert_payload(["ip_billing_distance_km"], {"ip_billing_distance_km": 0.42},
                                          received=ss.mon_received,
                                          when=datetime.now().isoformat(timespec="seconds"))
            ok, msg = send_webhook(ss.get("mon_webhook_url", ""), payload)
            (st.success if ok else st.error)(msg)
        report = incident_report_md(list(ss.mon_triggered),
                                    ss.mon_psi_history[-1] if ss.mon_psi_history else {},
                                    received=ss.mon_received,
                                    when=datetime.now().isoformat(timespec="seconds"))
        b.download_button("Download incident report", report,
                          file_name="drift_incident.md", mime="text/markdown")


def _render_live_monitor():
    _ensure_mon_state()
    bundle = _active_bundle()

    cols = st.columns([1, 2, 1.6, 1.6, 1, 1.2], vertical_alignment="bottom")
    running = cols[0].toggle("▶ Run", key="mon_running")
    cols[1].selectbox("Scenario", SCENARIOS, key="mon_scenario")
    interval = cols[2].select_slider("Interval (s)", options=[0.5, 1.0, 2.0, 3.0], value=1.0, key="mon_interval")
    cols[3].slider("Txns / tick", 10, 60, 30, step=10, key="mon_per_tick")
    if cols[4].button("Reset", key="mon_reset"):
        _reset_mon()
        st.rerun()
    with cols[5]:
        # Clickable alert bell, outside the auto-refresh fragment so it isn't torn
        # down each tick. Live pulses arrive via st.toast + the metric below.
        latest = st.session_state.mon_psi_history[-1] if st.session_state.mon_psi_history else {}
        _render_bell(list(st.session_state.mon_triggered), latest)

    # Retrain + alerting live outside the fragment so clicks/inputs are stable.
    act1, act2 = st.columns([1, 3], vertical_alignment="bottom")
    triggered = bool(st.session_state.mon_triggered)
    if act1.button("🔄 Retrain now", type="primary" if triggered else "secondary",
                   help="Refit the 3 models on fresh current-distribution data and redeploy in-session."):
        _do_retrain(st.session_state.mon_scenario)
        st.rerun()
    with act2:
        if st.session_state.get("mon_bundle"):
            st.caption("🟢 Serving an **in-session retrained** model. Reset to restore the deployed bundle.")
        _render_alerting_panel()

    run_every = interval if running else None

    @st.fragment(run_every=run_every)
    def _tick():
        if st.session_state.get("mon_running"):
            _advance_mon(_active_bundle(), int(st.session_state.mon_per_tick), st.session_state.mon_scenario)
        _render_dashboard(_active_bundle())

    _tick()


def render():
    st.title("📈 Monitoring — Drift")
    tab_reports, tab_live = st.tabs(["📄 Reports", "🔴 Live Monitor"])
    with tab_reports:
        _render_reports()
    with tab_live:
        _render_live_monitor()
