"""Live Feed page — a simulated stream of incoming transactions that grows over
time. Each tick generates fresh transactions, scores them with all 3 models, and
appends them to a running queue.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app_common import get_ensemble, model_keys
from simulate import DEFAULT_POOL_SIZE, decision_timeline, generate_pool, score_stream

_MAX_KEEP = 5000            # cap accumulated rows to bound memory
_SEED_BASE = 20240          # varied per pool regeneration for fresh data
_DECISION_COLORS = {"allow": "#e8f5e9", "review": "#fff8e1", "block": "#ffebee"}
_REVIEW_COLOR = "#F6C445"   # yellow — review
_BLOCK_COLOR = "#E5484D"    # red — block


def _ensure_state():
    ss = st.session_state
    ss.setdefault("live_stream", None)     # accumulated scored transactions
    ss.setdefault("live_pool", None)       # current fresh pool
    ss.setdefault("live_cursor", 0)        # position within the pool
    ss.setdefault("live_gen", 0)           # pool regeneration counter (seed)
    ss.setdefault("live_received", 0)      # total transactions received
    ss.setdefault("live_history", [])      # [(tick, total)] for the chart


def _reset_state():
    for k in ("live_stream", "live_pool", "live_cursor", "live_gen", "live_received", "live_history"):
        st.session_state.pop(k, None)
    _ensure_state()


def _refill_pool(k: int):
    """Ensure the pool has at least k rows left; regenerate (new seed) if not."""
    ss = st.session_state
    if ss.live_pool is None or ss.live_cursor + k > len(ss.live_pool):
        ss.live_gen += 1
        ss.live_pool = generate_pool(DEFAULT_POOL_SIZE, seed=_SEED_BASE + ss.live_gen)
        ss.live_cursor = 0


def _advance(bundle: dict, k: int):
    ss = st.session_state
    _refill_pool(k)
    rows = ss.live_pool.iloc[ss.live_cursor:ss.live_cursor + k].copy()
    ss.live_cursor += k

    scored = score_stream(rows, bundle)
    scored.insert(0, "arrival", range(ss.live_received + 1, ss.live_received + len(scored) + 1))
    ss.live_received += len(scored)

    ss.live_stream = scored if ss.live_stream is None else pd.concat([ss.live_stream, scored], ignore_index=True)
    if len(ss.live_stream) > _MAX_KEEP:
        ss.live_stream = ss.live_stream.iloc[-_MAX_KEEP:].reset_index(drop=True)

    tick = (ss.live_history[-1][0] + 1) if ss.live_history else 1
    ss.live_history.append((tick, ss.live_received))


def _color_decision(val):
    return f"background-color: {_DECISION_COLORS.get(val, '')}"


def _render(bundle: dict):
    ss = st.session_state
    keys = model_keys(bundle)
    stream = ss.live_stream

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total received", f"{ss.live_received:,}")
    if stream is not None and len(stream):
        c2.metric("In review", int((stream.agg_decision == "review").sum()))
        c3.metric("Blocked", int((stream.agg_decision == "block").sum()))
        c4.metric("Fraud caught", int(((stream.agg_decision != "allow") & (stream.isFraud == 1)).sum()))
    else:
        c2.metric("In review", 0); c3.metric("Blocked", 0); c4.metric("Fraud caught", 0)

    if ss.live_history:
        hist = pd.DataFrame(ss.live_history, columns=["tick", "received"]).set_index("tick")
        st.caption("Cumulative transactions received")
        st.line_chart(hist, height=160)

    if stream is None or not len(stream):
        st.info("Press **▶ Run** to start the incoming transaction stream.")
        return

    # Timeline of flagged transactions over time: review (yellow) + block (red),
    # stacked per arrival bucket.
    bin_size = st.slider("Timeline bucket (transactions)", 5, 100, 25, step=5, key="live_bin")
    timeline = decision_timeline(stream, bin_size)
    st.caption("Flagged transactions over time — **review** (yellow) · **block** (red)")
    st.bar_chart(
        timeline, height=220,
        color=[_REVIEW_COLOR, _BLOCK_COLOR],   # column order: ['review', 'block']
        x_label="arrival (transaction #)", y_label="flagged count",
    )

    only_flagged = st.checkbox("Only flagged (review/block)", value=False, key="live_only_flagged")
    view = stream[stream.agg_decision != "allow"] if only_flagged else stream
    view = view.sort_values("arrival", ascending=False)   # newest first

    score_cols = [f"{k}_score" for k in keys]
    cols = ["arrival", "agg_decision", *score_cols, "type", "amount",
            "num_failed_payment_attempts", "ip_billing_distance_km", "high_risk_country", "isFraud"]
    st.subheader(f"Incoming — showing {min(len(view), 200):,} of {len(stream):,} received")
    styler = (
        view[cols].head(200).style
        .format({**{c: "{:.3f}" for c in score_cols},
                 "amount": "{:,.0f}", "ip_billing_distance_km": "{:,.0f}"})
        .map(_color_decision, subset=["agg_decision"])
        .background_gradient(subset=score_cols, cmap="Reds", vmin=0, vmax=1)
    )
    st.dataframe(styler, use_container_width=True, height=460)


def render():
    st.title("📡 Live Transaction Feed")
    bundle = get_ensemble()
    _ensure_state()

    c1, c2, c3, c4 = st.columns([1, 2, 2, 1])
    running = c1.toggle("▶ Run", key="live_running")
    interval = c2.select_slider("Tick interval (s)", options=[0.5, 1.0, 2.0, 3.0, 5.0], value=2.0)
    per_tick = c3.slider("Txns / tick", 1, 20, 5)
    if c4.button("Reset"):
        _reset_state()
        st.rerun()

    st.caption(
        "Fresh synthetic transactions scored by **"
        + "**, **".join(bundle["models"][k]["model_name"] for k in model_keys(bundle))
        + "** (dest-history disabled, like real-time). Decision = max-risk aggregate."
    )

    # Auto-rerun this fragment every `interval` seconds while running; None pauses it.
    run_every = interval if running else None

    @st.fragment(run_every=run_every)
    def _stream_fragment():
        if st.session_state.get("live_running"):
            _advance(bundle, int(per_tick))
        _render(bundle)

    _stream_fragment()
