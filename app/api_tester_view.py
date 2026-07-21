"""API Tester page — build a transaction (from a preset or by hand) and score it
through the live FastAPI /score endpoint, showing each model's result + the
max-risk aggregate."""
from __future__ import annotations

import streamlit as st

from api_client import DEFAULT_BASE, health, score

# Editable transaction fields — (key, label, kind). Matches api.main.Transaction.
_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
_FLOATS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
           "ip_billing_distance_km"]
_INTS = ["account_age_days", "num_failed_payment_attempts", "hour_of_day",
         "txn_count_last_24h", "time_since_last_hours", "account_txn_total"]
_FLAGS = ["is_new_device", "shipping_billing_mismatch", "is_disposable_email",
          "high_risk_country", "is_night"]
_FIELDS = ["type"] + _FLOATS + _INTS + _FLAGS

# Preset transactions — realistic starting points the user can then edit.
PRESETS: dict[str, dict] = {
    "🟢 Legit payment (expect allow)": {
        "type": "PAYMENT", "amount": 42.5, "oldbalanceOrg": 5000.0, "newbalanceOrig": 4957.5,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "account_age_days": 900,
        "is_new_device": 0, "shipping_billing_mismatch": 0, "num_failed_payment_attempts": 0,
        "ip_billing_distance_km": 5.0, "is_disposable_email": 0, "high_risk_country": 0,
        "hour_of_day": 14, "is_night": 0, "txn_count_last_24h": 3,
        "time_since_last_hours": 6, "account_txn_total": 300,
    },
    "🔴 Fraudulent transfer — drained (expect block)": {
        "type": "TRANSFER", "amount": 21279.19, "oldbalanceOrg": 21279.19, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "account_age_days": 52,
        "is_new_device": 1, "shipping_billing_mismatch": 1, "num_failed_payment_attempts": 3,
        "ip_billing_distance_km": 1299.7, "is_disposable_email": 0, "high_risk_country": 1,
        "hour_of_day": 3, "is_night": 1, "txn_count_last_24h": 0,
        "time_since_last_hours": -1, "account_txn_total": 1,
    },
    "🟠 Suspicious cash-out (borderline)": {
        "type": "CASH_OUT", "amount": 6800.0, "oldbalanceOrg": 6800.0, "newbalanceOrig": 0.0,
        "oldbalanceDest": 1200.0, "newbalanceDest": 8000.0, "account_age_days": 140,
        "is_new_device": 1, "shipping_billing_mismatch": 0, "num_failed_payment_attempts": 1,
        "ip_billing_distance_km": 420.0, "is_disposable_email": 0, "high_risk_country": 0,
        "hour_of_day": 23, "is_night": 0, "txn_count_last_24h": 2,
        "time_since_last_hours": 1, "account_txn_total": 12,
    },
    "🆕 New-device high-value (risky)": {
        "type": "TRANSFER", "amount": 48000.0, "oldbalanceOrg": 60000.0, "newbalanceOrig": 12000.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 48000.0, "account_age_days": 20,
        "is_new_device": 1, "shipping_billing_mismatch": 1, "num_failed_payment_attempts": 2,
        "ip_billing_distance_km": 3200.0, "is_disposable_email": 1, "high_risk_country": 1,
        "hour_of_day": 4, "is_night": 1, "txn_count_last_24h": 5,
        "time_since_last_hours": 0, "account_txn_total": 2,
    },
}
_DEFAULT_PRESET = "🔴 Fraudulent transfer — drained (expect block)"
_DECISION_STYLE = {"allow": ("✅", "success"), "review": ("🟡", "warning"), "block": ("🛑", "error")}


def _load_preset(name: str):
    for k, v in PRESETS[name].items():
        st.session_state[f"at_{k}"] = v


def _ensure_defaults():
    if "at_type" not in st.session_state:
        _load_preset(_DEFAULT_PRESET)
    st.session_state.setdefault("at_base_url", DEFAULT_BASE)


def _collect_txn() -> dict:
    txn = {"type": st.session_state["at_type"]}
    for f in _FLOATS:
        txn[f] = float(st.session_state[f"at_{f}"])
    for f in _INTS + _FLAGS:
        txn[f] = int(st.session_state[f"at_{f}"])
    return txn


def _render_result(data: dict):
    agg = data.get("aggregate", {})
    icon, kind = _DECISION_STYLE.get(agg.get("decision", ""), ("❓", "info"))
    getattr(st, kind)(f"{icon} Aggregate decision: **{str(agg.get('decision', '?')).upper()}**  "
                      f"· rule: {agg.get('rule', '—')}"
                      + ("  · ⚠️ degraded (a model errored)" if agg.get("degraded") else ""))
    rows = []
    for key, m in data.get("models", {}).items():
        rows.append({
            "model": m.get("model_name", key),
            "fraud_probability": m.get("fraud_probability"),
            "decision": m.get("decision", m.get("error", "?")),
            "review_thr": m.get("review_threshold"),
            "block_thr": m.get("block_threshold"),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    with st.expander("Raw API response (JSON)"):
        st.json(data)


def render():
    st.title("🧪 API Tester")
    st.caption("Build a transaction — from a preset or by hand — and score it through the "
               "live **POST /score** endpoint. Each model scores independently; the verdict "
               "is the max-risk aggregate.")
    _ensure_defaults()

    top = st.columns([3, 1], vertical_alignment="bottom")
    top[0].text_input("API base URL", key="at_base_url")
    if top[1].button("Test connection", use_container_width=True):
        ok, info = health(st.session_state["at_base_url"])
        if ok:
            st.success(f"Connected · models: {', '.join(info.get('models', {}).values())} "
                       f"· rule: {info.get('aggregate_rule', '—')}")
        else:
            st.error(f"Cannot reach API: {info}\n\nStart it with: "
                     "`uvicorn api.main:app --host 127.0.0.1 --port 8000`")

    pcols = st.columns([3, 1], vertical_alignment="bottom")
    preset = pcols[0].selectbox("Preset", list(PRESETS), index=list(PRESETS).index(_DEFAULT_PRESET))
    if pcols[1].button("Load preset", use_container_width=True):
        _load_preset(preset)
        st.rerun()

    st.markdown("**Transaction**")
    c = st.columns(4)
    c[0].selectbox("type", _TYPES, key="at_type")
    grid = [(f, "float") for f in _FLOATS] + [(f, "int") for f in _INTS]
    for i, (f, kind) in enumerate(grid):
        col = c[(i + 1) % 4]
        if kind == "float":
            col.number_input(f, key=f"at_{f}", step=100.0, format="%.2f")
        else:
            col.number_input(f, key=f"at_{f}", step=1)
    fc = st.columns(len(_FLAGS))
    for col, f in zip(fc, _FLAGS):
        col.selectbox(f, [0, 1], key=f"at_{f}", format_func=lambda v: "Yes" if v else "No")

    if st.button("🚀 Score via API", type="primary"):
        ok, data = score(st.session_state["at_base_url"], _collect_txn())
        st.session_state["at_last"] = (ok, data)

    if "at_last" in st.session_state:
        ok, data = st.session_state["at_last"]
        st.divider()
        if ok:
            _render_result(data)
        else:
            st.error(f"Scoring failed: {data}\n\nIs the API running? "
                     "`uvicorn api.main:app --host 127.0.0.1 --port 8000`")
