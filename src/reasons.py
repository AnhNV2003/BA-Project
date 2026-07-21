"""Reason codes — transparent, rule-based "why was this flagged?".

We deliberately avoid a black-box attribution (SHAP) here: for a fraud-review
analyst — and for a regulator — a short, human-readable list of the concrete
risk signals present on a transaction is more actionable and more defensible
than a bar of feature weights. Each rule maps a known risk feature to a plain
phrase; `reason_codes` returns the top signals present, most severe first.

These are explanatory signals, not the model itself — the ensemble score is
still the decision driver. Reason codes explain; they do not decide.
"""
from __future__ import annotations

import pandas as pd

# (severity, predicate(row) -> bool, phrase(row) -> str)
# Higher severity sorts first. Predicates tolerate missing columns via row.get.
_RULES = [
    (5, lambda r: _num(r.get("num_failed_payment_attempts")) >= 2,
        lambda r: f"{int(_num(r.get('num_failed_payment_attempts')))} failed payment attempts"),
    (5, lambda r: _num(r.get("high_risk_country")) == 1, lambda r: "high-risk country"),
    (4, lambda r: _num(r.get("ip_billing_distance_km")) >= 500,
        lambda r: f"IP {int(_num(r.get('ip_billing_distance_km'))):,} km from billing"),
    (4, lambda r: _num(r.get("is_new_device")) == 1, lambda r: "new device"),
    (4, lambda r: _drained(r), lambda r: "account fully drained"),
    (3, lambda r: _num(r.get("shipping_billing_mismatch")) == 1, lambda r: "shipping ≠ billing"),
    (3, lambda r: _num(r.get("is_disposable_email")) == 1, lambda r: "disposable email"),
    (3, lambda r: _num(r.get("account_age_days")) <= 30 and _num(r.get("account_age_days")) >= 0,
        lambda r: f"new account ({int(_num(r.get('account_age_days')))}d)"),
    (2, lambda r: _high_value(r),
        lambda r: f"high-value {str(r.get('type', '')).lower()}"),
    (2, lambda r: _num(r.get("is_night")) == 1 or 0 <= _num(r.get("hour_of_day")) <= 5,
        lambda r: "night-time"),
]


def _num(v) -> float:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return -1.0
        return float(v)
    except (TypeError, ValueError):
        return -1.0


def _drained(r) -> bool:
    return _num(r.get("oldbalanceOrg")) > 0 and _num(r.get("newbalanceOrig")) == 0 \
        and str(r.get("type", "")) in ("TRANSFER", "CASH_OUT")


def _high_value(r) -> bool:
    return _num(r.get("amount")) >= 10_000 and str(r.get("type", "")) in ("TRANSFER", "CASH_OUT")


def reason_codes(row, top: int = 3) -> list[str]:
    """Return up to `top` reason phrases for one transaction, most severe first."""
    hits = [(sev, phrase(row)) for sev, pred, phrase in _RULES if pred(row)]
    hits.sort(key=lambda t: t[0], reverse=True)
    return [phrase for _, phrase in hits[:top]]


def reason_series(df: pd.DataFrame, top: int = 3, sep: str = " · ") -> pd.Series:
    """Vectorized-ish helper: a Series of joined reason strings for a frame."""
    return df.apply(lambda r: sep.join(reason_codes(r, top)) or "—", axis=1)
