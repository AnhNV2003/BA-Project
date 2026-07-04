"""Model monitoring (Module 7, draft) — data & prediction drift.

Core metric: Population Stability Index (PSI) per feature + on the model's
predicted scores. PSI is transparent and dependency-free; an Evidently HTML
report is attempted as an optional extra.

Drift bands (industry-standard):
    PSI < 0.10  : stable
    0.10–0.25   : moderate drift — monitor
    > 0.25      : significant drift — RETRAINING TRIGGER

Run:  python monitoring/drift.py
"""
from __future__ import annotations

import pathlib
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import build_features            # noqa: E402
from config import MODELS, DATA_PROCESSED      # noqa: E402

REPORTS = ROOT / "monitoring" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

MONITORED = ["amount", "account_age_days", "ip_billing_distance_km",
             "num_failed_payment_attempts", "is_new_device", "hour_of_day"]
RETRAIN_PSI = 0.25


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    q = np.quantile(ref, np.linspace(0, 1, bins + 1))
    q[0], q[-1] = -np.inf, np.inf
    q = np.unique(q)
    if len(q) < 3:
        return 0.0
    r = np.clip(np.histogram(ref, q)[0] / len(ref), 1e-4, None)
    c = np.clip(np.histogram(cur, q)[0] / len(cur), 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def band(v: float) -> str:
    return "stable" if v < 0.10 else ("moderate" if v < RETRAIN_PSI else "SIGNIFICANT")


def load():
    p = DATA_PROCESSED / "transactions_context.parquet"
    if not p.exists():
        p = p.with_suffix(".csv")
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def inject_drift(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Simulate a distribution shift a monitor SHOULD catch (higher amounts,
    farther IPs, more failed attempts) — e.g. an emerging fraud campaign."""
    d = df.copy()
    d["amount"] *= rng.uniform(1.25, 1.6)
    d["ip_billing_distance_km"] *= 1.8
    d["num_failed_payment_attempts"] = d["num_failed_payment_attempts"] + rng.poisson(0.6, len(d))
    return d


def main():
    rng = np.random.default_rng(0)
    df = load()
    bundle = joblib.load(MODELS / "fraud_model.joblib")

    # temporal split: earlier vs later (natural, expected-stable baseline)
    med = df["day_index"].median()
    ref = df[df["day_index"] <= med]
    cur_natural = df[df["day_index"] > med]
    cur_drift = inject_drift(cur_natural, rng)

    def score(x):
        X = build_features(x)[bundle["features"]].astype(float)
        return bundle["model"].predict_proba(X)[:, 1]

    s_ref, s_nat, s_drift = score(ref), score(cur_natural), score(cur_drift)

    rows = []
    for f in MONITORED:
        rows.append({
            "feature": f,
            "psi_natural": round(psi(ref[f].to_numpy(), cur_natural[f].to_numpy()), 3),
            "psi_drifted": round(psi(ref[f].to_numpy(), cur_drift[f].to_numpy()), 3),
        })
    rows.append({"feature": "PREDICTION_SCORE",
                 "psi_natural": round(psi(s_ref, s_nat), 3),
                 "psi_drifted": round(psi(s_ref, s_drift), 3)})
    table = pd.DataFrame(rows)
    table["trigger_natural"] = table["psi_natural"].map(band)
    table["trigger_drifted"] = table["psi_drifted"].map(band)

    triggered = table[table["psi_drifted"] >= RETRAIN_PSI]["feature"].tolist()

    report = ["# Monitoring — Drift Report (draft)\n",
              f"Reference: day ≤ {int(med)} ({len(ref):,} rows) · "
              f"Current: day > {int(med)} ({len(cur_natural):,} rows)\n",
              f"Retraining trigger: any monitored PSI ≥ **{RETRAIN_PSI}**.\n",
              "## PSI table (natural split vs simulated-campaign split)\n",
              table.to_markdown(index=False) + "\n",
              "## Retraining decision\n",
              (f"- Natural split: **no retraining needed** (all < {RETRAIN_PSI}).\n"
               f"- Simulated fraud campaign: **RETRAIN** — drift on: "
               f"{', '.join(triggered)}.\n")]
    (REPORTS / "drift_report.md").write_text("\n".join(report), encoding="utf-8")

    print(table.to_string(index=False))
    print(f"\nRetrain trigger (simulated campaign) fired on: {triggered}")
    print(f"Report -> monitoring/reports/drift_report.md")

    # -------- optional Evidently HTML (best-effort; PSI above is the deliverable) --------
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
        rep = Report([DataDriftPreset()])
        snap = rep.run(reference_data=ref[MONITORED], current_data=cur_drift[MONITORED])
        snap.save_html(str(REPORTS / "evidently_drift.html"))
        print("Evidently HTML -> monitoring/reports/evidently_drift.html")
    except Exception as e:
        print(f"[evidently optional] skipped ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
