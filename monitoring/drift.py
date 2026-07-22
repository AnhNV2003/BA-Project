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

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from infer import enrich                       # noqa: E402
from ensemble import load_ensemble, score_all  # noqa: E402
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


def _fmt_num(x: float) -> str:
    """Compact human-readable number: 1300 -> '1.3k', 165000 -> '165k', 480 -> '480'."""
    ax = abs(float(x))
    if ax >= 1e6:
        v, u = x / 1e6, "M"
    elif ax >= 1e3:
        v, u = x / 1e3, "k"
    else:
        v, u = x, ""
    return f"{v:.1f}".rstrip("0").rstrip(".") + u


def distribution_frame(ref, cur, bins: int = 10) -> pd.DataFrame:
    """Normalized reference vs current distribution for one feature, for the
    overlaid histogram that explains a PSI value.

    Discrete/low-cardinality features (e.g. is_new_device, hour_of_day) use one
    bucket per value; continuous features use reference-quantile bins (the same
    binning idea PSI uses). Columns: ['reference', 'current'] as fractions.
    """
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    uniq = np.unique(ref)

    if uniq.size <= min(bins, 12):
        vals = np.unique(np.concatenate([uniq, np.unique(cur)])) if cur.size else uniq
        r = np.array([(ref == v).mean() if ref.size else 0.0 for v in vals])
        c = np.array([(cur == v).mean() if cur.size else 0.0 for v in vals])
        idx = [str(int(v)) if float(v).is_integer() else f"{v:.2f}" for v in vals]
        return pd.DataFrame({"reference": r, "current": c}, index=idx)

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        edges = np.unique(np.quantile(ref, [0.0, 0.5, 1.0]))
    hist_edges = edges.astype(float).copy()
    hist_edges[0], hist_edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, hist_edges)[0] / max(1, ref.size)
    c = np.histogram(cur, hist_edges)[0] / max(1, cur.size)
    labels = [f"{_fmt_num(edges[i])}–{_fmt_num(edges[i + 1])}" for i in range(len(edges) - 1)]
    return pd.DataFrame({"reference": r, "current": c}, index=labels)


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


def _model_scores(df: pd.DataFrame, ensemble: dict) -> dict[str, np.ndarray]:
    """Per-model prediction scores for a window (streaming: dest-history off)."""
    enriched = enrich(df, use_dest_history=False)
    return score_all(enriched, ensemble)


def psi_map(ref: pd.DataFrame, cur: pd.DataFrame, ensemble: dict,
            features=MONITORED) -> dict[str, float]:
    """PSI for every monitored feature + one prediction row PER MODEL.

    Rows: <feature> for each monitored input, and PREDICTION_SCORE_<key> for
    each model in the ensemble. Reused by the CLI report and the dashboard.
    """
    out: dict[str, float] = {f: psi(ref[f].to_numpy(), cur[f].to_numpy()) for f in features}
    s_ref = _model_scores(ref, ensemble)
    s_cur = _model_scores(cur, ensemble)
    for key in s_ref:
        out[f"PREDICTION_SCORE_{key}"] = psi(s_ref[key], s_cur[key])
    return out


def compute_drift(ref: pd.DataFrame, cur: pd.DataFrame, ensemble: dict,
                  features=MONITORED) -> pd.DataFrame:
    """Single-scenario drift table: feature | psi | trigger. Used by the dashboard."""
    m = psi_map(ref, cur, ensemble, features)
    table = pd.DataFrame({"feature": list(m), "psi": [round(v, 3) for v in m.values()]})
    table["trigger"] = table["psi"].map(band)
    return table


def split_reference_current(df: pd.DataFrame):
    """Temporal split: earlier half = reference, later half = current."""
    med = df["day_index"].median()
    return df[df["day_index"] <= med], df[df["day_index"] > med], med


def main():
    rng = np.random.default_rng(0)
    df = load()
    ensemble = load_ensemble(MODELS / "fraud_ensemble.joblib")

    ref, cur_natural, med = split_reference_current(df)
    cur_drift = inject_drift(cur_natural, rng)

    nat = psi_map(ref, cur_natural, ensemble)
    dft = psi_map(ref, cur_drift, ensemble)

    names = list(nat)   # monitored features, then PREDICTION_SCORE_<key> per model
    table = pd.DataFrame({
        "feature": names,
        "psi_natural": [round(nat[n], 3) for n in names],
        "psi_drifted": [round(dft[n], 3) for n in names],
    })
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
