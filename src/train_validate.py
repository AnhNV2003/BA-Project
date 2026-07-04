"""Thin modelling slice (Modules 4-5, draft) — VALIDATES the synthetic layer.

Answers three questions the team needs before investing further:
  1. Do the synthetic features carry learnable signal?      (synth-only AUC-PR)
  2. Is there leakage?  (no single synth feature ~1.0 AUC; synth-only AUC < 1.0)
  3. Does the end-to-end pipeline produce a deployable model + a cost-based
     operating threshold?

Run:  python train_validate.py
"""
from __future__ import annotations

import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import DATA_PROCESSED, MODELS, SEED, COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE, COST_MANUAL_REVIEW
from features import feature_matrix, build_features, BASE_NUMERIC, SYNTH_NUMERIC

warnings.filterwarnings("ignore")


def load_processed() -> pd.DataFrame:
    p = DATA_PROCESSED / "transactions_context.parquet"
    if not p.exists():
        p = p.with_suffix(".csv")
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def ap_auc(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    s = model.predict_proba(Xte)[:, 1]
    return average_precision_score(yte, s), roc_auc_score(yte, s), s, model


def main():
    df = load_processed()
    y = df["isFraud"].to_numpy()
    print(f"Loaded {len(df):,} rows | fraud {y.sum():,} ({y.mean():.4%})")

    idx_tr, idx_te = train_test_split(
        np.arange(len(df)), test_size=0.30, random_state=SEED, stratify=y
    )
    ytr, yte = y[idx_tr], y[idx_te]
    spw = (ytr == 0).sum() / max(1, (ytr == 1).sum())  # scale_pos_weight

    def xgb():
        return XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, scale_pos_weight=spw,
            eval_metric="aucpr", n_jobs=-1, random_state=SEED, tree_method="hist",
        )

    # ---------- Experiment 1: feature-group contribution + leakage ----------
    print("\n=== Experiment 1: feature groups (XGBoost) ===")
    print(f"{'group':<12}{'#feat':>6}{'AUC-PR':>10}{'ROC-AUC':>10}")
    group_scores = {}
    for g in ["base", "synth", "realistic", "all"]:
        Xg, cols = feature_matrix(df, groups=g)
        Xg = Xg.to_numpy()
        ap, auc, _, _ = ap_auc(xgb(), Xg[idx_tr], ytr, Xg[idx_te], yte)
        group_scores[g] = (ap, auc)
        print(f"{g:<12}{len(cols):>6}{ap:>10.4f}{auc:>10.4f}")

    # ---------- Leakage guard: single-feature AUC on synthetic ----------
    print("\n=== Leakage check: single-feature ROC-AUC (synthetic) ===")
    Xall = build_features(df)
    singles = []
    for c in SYNTH_NUMERIC:
        v = Xall[c].to_numpy()[idx_te].astype(float)
        try:
            a = roc_auc_score(yte, v)
        except ValueError:
            a = 0.5
        singles.append((c, max(a, 1 - a)))  # direction-agnostic separability
    singles.sort(key=lambda t: -t[1])
    for c, a in singles[:5]:
        print(f"  {c:<28}{a:.3f}")
    max_single = singles[0][1]
    synth_only_auc = group_scores["synth"][1]
    print(f"\n  max single synthetic AUC = {max_single:.3f}  (want < ~0.98)")
    print(f"  synth-only model  AUC    = {synth_only_auc:.3f}  (want high but < 1.0)")
    verdict = "PASS (signal present, no single-feature leakage)" if (max_single < 0.98 and synth_only_auc < 0.999) else "CHECK — possible leakage"
    print(f"  VERDICT: {verdict}")

    # ---------- Experiment 2: model comparison (all features = main model) ----------
    # Simple, standard path: use every feature. (For a richer story with a real
    # precision/recall trade-off, switch groups to "realistic" — see Experiment 1.)
    print("\n=== Experiment 2: model comparison (all features) ===")
    Xall_m, cols_all = feature_matrix(df, groups="all")
    Xall_m = Xall_m.to_numpy()
    models = {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, class_weight="balanced_subsample", random_state=SEED),
        "XGBoost": xgb(),
    }
    print(f"{'model':<14}{'AUC-PR':>10}{'ROC-AUC':>10}")
    best = (None, -1, None, None)
    for name, mdl in models.items():
        ap, auc, s, fitted = ap_auc(mdl, Xall_m[idx_tr], ytr, Xall_m[idx_te], yte)
        print(f"{name:<14}{ap:>10.4f}{auc:>10.4f}")
        if ap > best[1]:
            best = (name, ap, s, fitted)
    best_name, best_ap, best_scores, best_model = best
    print(f"-> best: {best_name} (AUC-PR={best_ap:.4f})")

    # ---------- Cost-based threshold selection ----------
    print("\n=== Cost-based operating threshold (best model) ===")
    amt_te = df["amount"].to_numpy()[idx_te]
    thr, cost, prec, rec, f1 = choose_threshold(yte, best_scores, amt_te)
    base_cost = expected_cost(yte, np.zeros_like(yte), amt_te)  # block nothing
    print(f"  chosen threshold : {thr:.4f}")
    print(f"  precision/recall/F1: {prec:.3f} / {rec:.3f} / {f1:.3f}")
    print(f"  expected cost    : {cost:,.0f}   (vs {base_cost:,.0f} doing nothing"
          f"  -> {100*(1-cost/base_cost):.1f}% loss avoided)")

    # ---------- Feature importance (base vs synth share) ----------
    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_, index=cols_all).sort_values(ascending=False)
        base_share = imp[imp.index.isin(BASE_NUMERIC)].sum()
        synth_share = imp[imp.index.isin(SYNTH_NUMERIC)].sum()
        print(f"\n=== Feature importance ({best_name}) ===")
        print(f"  base share = {base_share:.2f} | synth share = {synth_share:.2f}")
        print("  top 10:")
        for c, w in imp.head(10).items():
            tag = "B" if c in BASE_NUMERIC else "S"
            print(f"    [{tag}] {c:<28}{w:.3f}")

    # ---------- Persist for P3 ----------
    MODELS.mkdir(exist_ok=True)
    joblib.dump({"model": best_model, "features": cols_all, "threshold": thr,
                 "model_name": best_name}, MODELS / "fraud_model.joblib")
    print(f"\nSaved model -> {MODELS / 'fraud_model.joblib'}")


def expected_cost(y_true, y_pred, amount):
    fn = (y_true == 1) & (y_pred == 0)
    fp = (y_true == 0) & (y_pred == 1)
    tp = (y_true == 1) & (y_pred == 1)
    return (COST_FALSE_NEGATIVE * amount[fn].sum()
            + COST_FALSE_POSITIVE * fp.sum()
            + COST_MANUAL_REVIEW * tp.sum())


def choose_threshold(y_true, scores, amount):
    best = (0.5, np.inf, 0, 0, 0)
    for thr in np.linspace(0.01, 0.99, 99):
        pred = (scores >= thr).astype(int)
        c = expected_cost(y_true, pred, amount)
        if c < best[1]:
            p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
            best = (thr, c, p, r, f1)
    return best


if __name__ == "__main__":
    main()
