"""Model development for PaySim fraud detection (Module 5)."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from xgboost import XGBClassifier

from config import (
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    COST_MANUAL_REVIEW,
    DATA_PROCESSED,
    DOCS,
    MODELS,
    SEED,
)
from features import (
    ALL_NUMERIC,
    BASE_NUMERIC,
    DEST_HISTORY_NUMERIC,
    ENCODED_CATEGORICAL,
    FeatureTransformer,
    LINEAR_GROUP,
    REALISTIC_NUMERIC,
    SYNTH_NUMERIC,
    TARGET,
    TREE_GROUP,
    TYPE_ONEHOT_COLS,
    prepare_feature_frame,
    time_split_indices,
)

warnings.filterwarnings("ignore")


FEATURE_GROUPS = {
    "base": BASE_NUMERIC + TYPE_ONEHOT_COLS,
    "dest": DEST_HISTORY_NUMERIC,
    "synth": SYNTH_NUMERIC + ENCODED_CATEGORICAL,
    "realistic": REALISTIC_NUMERIC,
    "all": ALL_NUMERIC,
}

# Groups that contain post-transaction balance signals (errorBalance*, orig_drained,
# newbalance*). In PaySim these near-deterministically encode the label (fraud drains
# the account), so they give artificial AUC-PR=1.0 but are unavailable at
# authorization time. They stay in the comparison table as a leaky upper bound, but
# are EXCLUDED from deployable bundle selection.
LEAKY_GROUPS = {"base", "all"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare PaySim fraud classifiers.")
    parser.add_argument("--models", default="logreg,rf,xgb", help="Comma-separated: logreg,rf,xgb")
    parser.add_argument("--feature-groups", default="base,dest,synth,realistic,all")
    parser.add_argument("--threshold-grid-size", type=int, default=101)
    parser.add_argument("--sample-frac", type=float, default=None, help="Optional stratified sample for quick smoke runs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate pipeline without fitting models.")
    return parser.parse_args()


def load_clean() -> pd.DataFrame:
    path = DATA_PROCESSED / "transactions_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python src/cleaning.py` first.")
    return pd.read_parquet(path)


def maybe_sample(df: pd.DataFrame, frac: float | None) -> pd.DataFrame:
    if frac is None:
        return df
    if not 0 < frac <= 1:
        raise ValueError("--sample-frac must be in (0, 1].")
    return (
        df.groupby(TARGET, group_keys=False)
        .sample(frac=frac, random_state=SEED)
        .sort_values(["step"])
        .reset_index(drop=True)
    )


def class_ratio(y: np.ndarray) -> float:
    pos = int(y.sum())
    neg = int(len(y) - pos)
    return float(neg / max(1, pos))


def model_specs(selected: list[str], scale_pos_weight: float) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for key in selected:
        if key == "logreg":
            specs.append({
                "key": "logreg",
                "name": "Logistic Regression",
                "matrix": LINEAR_GROUP,
                "model": LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1, random_state=SEED),
            })
        elif key == "rf":
            specs.append({
                "key": "rf",
                "name": "Random Forest",
                "matrix": TREE_GROUP,
                "model": RandomForestClassifier(
                    n_estimators=120,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                    random_state=SEED,
                ),
            })
        elif key == "xgb":
            specs.append({
                "key": "xgb",
                "name": "XGBoost",
                "matrix": TREE_GROUP,
                "model": XGBClassifier(
                    n_estimators=220,
                    max_depth=5,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="aucpr",
                    n_jobs=-1,
                    random_state=SEED,
                    tree_method="hist",
                ),
            })
        else:
            raise ValueError(f"Unknown model key: {key}")
    return specs


def subset_features(X: pd.DataFrame, group: str) -> pd.DataFrame:
    cols = [col for col in FEATURE_GROUPS[group] if col in X.columns]
    if not cols:
        raise ValueError(f"No columns available for feature group {group!r}")
    return X[cols]


def expected_cost(y_true: np.ndarray, y_pred: np.ndarray, amount: np.ndarray) -> float:
    fn = (y_true == 1) & (y_pred == 0)
    fp = (y_true == 0) & (y_pred == 1)
    flagged = y_pred == 1
    return float(
        COST_FALSE_NEGATIVE * amount[fn].sum()
        + COST_FALSE_POSITIVE * fp.sum()
        + COST_MANUAL_REVIEW * flagged.sum()
    )


def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, amount: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    fraud_amount = float(amount[y_true == 1].sum())
    caught_amount = float(amount[(y_true == 1) & (pred == 1)].sum())
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "expected_cost": expected_cost(y_true, pred, amount),
        "loss_avoided_pct": float(caught_amount / fraud_amount * 100) if fraud_amount else 0.0,
        "flagged_rate": float(pred.mean()),
        "fraud_amount": fraud_amount,
        "caught_fraud_amount": caught_amount,
    }


def choose_threshold(y_true: np.ndarray, scores: np.ndarray, amount: np.ndarray, grid_size: int) -> dict[str, float]:
    thresholds = np.linspace(0.001, 0.999, grid_size)
    best: dict[str, float] | None = None
    for threshold in thresholds:
        metrics = metrics_at_threshold(y_true, scores, amount, float(threshold))
        if best is None or metrics["expected_cost"] < best["expected_cost"]:
            best = metrics
    return best


def score_auc(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    return float(average_precision_score(y_true, scores)), float(roc_auc_score(y_true, scores))


def write_report(results: pd.DataFrame, best: dict[str, object], split_info: dict[str, object]) -> None:
    display_cols = [
        "model_name", "feature_group", "is_leaky", "n_features",
        "val_auc_pr", "val_roc_auc", "val_threshold", "val_expected_cost",
        "val_precision", "val_recall", "val_f1", "val_loss_avoided_pct",
        "test_auc_pr", "test_roc_auc", "test_expected_cost",
        "test_precision", "test_recall", "test_f1", "test_loss_avoided_pct",
    ]
    lines = [
        "# Model Development — PaySim Fraud Detection\n",
        "## Methodology\n",
        "- `prepare_feature_frame(full_df)` runs once before split, preserving causal `nameDest` history across train/validation/test.\n",
        "- `FeatureTransformer` fits frequency maps, imputer, scaler, and feature schema on train only.\n",
        "- Threshold and best model are selected on validation expected cost. Test is evaluated once for the selected configuration.\n",
        "- Cost uses missed fraud amount, false-positive friction cost, and review cost for all flagged transactions.\n",
        f"- **Leaky groups {sorted(LEAKY_GROUPS)} are excluded from deployable bundle selection** (post-transaction balances are unavailable at authorization time and near-deterministically encode the label). They remain below as a leaky upper-bound reference.\n",
        "\n## Split\n",
        pd.DataFrame([split_info]).to_markdown(index=False) + "\n",
        "\n## Best Deployable Configuration (non-leaky groups only)\n",
        pd.DataFrame([best]).drop(columns=["model_obj", "transformer"], errors="ignore").to_markdown(index=False) + "\n",
        "\n## Results (all groups; `is_leaky=True` = upper-bound reference, not deployable)\n",
        results[display_cols].sort_values(["is_leaky", "val_expected_cost", "model_name", "feature_group"]).to_markdown(index=False) + "\n",
        "\n## Notes\n",
        "- Leaky groups (`base`/`all`) may reach AUC-PR≈1.0 because PaySim post-transaction balances encode strong reconciliation signals; treat these as an upper bound, not a deployable result.\n",
        "- `realistic` and `dest` groups reflect deployable pre-authorization signal and drive the saved bundle.\n",
    ]
    (DOCS / "model_development.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    selected_models = [item.strip().lower() for item in args.models.split(",") if item.strip()]
    selected_groups = [item.strip().lower() for item in args.feature_groups.split(",") if item.strip()]
    unknown_groups = sorted(set(selected_groups) - set(FEATURE_GROUPS))
    if unknown_groups:
        raise ValueError(f"Unknown feature groups: {unknown_groups}")

    raw = maybe_sample(load_clean(), args.sample_frac).reset_index(drop=True)
    df = prepare_feature_frame(raw)
    train_idx, val_idx, test_idx, bounds = time_split_indices(df)
    train_df, val_df, test_df = df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]
    y_train = train_df[TARGET].to_numpy()
    y_val = val_df[TARGET].to_numpy()
    y_test = test_df[TARGET].to_numpy()
    amount_val = val_df["amount"].to_numpy(dtype=float)
    amount_test = test_df["amount"].to_numpy(dtype=float)

    transformer = FeatureTransformer().fit(train_df)
    X_train_tree = transformer.transform(train_df, TREE_GROUP)
    X_val_tree = transformer.transform(val_df, TREE_GROUP)
    X_test_tree = transformer.transform(test_df, TREE_GROUP)
    X_train_linear = transformer.transform(train_df, LINEAR_GROUP)
    X_val_linear = transformer.transform(val_df, LINEAR_GROUP)
    X_test_linear = transformer.transform(test_df, LINEAR_GROUP)

    val_dest_seen_rate = float(val_df["dest_seen_before"].mean())
    if val_dest_seen_rate < 0.20:
        raise RuntimeError(f"dest-history appears reset on validation: seen rate={val_dest_seen_rate:.4f}")

    split_info = {
        "rows": len(df),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_fraud_rate": y_train.mean(),
        "val_fraud_rate": y_val.mean(),
        "test_fraud_rate": y_test.mean(),
        "val_dest_seen_before_rate": val_dest_seen_rate,
        "val_max_dest_txn_count_so_far": int(val_df["dest_txn_count_so_far"].max()),
        **bounds,
    }
    print(f"[m5] rows={len(df):,} train/val/test={len(train_df):,}/{len(val_df):,}/{len(test_df):,}")
    print(f"[m5] val_dest_seen_before={val_dest_seen_rate:.4f} max_count={split_info['val_max_dest_txn_count_so_far']}")
    print(f"[m5] train scale_pos_weight={class_ratio(y_train):.2f}")

    if args.dry_run:
        print("[m5] dry-run complete; no models fitted.")
        return

    specs = model_specs(selected_models, class_ratio(y_train))
    rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None

    for spec in specs:
        for group in selected_groups:
            matrix = spec["matrix"]
            X_train_src, X_val_src, X_test_src = (
                (X_train_linear, X_val_linear, X_test_linear)
                if matrix == LINEAR_GROUP else
                (X_train_tree, X_val_tree, X_test_tree)
            )
            X_train = subset_features(X_train_src, group)
            X_val = subset_features(X_val_src, group)
            X_test = subset_features(X_test_src, group)

            model = spec["model"]
            print(f"[m5] fitting {spec['name']} / {group} ({X_train.shape[1]} features)")
            model.fit(X_train, y_train)
            val_scores = model.predict_proba(X_val)[:, 1]
            test_scores = model.predict_proba(X_test)[:, 1]
            val_auc_pr, val_roc_auc = score_auc(y_val, val_scores)
            test_auc_pr, test_roc_auc = score_auc(y_test, test_scores)
            val_threshold_metrics = choose_threshold(y_val, val_scores, amount_val, args.threshold_grid_size)
            test_metrics = metrics_at_threshold(y_test, test_scores, amount_test, val_threshold_metrics["threshold"])

            row = {
                "model_key": spec["key"],
                "model_name": spec["name"],
                "feature_group": group,
                "is_leaky": group in LEAKY_GROUPS,
                "matrix": matrix,
                "n_features": X_train.shape[1],
                "val_auc_pr": val_auc_pr,
                "val_roc_auc": val_roc_auc,
                "val_threshold": val_threshold_metrics["threshold"],
                "val_expected_cost": val_threshold_metrics["expected_cost"],
                "val_precision": val_threshold_metrics["precision"],
                "val_recall": val_threshold_metrics["recall"],
                "val_f1": val_threshold_metrics["f1"],
                "val_loss_avoided_pct": val_threshold_metrics["loss_avoided_pct"],
                "val_flagged_rate": val_threshold_metrics["flagged_rate"],
                "test_auc_pr": test_auc_pr,
                "test_roc_auc": test_roc_auc,
                "test_expected_cost": test_metrics["expected_cost"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_loss_avoided_pct": test_metrics["loss_avoided_pct"],
                "test_flagged_rate": test_metrics["flagged_rate"],
            }
            rows.append(row)
            # Deployable bundle is chosen only among non-leaky groups; leaky
            # groups (base/all) remain in the table as an upper-bound reference.
            if group not in LEAKY_GROUPS:
                candidate = {**row, "model_obj": model, "transformer": transformer}
                if best is None or row["val_expected_cost"] < best["val_expected_cost"]:
                    best = candidate

    if best is None:
        raise RuntimeError(
            "No deployable (non-leaky) feature group was trained. "
            f"Groups {sorted(LEAKY_GROUPS)} are excluded from bundle selection; "
            "include at least one of realistic/dest/synth via --feature-groups."
        )

    results = pd.DataFrame(rows)
    results.to_csv(DOCS / "model_results.csv", index=False)

    bundle = {
        "model": best["model_obj"],
        "transformer": best["transformer"],
        "feature_group": best["feature_group"],
        "leaky_excluded_from_selection": sorted(LEAKY_GROUPS),
        "matrix": best["matrix"],
        "features": subset_features(
            X_train_linear if best["matrix"] == LINEAR_GROUP else X_train_tree,
            best["feature_group"],
        ).columns.tolist(),
        "threshold": best["val_threshold"],
        "metrics": {k: v for k, v in best.items() if k not in {"model_obj", "transformer"}},
        "split_info": split_info,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    MODELS.mkdir(exist_ok=True)
    joblib.dump(bundle, MODELS / "fraud_model.joblib")

    serializable_best = {k: v for k, v in best.items() if k not in {"model_obj", "transformer"}}
    write_report(results, serializable_best, split_info)

    print("[m5] best DEPLOYABLE (non-leaky) by validation cost:")
    print(json.dumps(serializable_best, indent=2, default=float))
    print(f"[m5] results -> {DOCS / 'model_results.csv'}")
    print(f"[m5] report -> {DOCS / 'model_development.md'}")
    print(f"[m5] bundle -> {MODELS / 'fraud_model.joblib'}")


if __name__ == "__main__":
    main()
