"""Feature engineering for PaySim (Module 4).

The module keeps the simple `feature_matrix` API used by older M5 code, and
adds a train-fitted `FeatureTransformer` for leakage-aware train/serve use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from config import DATA_PROCESSED, DOCS, MODELS


TARGET = "isFraud"
TREE_GROUP = "tree"
LINEAR_GROUP = "linear"
TRANSFORMER_NAME = "feature_transformer.joblib"
PREVIEW_NAME = "features_preview.parquet"

TYPE_VALUES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
FREQ_COLS = ["browser", "device_os", "billing_country"]
DISPLAY_ID_COLS = [
    "customer_id", "customer_name", "email", "billing_city", "device_id",
    "nameOrig", "nameDest",
]

BASE_NUMERIC = [
    "amount", "log_amount", "amount_cents",
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "errorBalanceOrig", "errorBalanceDest",
    "orig_drained", "dest_was_empty",
    "is_transfer", "is_cash_out",
]

BASE_REALISTIC = [
    "amount", "log_amount", "amount_cents",
    "oldbalanceOrg", "oldbalanceDest",
    "is_transfer", "is_cash_out",
]

SYNTH_NUMERIC = [
    "account_age_days",
    "is_new_device", "shipping_billing_mismatch", "num_failed_payment_attempts",
    "ip_billing_distance_km", "log_ip_distance",
    "is_disposable_email", "high_risk_country",
    "hour_of_day", "is_night",
    "txn_count_last_24h", "time_since_last_hours", "account_txn_total",
]

DEST_HISTORY_NUMERIC = [
    "dest_seen_before",
    "dest_txn_count_so_far",
    "dest_amount_sum_so_far",
    "dest_amount_mean_so_far",
    "dest_amount_std_so_far",
    "dest_unique_senders_so_far",
    "dest_cash_in_count_so_far",
    "dest_cash_out_count_so_far",
    "dest_debit_count_so_far",
    "dest_payment_count_so_far",
    "dest_transfer_count_so_far",
    "time_since_dest_last_seen",
    "amount_to_dest_mean_ratio",
    "amount_dest_zscore",
    "dest_freq_so_far",
    "orig_freq_so_far",
]

TYPE_ONEHOT_COLS = [f"type__{value}" for value in TYPE_VALUES]
FREQ_FEATURE_COLS = [f"{col}_freq_train" for col in FREQ_COLS]
ENCODED_CATEGORICAL = TYPE_ONEHOT_COLS + FREQ_FEATURE_COLS

ALL_NUMERIC = BASE_NUMERIC + DEST_HISTORY_NUMERIC + SYNTH_NUMERIC + ENCODED_CATEGORICAL
REALISTIC_NUMERIC = BASE_REALISTIC + DEST_HISTORY_NUMERIC + SYNTH_NUMERIC + ENCODED_CATEGORICAL


def load_feature_source() -> pd.DataFrame:
    path = DATA_PROCESSED / "transactions_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python src/cleaning.py` first.")
    return pd.read_parquet(path)


def time_split_indices(df: pd.DataFrame, test_frac: float = 0.15, val_frac: float = 0.15):
    if test_frac + val_frac >= 1:
        raise ValueError("test_frac + val_frac must be < 1")
    order = np.lexsort((np.arange(len(df)), df["step"].to_numpy()))
    n = len(order)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    n_train = n - n_val - n_test
    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]
    bounds = {
        "train_step_min": int(df.iloc[train_idx]["step"].min()),
        "train_step_max": int(df.iloc[train_idx]["step"].max()),
        "val_step_min": int(df.iloc[val_idx]["step"].min()),
        "val_step_max": int(df.iloc[val_idx]["step"].max()),
        "test_step_min": int(df.iloc[test_idx]["step"].min()),
        "test_step_max": int(df.iloc[test_idx]["step"].max()),
    }
    return train_idx, val_idx, test_idx, bounds


def add_destination_history(df: pd.DataFrame) -> pd.DataFrame:
    """Add past-only `nameDest` aggregations using stable step/index order."""
    x = df.copy()
    original_index = x.index
    ordered = x.assign(_orig_pos=np.arange(len(x))).sort_values(["step", "_orig_pos"], kind="mergesort")
    g = ordered.groupby("nameDest", sort=False)

    ordered["dest_txn_count_so_far"] = g.cumcount().astype("int32")
    ordered["dest_seen_before"] = (ordered["dest_txn_count_so_far"] > 0).astype("int8")

    amount = ordered["amount"].astype(float)
    ordered["dest_amount_sum_so_far"] = (
        amount.groupby(ordered["nameDest"]).cumsum() - amount
    ).astype("float32")
    amount_sq = amount * amount
    ordered["_dest_amount_sumsq_so_far"] = amount_sq.groupby(ordered["nameDest"]).cumsum() - amount_sq
    count = ordered["dest_txn_count_so_far"].astype(float)
    mean = ordered["dest_amount_sum_so_far"] / count.replace(0, np.nan)
    var = (ordered["_dest_amount_sumsq_so_far"] / count.replace(0, np.nan)) - (mean * mean)
    ordered["dest_amount_mean_so_far"] = mean.fillna(0).astype("float32")
    ordered["dest_amount_std_so_far"] = np.sqrt(var.clip(lower=0)).fillna(0).astype("float32")

    ordered["_pair_first_seen"] = ~ordered.duplicated(["nameDest", "nameOrig"], keep="first")
    ordered["dest_unique_senders_so_far"] = (
        ordered.groupby("nameDest")["_pair_first_seen"].cumsum() - ordered["_pair_first_seen"].astype(int)
    ).astype("int32")

    prev_step = g["step"].shift(1)
    ordered["time_since_dest_last_seen"] = (ordered["step"] - prev_step).fillna(-1).astype("int16")

    for txn_type in TYPE_VALUES:
        col = f"dest_{txn_type.lower()}_count_so_far"
        current = (ordered["type"] == txn_type).astype("int32")
        ordered[col] = (
            current.groupby(ordered["nameDest"]).cumsum() - current
        ).astype("int32")

    ordered["amount_to_dest_mean_ratio"] = (
        ordered["amount"] / ordered["dest_amount_mean_so_far"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0).clip(upper=1_000).astype("float32")
    ordered["amount_dest_zscore"] = (
        (ordered["amount"] - ordered["dest_amount_mean_so_far"])
        / ordered["dest_amount_std_so_far"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=-1_000, upper=1_000).astype("float32")

    prior_rows = pd.Series(np.arange(len(ordered)), index=ordered.index).replace(0, np.nan)
    ordered["dest_freq_so_far"] = (
        ordered["dest_txn_count_so_far"] / prior_rows
    ).fillna(0).astype("float32")
    ordered["orig_freq_so_far"] = (
        ordered.groupby("nameOrig").cumcount() / prior_rows
    ).fillna(0).astype("float32")

    drop_cols = ["_orig_pos", "_dest_amount_sumsq_so_far", "_pair_first_seen"]
    out = ordered.sort_values("_orig_pos", kind="mergesort").drop(columns=drop_cols)
    out.index = original_index
    return out


def add_base_and_context_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["log_amount"] = np.log1p(x["amount"])
    x["amount_cents"] = ((x["amount"] * 100).round().astype("int64") % 100).astype("int8")
    x["errorBalanceOrig"] = x["oldbalanceOrg"] - x["amount"] - x["newbalanceOrig"]
    x["errorBalanceDest"] = x["oldbalanceDest"] + x["amount"] - x["newbalanceDest"]
    x["orig_drained"] = ((x["newbalanceOrig"] == 0) & (x["oldbalanceOrg"] > 0)).astype("int8")
    x["dest_was_empty"] = (x["oldbalanceDest"] == 0).astype("int8")
    x["is_transfer"] = (x["type"] == "TRANSFER").astype("int8")
    x["is_cash_out"] = (x["type"] == "CASH_OUT").astype("int8")
    x["log_ip_distance"] = np.log1p(x["ip_billing_distance_km"])
    return x


def add_fixed_onehot(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for value in TYPE_VALUES:
        x[f"type__{value}"] = (x["type"] == value).astype("int8")
    return x


def has_destination_history(df: pd.DataFrame) -> bool:
    return set(DEST_HISTORY_NUMERIC).issubset(df.columns)


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Precompute causal destination history once before train/val/test split."""
    x = add_destination_history(df)
    x = add_base_and_context_features(x)
    x = add_fixed_onehot(x)
    for col in FREQ_FEATURE_COLS:
        if col not in x:
            x[col] = 0.0
    return x


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return df plus deterministic numeric features.

    Frequency columns are set to past-only defaults here. `FeatureTransformer`
    overwrites categorical frequency columns with train-fitted maps. This
    legacy helper may compute destination history on the provided frame; the
    main M4/M5 path must call `prepare_feature_frame(full_df)` before splitting.
    """
    if has_destination_history(df):
        x = df.copy()
    else:
        x = prepare_feature_frame(df)
    for col in FREQ_FEATURE_COLS:
        if col not in x:
            x[col] = 0.0
    return x


def require_destination_history(df: pd.DataFrame) -> None:
    missing = [col for col in DEST_HISTORY_NUMERIC if col not in df.columns]
    if missing:
        raise ValueError(
            "Destination-history columns are missing. Run "
            "`prepare_feature_frame(full_df)` once before splitting; do not "
            "call FeatureTransformer on raw train/val/test subsets."
        )


def _select_cols(cols: list[str], df: pd.DataFrame) -> list[str]:
    return [c for c in cols if c in df.columns]


def feature_matrix(df: pd.DataFrame, groups: str = "all") -> tuple[pd.DataFrame, list[str]]:
    """Return (X, feature_names) for base/synth/dest/realistic/all."""
    x = build_features(df)
    if groups == "base":
        cols = BASE_NUMERIC + TYPE_ONEHOT_COLS
    elif groups == "synth":
        cols = SYNTH_NUMERIC + ENCODED_CATEGORICAL
    elif groups == "dest":
        cols = DEST_HISTORY_NUMERIC
    elif groups == "realistic":
        cols = REALISTIC_NUMERIC
    elif groups == "all":
        cols = ALL_NUMERIC
    else:
        raise ValueError(f"unknown feature group: {groups}")
    cols = _select_cols(cols, x)
    return x[cols].astype("float32"), cols


@dataclass
class FeatureTransformer:
    drop_cols: list[str] = field(default_factory=lambda: list(DISPLAY_ID_COLS))
    frequency_cols: list[str] = field(default_factory=lambda: list(FREQ_COLS))
    fitted_: bool = False
    freq_maps_: dict[str, dict[object, float]] = field(default_factory=dict)
    tree_feature_names_: list[str] = field(default_factory=list)
    linear_feature_names_: list[str] = field(default_factory=list)
    linear_imputer_: SimpleImputer | None = None
    linear_scaler_: StandardScaler | None = None
    fit_stats_: dict[str, object] = field(default_factory=dict)

    def fit(self, train_df: pd.DataFrame) -> "FeatureTransformer":
        require_destination_history(train_df)
        x = build_features(train_df)
        self.freq_maps_ = {}
        for col in self.frequency_cols:
            values = train_df[col].astype("string").fillna("__missing__")
            self.freq_maps_[col] = values.value_counts(normalize=True).to_dict()
        x = self._apply_frequency_maps(x)

        feature_names = _select_cols(ALL_NUMERIC, x)
        constant_cols = [c for c in feature_names if x[c].nunique(dropna=True) <= 1]
        feature_names = [c for c in feature_names if c not in constant_cols]
        self.tree_feature_names_ = feature_names
        self.linear_feature_names_ = feature_names

        self.linear_imputer_ = SimpleImputer(strategy="median")
        imputed = self.linear_imputer_.fit_transform(x[self.linear_feature_names_])
        self.linear_scaler_ = StandardScaler()
        self.linear_scaler_.fit(imputed)

        self.fit_stats_ = {
            "train_rows": len(train_df),
            "all_numeric_before_constant_drop": len(_select_cols(ALL_NUMERIC, x)),
            "constant_dropped": len(constant_cols),
            "tree_features": len(self.tree_feature_names_),
            "linear_features": len(self.linear_feature_names_),
            "dest_history_features": len(DEST_HISTORY_NUMERIC),
            "encoded_categorical_features": len(ENCODED_CATEGORICAL),
            "display_id_dropped": self.drop_cols,
            "constant_columns": constant_cols,
        }
        self.fitted_ = True
        return self

    def _apply_frequency_maps(self, x: pd.DataFrame) -> pd.DataFrame:
        out = x.copy()
        for col in self.frequency_cols:
            feature = f"{col}_freq_train"
            mapping = self.freq_maps_.get(col, {})
            out[feature] = (
                out[col].astype("string").fillna("__missing__").map(mapping).fillna(0.0).astype("float32")
            )
        return out

    def transform(self, df: pd.DataFrame, group: str = TREE_GROUP) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("FeatureTransformer must be fit before transform.")
        require_destination_history(df)
        x = self._apply_frequency_maps(build_features(df))
        if group == TREE_GROUP:
            return x.reindex(columns=self.tree_feature_names_).astype("float32")
        if group == LINEAR_GROUP:
            raw = x.reindex(columns=self.linear_feature_names_).astype("float32")
            arr = self.linear_imputer_.transform(raw)
            arr = self.linear_scaler_.transform(arr).astype("float32")
            return pd.DataFrame(arr, columns=self.linear_feature_names_, index=df.index)
        raise ValueError(f"unknown group: {group}")

    def feature_names(self, group: str = TREE_GROUP) -> list[str]:
        if group == TREE_GROUP:
            return list(self.tree_feature_names_)
        if group == LINEAR_GROUP:
            return list(self.linear_feature_names_)
        raise ValueError(f"unknown group: {group}")


def imbalance_summary(y: pd.Series | np.ndarray) -> dict[str, float]:
    arr = np.asarray(y)
    pos = int(arr.sum())
    neg = int(len(arr) - pos)
    return {
        "rows": int(len(arr)),
        "positives": pos,
        "negatives": neg,
        "positive_rate": float(pos / max(1, len(arr))),
        "scale_pos_weight": float(neg / max(1, pos)),
    }


def _transform_rejects_raw_subset(transformer: FeatureTransformer, df: pd.DataFrame) -> bool:
    raw_subset = df.drop(columns=[c for c in DEST_HISTORY_NUMERIC if c in df.columns]).head(10)
    try:
        transformer.transform(raw_subset, TREE_GROUP)
    except ValueError:
        return True
    return False


def write_feature_report(
    df: pd.DataFrame,
    transformer: FeatureTransformer,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    bounds: dict[str, int],
) -> None:
    y = df[TARGET]
    train_imb = imbalance_summary(df.iloc[train_idx][TARGET])
    full_imb = imbalance_summary(y)
    stats = transformer.fit_stats_

    sample_features = build_features(df.iloc[: min(100_000, len(df))])
    val_dest_seen_rate = float(df.iloc[val_idx]["dest_seen_before"].mean())
    val_dest_max_count = int(df.iloc[val_idx]["dest_txn_count_so_far"].max())
    test_dest_seen_rate = float(df.iloc[test_idx]["dest_seen_before"].mean())
    test_dest_max_count = int(df.iloc[test_idx]["dest_txn_count_so_far"].max())
    leakage_checks = pd.DataFrame([
        {
            "check": "first transaction for each destination has dest_txn_count_so_far=0",
            "status": bool((sample_features.loc[sample_features["dest_txn_count_so_far"] == 0, "dest_seen_before"] == 0).all()),
        },
        {
            "check": "time_since_dest_last_seen=-1 for unseen destinations",
            "status": bool((sample_features.loc[sample_features["dest_txn_count_so_far"] == 0, "time_since_dest_last_seen"] == -1).all()),
        },
        {
            "check": "raw display/id columns absent from feature matrix",
            "status": not any(c in transformer.tree_feature_names_ for c in DISPLAY_ID_COLS),
        },
        {
            "check": "transformer rejects raw subsets without precomputed dest-history",
            "status": _transform_rejects_raw_subset(transformer, df),
        },
    ])

    groups = pd.DataFrame([
        {"group": "base", "n_features": len(BASE_NUMERIC + TYPE_ONEHOT_COLS), "purpose": "full PaySim amount/balance signature"},
        {"group": "dest", "n_features": len(DEST_HISTORY_NUMERIC), "purpose": "past-only destination-account behaviour"},
        {"group": "synth", "n_features": len(SYNTH_NUMERIC + ENCODED_CATEGORICAL), "purpose": "M1 synthetic/contextual risk"},
        {"group": "realistic", "n_features": len(REALISTIC_NUMERIC), "purpose": "authorization-time-style features plus context"},
        {"group": "all", "n_features": len(ALL_NUMERIC), "purpose": "full feature set"},
    ])

    report = [
        "# Feature Engineering — PaySim Fraud Detection\n",
        "## Source / Split\n",
        f"- Source: `data/processed/transactions_clean.parquet` with **{len(df):,} rows**.\n",
        f"- Train/validation/test rows: **{len(train_idx):,} / {len(val_idx):,} / {len(test_idx):,}**.\n",
        (
            f"- Step windows: train {bounds['train_step_min']}..{bounds['train_step_max']}; "
            f"validation {bounds['val_step_min']}..{bounds['val_step_max']}; "
            f"test {bounds['test_step_min']}..{bounds['test_step_max']}.\n"
        ),
        "- Transformer state is fit on train only; train-fitted frequency maps, imputer, and scaler are reused for validation/test.\n",
        "- Destination-history features are precomputed once on the full chronological frame before splitting, so validation/test rows can see train history but never future rows.\n",
        "\n## Engineered Features\n",
        "- Base PaySim: amount, log amount, cents, transfer/cash-out flags, balance reconciliation errors, drained/empty balance flags.\n",
        "- Destination history: past-only counts, amount sums/means/std, unique senders, type counts, recency, amount ratio/z-score, past frequency.\n",
        "- Synthetic/context: account age, device/risk flags, IP distance, time features, illustrative synthetic-customer velocity.\n",
        "- Categorical encoding: fixed one-hot for `type`; train-frequency encoding for `browser`, `device_os`, and `billing_country`.\n",
        "- Dropped from model matrices: raw IDs/display fields (`nameOrig`, `nameDest`, `customer_id`, `customer_name`, `email`, `billing_city`, `device_id`).\n",
        "\n## Feature Groups\n",
        groups.to_markdown(index=False) + "\n",
        "\n## Transformer Output\n",
        pd.DataFrame([
            {"matrix": "tree", "n_features": stats["tree_features"], "scaling": "none"},
            {"matrix": "linear", "n_features": stats["linear_features"], "scaling": "median impute + StandardScaler"},
        ]).to_markdown(index=False) + "\n",
        f"- Constant columns dropped on train: **{stats['constant_dropped']}**.\n",
        f"- Destination-history features: **{stats['dest_history_features']}**.\n",
        f"- Encoded categorical features: **{stats['encoded_categorical_features']}**.\n",
        "\n## Imbalance Handling Guidance for M5\n",
        f"- Full fraud prevalence: **{full_imb['positive_rate']:.4%}**.\n",
        f"- Train fraud prevalence: **{train_imb['positive_rate']:.4%}**.\n",
        f"- Recommended model weight default: `scale_pos_weight={train_imb['scale_pos_weight']:.2f}` or class weights.\n",
        "- SMOTE/undersampling, if used, must be applied only inside the train fold after transformation.\n",
        "\n## Leakage Checks\n",
        leakage_checks.assign(status=leakage_checks["status"].map({True: "PASS", False: "FAIL"})).to_markdown(index=False) + "\n",
        "\n## Destination History Distribution Check\n",
        pd.DataFrame([
            {"split": "validation", "dest_seen_before_rate": round(val_dest_seen_rate, 4), "max_dest_txn_count_so_far": val_dest_max_count},
            {"split": "test", "dest_seen_before_rate": round(test_dest_seen_rate, 4), "max_dest_txn_count_so_far": test_dest_max_count},
        ]).to_markdown(index=False) + "\n",
        "\n## Notes\n",
        "- `nameDest`/`nameOrig` raw strings are not used as features; only past-derived aggregates or train-frequency encodings are used.\n",
        "- Full balance features are intentionally separated from `realistic` features because PaySim post-transaction balances can make the task artificially easy.\n",
    ]
    (DOCS / "feature_engineering.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    df = load_feature_source().reset_index(drop=True)
    df = prepare_feature_frame(df)
    train_idx, val_idx, test_idx, bounds = time_split_indices(df)
    transformer = FeatureTransformer().fit(df.iloc[train_idx])

    X_train_tree = transformer.transform(df.iloc[train_idx], TREE_GROUP)
    X_val_tree = transformer.transform(df.iloc[val_idx[: min(5_000, len(val_idx))]], TREE_GROUP)
    X_test_tree = transformer.transform(df.iloc[test_idx[: min(5_000, len(test_idx))]], TREE_GROUP)
    X_train_linear = transformer.transform(df.iloc[train_idx], LINEAR_GROUP)

    assert X_train_tree.columns.tolist() == X_val_tree.columns.tolist() == X_test_tree.columns.tolist()
    assert X_train_tree.select_dtypes(exclude=["number"]).empty
    assert X_train_linear.select_dtypes(exclude=["number"]).empty
    assert int(X_train_linear.isna().sum().sum()) == 0
    assert not any(c in X_train_tree.columns for c in DISPLAY_ID_COLS)
    assert float(df.iloc[val_idx]["dest_seen_before"].mean()) > 0.20
    assert int(df.iloc[val_idx]["dest_txn_count_so_far"].max()) > 10

    MODELS.mkdir(exist_ok=True)
    sys.modules.setdefault("features", sys.modules[__name__])
    FeatureTransformer.__module__ = "features"
    joblib.dump(transformer, MODELS / TRANSFORMER_NAME)
    write_feature_report(df, transformer, train_idx, val_idx, test_idx, bounds)

    preview_idx = np.r_[train_idx[:200], val_idx[:100], test_idx[:100]]
    preview = transformer.transform(df.iloc[preview_idx], LINEAR_GROUP)
    preview.insert(0, TARGET, df.iloc[preview_idx][TARGET].to_numpy())
    preview.to_parquet(DATA_PROCESSED / PREVIEW_NAME, index=False)

    train_imb = imbalance_summary(df.iloc[train_idx][TARGET])
    print(f"[features] rows={len(df):,} train/val/test={len(train_idx):,}/{len(val_idx):,}/{len(test_idx):,}")
    print(f"[features] tree_features={len(transformer.tree_feature_names_):,} linear_features={len(transformer.linear_feature_names_):,}")
    print(f"[features] train_fraud={train_imb['positives']:,} ({train_imb['positive_rate']:.4%}) scale_pos_weight={train_imb['scale_pos_weight']:.2f}")
    print(f"[features] transformer -> {MODELS / TRANSFORMER_NAME}")
    print(f"[features] report -> {DOCS / 'feature_engineering.md'}")
    print(f"[features] preview -> {DATA_PROCESSED / PREVIEW_NAME}")


if __name__ == "__main__":
    main()
