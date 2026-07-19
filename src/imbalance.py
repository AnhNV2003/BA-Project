"""Train-fold-only imbalance utilities for Module 5 experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def class_weight_scale_pos(y: pd.Series | np.ndarray) -> float:
    arr = np.asarray(y)
    positives = int(arr.sum())
    negatives = int(len(arr) - positives)
    return float(negatives / max(1, positives))


def resample_train_only(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    strategy: str = "none",
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Apply optional resampling to a training fold only.

    Validation/test folds must never be passed here. M5 should call this after
    split and feature transformation, and only for models that benefit from
    explicit resampling.
    """
    y_arr = np.asarray(y_train)
    if strategy in {"none", "class_weight", "scale_pos_weight", ""}:
        return X_train, y_arr
    if strategy == "undersample":
        sampler = RandomUnderSampler(random_state=random_state)
    elif strategy == "smote":
        if X_train.isna().any().any():
            raise ValueError(
                "SMOTE does not accept NaN values. Use the linear feature group "
                "or impute the training matrix before strategy='smote'."
            )
        sampler = SMOTE(random_state=random_state)
    else:
        raise ValueError("strategy must be one of: none, class_weight, undersample, smote")
    X_res, y_res = sampler.fit_resample(X_train, y_arr)
    return pd.DataFrame(X_res, columns=X_train.columns), y_res
