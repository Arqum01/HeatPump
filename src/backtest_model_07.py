"""Walk-forward validation for dual-target heat pump models.

Concepts:
- Rolling-origin evaluation for realistic temporal generalization.
- Per-slice diagnostics by system and capacity to expose heterogeneity.
- Energy and COP aggregate checks alongside pointwise error metrics.
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBClassifier, XGBRegressor


INPUT_PATH = "data/processed/daikin_clean.csv"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_ELEC = "heatpump_elec"
TARGET_HEAT = "heatpump_heat"


FEATURE_POLICY_MODE = os.getenv("FEATURE_POLICY_MODE", "enhanced_onestep").strip().lower()

if FEATURE_POLICY_MODE not in {"strict_production", "enhanced_onestep"}:
    raise ValueError("FEATURE_POLICY_MODE must be 'strict_production' or 'enhanced_onestep'")

BASE_COMMON_FEATURES = [
    "heatpump_outsideT",
    "outsideT_3h_avg",
    "hdh",
    "heatpump_roomT",
    "temp_deficit",
    "load_ratio",
    "capacity_kw",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "day_of_week",
    "is_heating_season",
]

ELEC_MEMORY_FEATURES = [
    "elec_lag1",
    "elec_lag24",
    "elec_lag168",
    "elec_lag1_pct",
    "elec_lag2",
    "elec_lag3",
    "elec_lag4",
    "elec_lag6",
]

THERMAL_STATE_FEATURES = [
    "flowT_lag1",
    "returnT_lag1",
    "deltaT_house_lag1",
    "deltaT_lift_lag1",
    "lift_per_kw_lag1",
    "flowrate_lag1",
]

TARGET_DERIVED_STATE_FEATURES = [
    "heating_on_lag1",
    "heating_on_lag24",
    "run_hours",
    "cop_lag1",
    "was_defrost_lag1",
]

HEAT_MEMORY_FEATURES = [
    "heat_lag1",
    "heat_lag24",
    "heat_lag168",
]


def model_hyperparams(label: str) -> dict:
    """Return baseline XGBoost hyperparameters for a given target.

    Args:
        label: Target name, expected ``electricity`` or ``heat``.

    Returns:
        dict: Hyperparameter mapping passed to ``XGBRegressor``.
    """
    if label == "electricity":
        return {
            "n_estimators": 2200,
            "learning_rate": 0.02,
            "max_depth": 7,
            "subsample": 0.8,
            "colsample_bytree": 0.75,
            "min_child_weight": 8,
            "reg_alpha": 0.2,
            "reg_lambda": 2.0,
            "gamma": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "early_stopping_rounds": 120,
            "eval_metric": "mae",
        }
    if label == "heat":
        return {
            "n_estimators": 2600,
            "learning_rate": 0.015,
            "max_depth": 8,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "min_child_weight": 6,
            "reg_alpha": 0.1,
            "reg_lambda": 1.8,
            "gamma": 0.05,
            "random_state": 42,
            "n_jobs": -1,
            "early_stopping_rounds": 150,
            "eval_metric": "mae",
        }
    raise ValueError("label must be 'electricity' or 'heat'")


def build_feature_lists(df: pd.DataFrame):
    """Build electricity and heat feature lists from policy and metadata columns.

    Args:
        df: Input DataFrame used to discover available metadata features.

    Returns:
        tuple[list[str], list[str]]: Ordered electricity and heat feature names.
    """
    meta_cols = []
    for c in df.columns:
        if c.startswith("system_") and pd.api.types.is_numeric_dtype(df[c]):
            meta_cols.append(c)
    meta_cols = sorted(meta_cols)

    if FEATURE_POLICY_MODE == "strict_production":
        elec_features = BASE_COMMON_FEATURES + meta_cols
        heat_features = BASE_COMMON_FEATURES + meta_cols
    elif FEATURE_POLICY_MODE == "enhanced_onestep":
        elec_features = (
            BASE_COMMON_FEATURES
            + ELEC_MEMORY_FEATURES
            + THERMAL_STATE_FEATURES
            + TARGET_DERIVED_STATE_FEATURES
            + meta_cols
        )
        heat_features = (
            BASE_COMMON_FEATURES
            + ELEC_MEMORY_FEATURES
            + THERMAL_STATE_FEATURES
            + TARGET_DERIVED_STATE_FEATURES
            + HEAT_MEMORY_FEATURES
            + meta_cols
        )
    else:
        raise ValueError("invalid FEATURE_POLICY_MODE")

    elec_features = list(dict.fromkeys(elec_features))
    heat_features = list(dict.fromkeys(heat_features))
    return elec_features, heat_features


def load_data():
    """Load cleaned data and validate required modeling columns.

    Returns:
        tuple[pd.DataFrame, list[str], list[str]]: Cleaned dataset plus
        electricity/heat feature lists.

    Raises:
        ValueError: If any required columns are missing.
    """
    df = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])
    df = df.sort_values(["timestamp", "series_id"]).reset_index(drop=True)
    elec_features, heat_features = build_feature_lists(df)

    required = (
        ["timestamp", "series_id", TARGET_ELEC, TARGET_HEAT, "heating_on", "capacity_kw"]
        + elec_features
        + heat_features
    )
    required = list(dict.fromkeys(required))
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=[TARGET_ELEC, TARGET_HEAT]).copy()
    return df, elec_features, heat_features


def apply_cop_guardrail(pred_elec_w, pred_heat_w, cop_min=0.0, cop_max=8.0):
    """Constrain predictions to physically plausible non-negative COP bounds.

    Args:
        pred_elec_w: Electricity predictions in watts.
        pred_heat_w: Heat predictions in watts.
        cop_min: Minimum allowed COP.
        cop_max: Maximum allowed COP.

    Returns:
        tuple[np.ndarray, np.ndarray, int]: Corrected electricity predictions,
        corrected heat predictions, and number of adjusted rows.
    """
    elec_nonneg = np.maximum(pred_elec_w.astype(float), 0.0)
    heat_nonneg = np.maximum(pred_heat_w.astype(float), 0.0)

    eps = 1e-6
    elec_kwh = elec_nonneg / 1000.0
    heat_kwh = heat_nonneg / 1000.0

    valid_elec = elec_kwh > eps
    min_heat_kwh = np.zeros_like(heat_kwh)
    max_heat_kwh = np.zeros_like(heat_kwh)
    max_heat_kwh[valid_elec] = elec_kwh[valid_elec] * cop_max
    min_heat_kwh[valid_elec] = elec_kwh[valid_elec] * cop_min

    heat_kwh_clipped = np.clip(heat_kwh, min_heat_kwh, max_heat_kwh)
    heat_w_clipped = heat_kwh_clipped * 1000.0
    adjusted_count = int(np.count_nonzero(np.abs(heat_w_clipped - heat_nonneg) > 1e-6))
    return elec_nonneg, heat_w_clipped, adjusted_count


def train_runtime_classifier(X_train, X_test, y_train_on, y_test_on):
    """Train ON/OFF runtime classifier for a fold.

    Args:
        X_train: Training feature matrix.
        X_test: Testing feature matrix.
        y_train_on: Binary train ON labels.
        y_test_on: Binary test ON labels.

    Returns:
        tuple[XGBClassifier, np.ndarray]: Fitted classifier and hard predictions.
    """
    clf = XGBClassifier(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.5,
        gamma=0.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )
    clf.fit(X_train, y_train_on, eval_set=[(X_test, y_test_on)], verbose=False)
    on_proba = clf.predict_proba(X_test)[:, 1]
    on_pred = (on_proba >= 0.5).astype(int)
    return clf, on_pred


def train_single_target(X_train, X_test, y_train, y_test, label, X_eval=None, y_eval=None):
    """Train one target regressor and score fold test rows.

    Args:
        X_train: Training feature matrix.
        X_test: Testing feature matrix.
        y_train: Training target vector.
        y_test: Testing target vector.
        label: Target label used for parameter selection.
        X_eval: Optional eval matrix for early stopping.
        y_eval: Optional eval target for early stopping.

    Returns:
        tuple[XGBRegressor, np.ndarray]: Trained model and test predictions.
    """
    model = XGBRegressor(**model_hyperparams(label))
    eval_X = X_test if X_eval is None else X_eval
    eval_y = y_test if y_eval is None else y_eval
    model.fit(X_train, y_train, eval_set=[(eval_X, eval_y)], verbose=False)
    preds = model.predict(X_test)
    return model, preds


def evaluate_fold(test_df, pred_elec_w, pred_heat_w, on_pred):
    """Evaluate one walk-forward fold with pointwise and aggregate metrics.

    Args:
        test_df: Fold test DataFrame containing true targets.
        pred_elec_w: Electricity predictions.
        pred_heat_w: Heat predictions.
        on_pred: Runtime ON/OFF predictions.

    Returns:
        dict: Fold-level metrics including MAE, R2, energy, and COP errors.
    """
    y_elec_test = test_df[TARGET_ELEC].astype(float)
    y_heat_test = test_df[TARGET_HEAT].astype(float)

    off_mask = on_pred == 0
    pred_elec_w = pred_elec_w.astype(float).copy()
    pred_heat_w = pred_heat_w.astype(float).copy()

    pred_elec_w[off_mask] = 50.0
    pred_heat_w[off_mask] = 0.0

    pred_elec_corr_w, pred_heat_corr_w, cop_adjusted_count = apply_cop_guardrail(
        pred_elec_w, pred_heat_w, cop_min=0.0, cop_max=8.0
    )

    true_elec_kwh = y_elec_test.sum() / 1000.0
    pred_elec_kwh = pred_elec_corr_w.sum() / 1000.0
    true_heat_kwh = y_heat_test.sum() / 1000.0
    pred_heat_kwh = pred_heat_corr_w.sum() / 1000.0

    eps = 1e-6
    return {
        "r2_elec": float(r2_score(y_elec_test, pred_elec_corr_w)),
        "r2_heat": float(r2_score(y_heat_test, pred_heat_corr_w)),
        "mae_elec_w": float(mean_absolute_error(y_elec_test, pred_elec_corr_w)),
        "mae_heat_w": float(mean_absolute_error(y_heat_test, pred_heat_corr_w)),
        "energy_err_pct": float(abs(pred_elec_kwh - true_elec_kwh) / max(true_elec_kwh, eps) * 100),
        "heat_err_pct": float(abs(pred_heat_kwh - true_heat_kwh) / max(true_heat_kwh, eps) * 100),
        "true_cop": float(true_heat_kwh / max(true_elec_kwh, eps)),
        "pred_cop": float(pred_heat_kwh / max(pred_elec_kwh, eps)),
        "cop_err_pct": float(
            abs((pred_heat_kwh / max(pred_elec_kwh, eps)) - (true_heat_kwh / max(true_elec_kwh, eps)))
            / max(true_heat_kwh / max(true_elec_kwh, eps), eps)
            * 100
        ),
        "runtime_false_negative_count": int(((test_df["heating_on"].astype(int) == 1) & (on_pred == 0)).sum()),
        "cop_guardrail_adjusted_count": int(cop_adjusted_count),
        "rows": int(len(test_df)),
    }


def slice_metrics(test_df, pred_elec_w, pred_heat_w):
    """Compute slice-level diagnostics by capacity and system.

    Args:
        test_df: Fold test DataFrame with slice keys.
        pred_elec_w: Electricity predictions.
        pred_heat_w: Heat predictions.

    Returns:
        pd.DataFrame: Slice metrics table for both capacity and system slices.
    """
    out = []
    temp = test_df.copy()
    temp["pred_heatpump_elec"] = pred_elec_w
    temp["pred_heatpump_heat"] = pred_heat_w

    for capacity_kw, g in temp.groupby("capacity_kw"):
        out.append({
            "slice_type": "capacity_kw",
            "slice_value": str(capacity_kw),
            "rows": int(len(g)),
            "r2_elec": float(r2_score(g[TARGET_ELEC], g["pred_heatpump_elec"])) if len(g) > 1 else np.nan,
            "r2_heat": float(r2_score(g[TARGET_HEAT], g["pred_heatpump_heat"])) if len(g) > 1 else np.nan,
            "mae_elec_w": float(mean_absolute_error(g[TARGET_ELEC], g["pred_heatpump_elec"])),
            "mae_heat_w": float(mean_absolute_error(g[TARGET_HEAT], g["pred_heatpump_heat"])),
        })

    for series_id, g in temp.groupby("series_id"):
        out.append({
            "slice_type": "series_id",
            "slice_value": str(series_id),
            "rows": int(len(g)),
            "r2_elec": float(r2_score(g[TARGET_ELEC], g["pred_heatpump_elec"])) if len(g) > 1 else np.nan,
            "r2_heat": float(r2_score(g[TARGET_HEAT], g["pred_heatpump_heat"])) if len(g) > 1 else np.nan,
            "mae_elec_w": float(mean_absolute_error(g[TARGET_ELEC], g["pred_heatpump_elec"])),
            "mae_heat_w": float(mean_absolute_error(g[TARGET_HEAT], g["pred_heatpump_heat"])),
        })

    return pd.DataFrame(out)


def make_walk_forward_folds(df, min_train_ratio=0.50, test_ratio=0.10, step_ratio=0.10):
    """Construct rolling-origin folds with expanding train windows.

    Args:
        df: Input DataFrame with timestamp column.
        min_train_ratio: Initial train window as a fraction of unique timestamps.
        test_ratio: Test window size as a fraction of unique timestamps.
        step_ratio: Step size between fold boundaries as a fraction.

    Returns:
        list[tuple[int, pd.DataFrame, pd.DataFrame]]: Fold id with train/test partitions.
    """
    unique_ts = pd.Index(df["timestamp"].sort_values().unique())
    n = len(unique_ts)

    train_size = max(1, int(n * min_train_ratio))
    test_size = max(1, int(n * test_ratio))
    step_size = max(1, int(n * step_ratio))

    folds = []
    fold_id = 1
    train_end = train_size

    while train_end + test_size <= n:
        train_end_ts = unique_ts[train_end]
        test_end_idx = min(train_end + test_size, n)
        test_end_ts = unique_ts[test_end_idx - 1]

        train_df = df[df["timestamp"] < train_end_ts].copy()
        test_df = df[(df["timestamp"] >= train_end_ts) & (df["timestamp"] <= test_end_ts)].copy()

        if len(train_df) > 0 and len(test_df) > 0:
            folds.append((fold_id, train_df, test_df))

        fold_id += 1
        train_end += step_size

    return folds


def run_backtest():
    """Run full walk-forward backtest and persist fold/slice outputs.

    Returns:
        None. Writes fold metrics, slice metrics, and summary artifacts to disk.
    """
    df, elec_features, heat_features = load_data()
    folds = make_walk_forward_folds(df)

    fold_rows = []
    slice_rows = []

    for fold_id, train_df, test_df in folds:
        X_elec_train = train_df[elec_features]
        X_elec_test = test_df[elec_features]
        X_heat_train = train_df[heat_features]
        X_heat_test = test_df[heat_features]

        y_elec_train = train_df[TARGET_ELEC].astype(float)
        y_heat_train = train_df[TARGET_HEAT].astype(float)
        y_on_train = train_df["heating_on"].astype(int)
        y_on_test = test_df["heating_on"].astype(int)

        on_train_mask = y_on_train == 1
        on_test_mask = y_on_test == 1

        elec_imputer = SimpleImputer(strategy="median")
        heat_imputer = SimpleImputer(strategy="median")

        X_elec_train_imp = elec_imputer.fit_transform(X_elec_train)
        X_elec_test_imp = elec_imputer.transform(X_elec_test)
        X_heat_train_imp = heat_imputer.fit_transform(X_heat_train)
        X_heat_test_imp = heat_imputer.transform(X_heat_test)

        elec_model, pred_elec_w = train_single_target(
            X_elec_train_imp[on_train_mask.to_numpy()],
            X_elec_test_imp,
            y_elec_train[on_train_mask],
            test_df[TARGET_ELEC].astype(float),
            "electricity",
            X_eval=X_elec_test_imp[on_test_mask.to_numpy()],
            y_eval=test_df[TARGET_ELEC].astype(float)[on_test_mask],
        )

        heat_model, pred_heat_w = train_single_target(
            X_heat_train_imp[on_train_mask.to_numpy()],
            X_heat_test_imp,
            y_heat_train[on_train_mask],
            test_df[TARGET_HEAT].astype(float),
            "heat",
            X_eval=X_heat_test_imp[on_test_mask.to_numpy()],
            y_eval=test_df[TARGET_HEAT].astype(float)[on_test_mask],
        )

        _, on_pred = train_runtime_classifier(
            X_elec_train_imp,
            X_elec_test_imp,
            y_on_train,
            y_on_test,
        )

        metrics = evaluate_fold(test_df, pred_elec_w, pred_heat_w, on_pred)
        metrics["fold_id"] = fold_id
        metrics["train_start"] = str(train_df["timestamp"].min())
        metrics["train_end"] = str(train_df["timestamp"].max())
        metrics["test_start"] = str(test_df["timestamp"].min())
        metrics["test_end"] = str(test_df["timestamp"].max())
        fold_rows.append(metrics)

        temp_pred_elec = np.maximum(pred_elec_w.astype(float), 0.0)
        temp_pred_heat = np.maximum(pred_heat_w.astype(float), 0.0)
        fold_slice = slice_metrics(test_df, temp_pred_elec, temp_pred_heat)
        fold_slice["fold_id"] = fold_id
        slice_rows.append(fold_slice)

    fold_df = pd.DataFrame(fold_rows)
    slice_df = pd.concat(slice_rows, ignore_index=True) if slice_rows else pd.DataFrame()

    summary = {
        "generated_at": datetime.now().isoformat(),
        "feature_policy_mode": FEATURE_POLICY_MODE,
        "fold_count": int(len(fold_df)),
        "avg_r2_elec": float(fold_df["r2_elec"].mean()),
        "avg_r2_heat": float(fold_df["r2_heat"].mean()),
        "avg_mae_elec_w": float(fold_df["mae_elec_w"].mean()),
        "avg_mae_heat_w": float(fold_df["mae_heat_w"].mean()),
        "avg_energy_err_pct": float(fold_df["energy_err_pct"].mean()),
        "avg_heat_err_pct": float(fold_df["heat_err_pct"].mean()),
        "avg_cop_err_pct": float(fold_df["cop_err_pct"].mean()),
        "max_runtime_false_negative_count": int(fold_df["runtime_false_negative_count"].max()),
    }

    fold_path = f"{OUTPUT_DIR}/walk_forward_folds.csv"
    slice_path = f"{OUTPUT_DIR}/walk_forward_slices.csv"
    summary_path = f"{OUTPUT_DIR}/walk_forward_summary.json"

    fold_df.to_csv(fold_path, index=False)
    slice_df.to_csv(slice_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(fold_df)
    print(json.dumps(summary, indent=2))
    print(f"Saved: {fold_path}")
    print(f"Saved: {slice_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    run_backtest()

