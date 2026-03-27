"""Training stage for dual-target heat pump forecasting.

Strategy stack:
- Target-specific walk-forward tuning for electricity and heat.
- Multi-strategy target formulation search (raw, log, ratio-based).
- Slice-aware calibration for system and capacity heterogeneity.
- Guardrailed post-processing for physically plausible outputs.
"""

import json
import os
from datetime import datetime
import joblib

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, r2_score
from xgboost import XGBClassifier, XGBRegressor

INPUT_PATH = "data/processed/daikin_clean.csv"
MODEL_DIR = "models"
REPORT_PATH = f"{MODEL_DIR}/training_report.txt"

RUNTIME_MODEL_PATH_TEMPLATE = f"{MODEL_DIR}/xgb_runtime_clf_v{{run_tag}}.joblib"
ELEC_IMPUTER_PATH_TEMPLATE = f"{MODEL_DIR}/imputer_elec_v{{run_tag}}.joblib"
HEAT_IMPUTER_PATH_TEMPLATE = f"{MODEL_DIR}/imputer_heat_v{{run_tag}}.joblib"
PIPELINE_META_PATH_TEMPLATE = f"{MODEL_DIR}/pipeline_meta_v{{run_tag}}.json"
FEATURE_SCHEMA_PATH_TEMPLATE = f"{MODEL_DIR}/feature_schema_v{{run_tag}}.json"
SLICE_CALIBRATOR_PATH_TEMPLATE = f"{MODEL_DIR}/slice_calibrators_v{{run_tag}}.json"



TARGET_ELEC = "heatpump_elec"
TARGET_HEAT = "heatpump_heat"

FEATURE_POLICY_MODE = os.getenv("FEATURE_POLICY_MODE", "enhanced_onestep").strip().lower()
TARGET_STRATEGY_CANDIDATES = ["none", "log1p_both", "cop_ratio_heat"]
WALK_FORWARD_SPLITS = 3
ENABLE_WALK_FORWARD_TUNING = True
ENABLE_SLICE_CALIBRATION = True
# allowed values:
# "strict_production"  -> only features known safely at prediction time
# "enhanced_onestep"   -> allows lagged observed telemetry from previous hours

if FEATURE_POLICY_MODE not in {"strict_production", "enhanced_onestep"}:
    raise ValueError(
        "FEATURE_POLICY_MODE must be 'strict_production' or 'enhanced_onestep'"
    )

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

os.makedirs(MODEL_DIR, exist_ok=True)


def parse_train_system_ids() -> list[int] | None:
    """Parse optional training system selection from environment.

    Supported variables (first non-empty wins):
    - TRAIN_SYSTEM_IDS
    - SYSTEM_IDS
    """
    raw = os.getenv("TRAIN_SYSTEM_IDS", "").strip()
    if not raw:
        raw = os.getenv("SYSTEM_IDS", "").strip()
    if not raw:
        return None

    ids = []
    seen = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            sid = int(token)
        except ValueError:
            continue
        if sid not in seen:
            seen.add(sid)
            ids.append(sid)

    return ids if ids else None


def model_hyperparams(label: str) -> dict:
    """Return baseline XGBoost hyperparameters for a target label.

    Args:
        label: Target identifier, either ``electricity`` or ``heat``.

    Returns:
        dict: Parameter dictionary suitable for ``XGBRegressor``.
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


def load_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load cleaned data, validate required columns, and build feature lists.

    Returns:
        tuple[pd.DataFrame, list[str], list[str]]: Loaded DataFrame, electricity
        feature list, and heat feature list.

    Raises:
        ValueError: If required serving/training columns are missing.
    """
    df = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])
    df = df.sort_values(["timestamp", "system_id"]).reset_index(drop=True)

    selected_systems = parse_train_system_ids()
    if selected_systems:
        df = df[df["system_id"].isin(selected_systems)].copy()
        if df.empty:
            raise ValueError(
                f"No rows found for selected training systems: {selected_systems}. "
                "Check TRAIN_SYSTEM_IDS/SYSTEM_IDS or regenerate data."
            )
        print(f"Training system filter applied: {selected_systems}")

    elec_features, heat_features = build_feature_lists(df)

    required = (
        ["timestamp", "system_id", TARGET_ELEC, TARGET_HEAT, "heating_on"]
        + elec_features
        + heat_features
    )
    required = list(dict.fromkeys(required))

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns for feature policy '{FEATURE_POLICY_MODE}': {missing}"
        )

    before = len(df)
    df = df.dropna(subset=[TARGET_ELEC, TARGET_HEAT]).copy()

    print(f"Loaded rows: {before} -> {len(df)} after label checks")
    print(f"Systems: {sorted(df['system_id'].unique())}")
    print(f"Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"Feature policy mode: {FEATURE_POLICY_MODE}")
    print(f"Electricity features: {len(elec_features)}")
    print(f"Heat features: {len(heat_features)}")

    return df, elec_features, heat_features

def build_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Build target-specific feature sets according to policy mode.

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
        raise ValueError(
            "FEATURE_POLICY_MODE must be 'strict_production' or 'enhanced_onestep'"
        )

    elec_features = list(dict.fromkeys(elec_features))
    heat_features = list(dict.fromkeys(heat_features))
    return elec_features, heat_features

def build_feature_schema(
    df: pd.DataFrame,
    elec_features: list[str],
    heat_features: list[str],
) -> dict:
    """Create a serving schema capturing feature order and expected dtypes.

    Args:
        df: Training DataFrame used for dtype inference.
        elec_features: Ordered electricity feature list.
        heat_features: Ordered heat feature list.

    Returns:
        dict: Schema dictionary used by serving validation.
    """
    union_features = list(dict.fromkeys(elec_features + heat_features))

    def infer_dtype(series: pd.Series) -> str:
        """Map pandas dtype families into compact schema labels."""
        if pd.api.types.is_integer_dtype(series):
            return "int"
        if pd.api.types.is_float_dtype(series):
            return "float"
        if pd.api.types.is_bool_dtype(series):
            return "bool"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        return "string"

    feature_dtypes = {}
    for col in union_features:
        feature_dtypes[col] = infer_dtype(df[col])

    schema = {
        "timestamp_column": "timestamp",
        "id_column": "system_id",
        "targets": [TARGET_ELEC, TARGET_HEAT],
        "electricity_features_ordered": elec_features,
        "heat_features_ordered": heat_features,
        "all_serving_features_ordered": union_features,
        "feature_dtypes": feature_dtypes,
        "required_serving_columns": union_features,
    }
    return schema

def validate_inference_frame(
    df: pd.DataFrame,
    feature_schema: dict,
    allow_extra_columns: bool = True,
) -> pd.DataFrame:
    """Validate and coerce an inference frame against the saved schema.

    Args:
        df: Candidate inference DataFrame.
        feature_schema: Schema emitted during training.
        allow_extra_columns: Whether non-required columns are permitted.

    Returns:
        dict: Validated full frame plus target-specific feature views.
    """
    required_cols = feature_schema["required_serving_columns"]
    dtype_map = feature_schema["feature_dtypes"]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required inference columns: {missing}")

    extra = [c for c in df.columns if c not in required_cols]
    if extra and not allow_extra_columns:
        raise ValueError(f"Unexpected extra inference columns: {extra}")

    validated = df.copy()

    for col in required_cols:
        expected = dtype_map.get(col, "float")

        if expected in {"float", "int", "bool"}:
            validated[col] = pd.to_numeric(validated[col], errors="coerce")
            if expected == "int":
                validated[col] = validated[col].astype("Int64")
            elif expected == "bool":
                validated[col] = validated[col].astype("Int64")
            else:
                validated[col] = validated[col].astype(float)
        elif expected == "datetime":
            validated[col] = pd.to_datetime(validated[col], errors="coerce")
        else:
            validated[col] = validated[col].astype(str)

    elec_view = validated[feature_schema["electricity_features_ordered"]].copy()
    heat_view = validated[feature_schema["heat_features_ordered"]].copy()

    return {
        "validated_frame": validated,
        "electricity_frame": elec_view,
        "heat_frame": heat_view,
        "missing_columns": missing,
        "extra_columns": extra,
    }


def time_split(df: pd.DataFrame, train_ratio: float = 0.8):
    """Split data chronologically into train and test partitions.

    Args:
        df: Input DataFrame with timestamp column.
        train_ratio: Fraction of unique timestamps assigned to training.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Train and test DataFrames.
    """
    unique_ts = pd.Index(df["timestamp"].sort_values().unique())

    split_idx = int(len(unique_ts) * train_ratio)
    split_idx = max(1, min(split_idx, len(unique_ts) - 1))
    split_ts = unique_ts[split_idx]

    train_df = df[df["timestamp"] < split_ts].copy()
    test_df = df[df["timestamp"] >= split_ts].copy()

    print(f"Time split at timestamp boundary: {split_ts}")
    print(f"Unique timestamps total: {len(unique_ts)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Train max timestamp: {train_df['timestamp'].max()}")
    print(f"Test min timestamp: {test_df['timestamp'].min()}")

    return train_df, test_df

def validate_split_integrity(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Verify there is no temporal overlap between train and test splits.

    Args:
        train_df: Training partition.
        test_df: Testing partition.

    Returns:
        dict: Diagnostics summary including overlap checks.

    Raises:
        ValueError: If overlap or ordering violations are detected.
    """
    train_max_ts = train_df["timestamp"].max()
    test_min_ts = test_df["timestamp"].min()

    train_ts = set(train_df["timestamp"].unique())
    test_ts = set(test_df["timestamp"].unique())
    overlap_ts = sorted(train_ts.intersection(test_ts))

    is_valid = (train_max_ts < test_min_ts) and (len(overlap_ts) == 0)

    diagnostics = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_max_timestamp": str(train_max_ts),
        "test_min_timestamp": str(test_min_ts),
        "overlap_timestamp_count": int(len(overlap_ts)),
        "overlap_sample": [str(x) for x in overlap_ts[:5]],
        "valid_time_split": bool(is_valid),
    }

    print("\n=== SPLIT INTEGRITY CHECK ===")
    print(f"Train max timestamp: {diagnostics['train_max_timestamp']}")
    print(f"Test min timestamp:  {diagnostics['test_min_timestamp']}")
    print(f"Overlap timestamp count: {diagnostics['overlap_timestamp_count']}")
    print(f"Valid time split: {diagnostics['valid_time_split']}")
    print("=== END SPLIT CHECK ===\n")

    if not is_valid:
        raise ValueError(
            "Train/test split integrity failed: overlapping timestamps detected."
        )

    return diagnostics



def impute_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fit median imputation on training features and apply to both splits.

    Args:
        X_train: Training feature matrix.
        X_test: Testing feature matrix.

    Returns:
        tuple[np.ndarray, np.ndarray, SimpleImputer]: Imputed train matrix,
        imputed test matrix, and fitted imputer.
    """
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    return X_train_imp, X_test_imp, imputer


def tuning_candidates(label: str) -> list[dict]:
    """Generate a compact hyperparameter candidate set for walk-forward tuning.

    Args:
        label: Target identifier used to seed baseline parameters.

    Returns:
        list[dict]: Candidate parameter dictionaries.
    """
    base = model_hyperparams(label)
    candidates = [base]

    candidates.append({
        **base,
        "learning_rate": base["learning_rate"] * 0.75,
        "n_estimators": int(base["n_estimators"] * 1.2),
    })
    candidates.append({
        **base,
        "learning_rate": base["learning_rate"] * 1.25,
        "n_estimators": int(base["n_estimators"] * 0.85),
    })
    candidates.append({
        **base,
        "max_depth": max(4, base["max_depth"] - 1),
        "min_child_weight": base["min_child_weight"] + 2,
    })
    candidates.append({
        **base,
        "max_depth": base["max_depth"] + 1,
        "min_child_weight": max(1, base["min_child_weight"] - 2),
    })

    return candidates


def walk_forward_tune_target(
    X_train: np.ndarray,
    y_train: pd.Series,
    label: str,
) -> tuple[dict, list[dict]]:
    """Select target hyperparameters using forward-chaining validation.

    Args:
        X_train: Training feature matrix for one target.
        y_train: Target vector for one target.
        label: Target name used for baseline parameter lookup.

    Returns:
        tuple[dict, list[dict]]: Best parameter dictionary and per-candidate
        evaluation rows.
    """
    # Tune with forward-chaining splits to mimic production chronology.
    y_values = np.asarray(y_train, dtype=float)
    if not ENABLE_WALK_FORWARD_TUNING or len(y_values) < 500:
        return model_hyperparams(label), []

    splitter = TimeSeriesSplit(n_splits=WALK_FORWARD_SPLITS)
    candidates = tuning_candidates(label)
    best_params = candidates[0]
    best_mae = float("inf")
    rows = []

    print(f"\nWalk-forward tuning for {label} ({len(candidates)} candidates)...")

    for idx, params in enumerate(candidates, start=1):
        fold_maes = []
        for train_idx, val_idx in splitter.split(X_train):
            X_tr = X_train[train_idx]
            y_tr = y_values[train_idx]
            X_val = X_train[val_idx]
            y_val = y_values[val_idx]

            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            pred = model.predict(X_val)
            fold_maes.append(mean_absolute_error(y_val, pred))

        avg_mae = float(np.mean(fold_maes))
        rows.append(
            {
                "label": label,
                "candidate": idx,
                "avg_mae_w": avg_mae,
                "params": params,
            }
        )
        print(f"  candidate {idx}: avg_mae={avg_mae:.2f}W")

        if avg_mae < best_mae:
            best_mae = avg_mae
            best_params = params

    print(f"Selected {label} params with avg_mae={best_mae:.2f}W")
    return best_params, rows


def strategy_train_predict(
    strategy: str,
    X_elec_train_on: np.ndarray,
    X_elec_test: np.ndarray,
    X_heat_train_on: np.ndarray,
    X_heat_test: np.ndarray,
    y_elec_train_on: pd.Series,
    y_heat_train_on: pd.Series,
    y_elec_test: pd.Series,
    y_heat_test: pd.Series,
    X_elec_eval_on: np.ndarray,
    X_heat_eval_on: np.ndarray,
    y_elec_eval_on: pd.Series,
    y_heat_eval_on: pd.Series,
    elec_params: dict,
    heat_params: dict,
) -> dict:
    """Train and score one target-transformation strategy.

    Args:
        strategy: Strategy key (``none``, ``log1p_both``, ``cop_ratio_heat``).
        X_elec_train_on: ON-state electricity train features.
        X_elec_test: Electricity test features.
        X_heat_train_on: ON-state heat train features.
        X_heat_test: Heat test features.
        y_elec_train_on: ON-state electricity train target.
        y_heat_train_on: ON-state heat train target.
        y_elec_test: Electricity test target.
        y_heat_test: Heat test target.
        X_elec_eval_on: ON-state electricity eval features.
        X_heat_eval_on: ON-state heat eval features.
        y_elec_eval_on: ON-state electricity eval target.
        y_heat_eval_on: ON-state heat eval target.
        elec_params: Tuned electricity model parameters.
        heat_params: Tuned heat model parameters.

    Returns:
        dict: Trained models and train/test predictions for the strategy.
    """
    # Compare alternative target parameterizations under one evaluation path.
    eps = 1e-6

    if strategy == "none":
        elec_model, pred_elec_test = train_single_target(
            X_elec_train_on,
            X_elec_test,
            y_elec_train_on,
            y_elec_test,
            "electricity",
            X_eval=X_elec_eval_on,
            y_eval=y_elec_eval_on,
            hyperparams=elec_params,
        )
        heat_model, pred_heat_test = train_single_target(
            X_heat_train_on,
            X_heat_test,
            y_heat_train_on,
            y_heat_test,
            "heat",
            X_eval=X_heat_eval_on,
            y_eval=y_heat_eval_on,
            hyperparams=heat_params,
        )

        pred_elec_train = elec_model.predict(X_elec_train_on)
        pred_heat_train = heat_model.predict(X_heat_train_on)

    elif strategy == "log1p_both":
        elec_model, pred_elec_test_log = train_single_target(
            X_elec_train_on,
            X_elec_test,
            np.log1p(np.maximum(y_elec_train_on, 0.0)),
            np.log1p(np.maximum(y_elec_test, 0.0)),
            "electricity",
            X_eval=X_elec_eval_on,
            y_eval=np.log1p(np.maximum(y_elec_eval_on, 0.0)),
            hyperparams=elec_params,
        )
        heat_model, pred_heat_test_log = train_single_target(
            X_heat_train_on,
            X_heat_test,
            np.log1p(np.maximum(y_heat_train_on, 0.0)),
            np.log1p(np.maximum(y_heat_test, 0.0)),
            "heat",
            X_eval=X_heat_eval_on,
            y_eval=np.log1p(np.maximum(y_heat_eval_on, 0.0)),
            hyperparams=heat_params,
        )

        pred_elec_test = np.expm1(pred_elec_test_log)
        pred_heat_test = np.expm1(pred_heat_test_log)
        pred_elec_train = np.expm1(elec_model.predict(X_elec_train_on))
        pred_heat_train = np.expm1(heat_model.predict(X_heat_train_on))

    elif strategy == "cop_ratio_heat":
        elec_model, pred_elec_test = train_single_target(
            X_elec_train_on,
            X_elec_test,
            y_elec_train_on,
            y_elec_test,
            "electricity",
            X_eval=X_elec_eval_on,
            y_eval=y_elec_eval_on,
            hyperparams=elec_params,
        )

        ratio_train = (y_heat_train_on / np.maximum(y_elec_train_on, eps)).astype(float)
        ratio_eval = (y_heat_eval_on / np.maximum(y_elec_eval_on, eps)).astype(float)
        heat_model, pred_ratio_test = train_single_target(
            X_heat_train_on,
            X_heat_test,
            ratio_train,
            ratio_eval,
            "heat",
            X_eval=X_heat_eval_on,
            y_eval=ratio_eval,
            hyperparams=heat_params,
        )

        pred_ratio_test = np.maximum(pred_ratio_test, 0.0)
        pred_elec_train = elec_model.predict(X_elec_train_on)
        pred_ratio_train = np.maximum(heat_model.predict(X_heat_train_on), 0.0)
        pred_heat_test = pred_ratio_test * np.maximum(pred_elec_test, eps)
        pred_heat_train = pred_ratio_train * np.maximum(pred_elec_train, eps)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return {
        "strategy": strategy,
        "elec_model": elec_model,
        "heat_model": heat_model,
        "pred_elec_test": np.maximum(pred_elec_test, 0.0),
        "pred_heat_test": np.maximum(pred_heat_test, 0.0),
        "pred_elec_train_on": np.maximum(pred_elec_train, 0.0),
        "pred_heat_train_on": np.maximum(pred_heat_train, 0.0),
    }


def build_slice_calibrators(
    train_on_df: pd.DataFrame,
    pred_elec_on: np.ndarray,
    pred_heat_on: np.ndarray,
) -> dict:
    """Estimate multiplicative calibration factors per system-capacity slice.

    Args:
        train_on_df: ON-state training rows with system metadata and targets.
        pred_elec_on: Electricity predictions on train-on rows.
        pred_heat_on: Heat predictions on train-on rows.

    Returns:
        dict: Mapping of ``system|capacity`` keys to multiplier parameters.
    """
    # Learn multiplicative per-slice correction from train-on predictions.
    calibrated = train_on_df[["system_id", "capacity_kw", TARGET_ELEC, TARGET_HEAT]].copy()
    calibrated["pred_elec"] = np.asarray(pred_elec_on, dtype=float)
    calibrated["pred_heat"] = np.asarray(pred_heat_on, dtype=float)

    calibrators = {}
    for (system_id, capacity_kw), grp in calibrated.groupby(["system_id", "capacity_kw"]):
        key = f"{int(system_id)}|{int(capacity_kw)}"
        pred_e_sum = max(float(grp["pred_elec"].sum()), 1e-6)
        pred_h_sum = max(float(grp["pred_heat"].sum()), 1e-6)
        true_e_sum = float(grp[TARGET_ELEC].sum())
        true_h_sum = float(grp[TARGET_HEAT].sum())

        elec_mult = np.clip(true_e_sum / pred_e_sum, 0.7, 1.3)
        heat_mult = np.clip(true_h_sum / pred_h_sum, 0.7, 1.3)
        calibrators[key] = {
            "system_id": int(system_id),
            "capacity_kw": int(capacity_kw),
            "elec_mult": float(elec_mult),
            "heat_mult": float(heat_mult),
        }

    return calibrators


def apply_slice_calibrators(
    df: pd.DataFrame,
    pred_elec_w: np.ndarray,
    pred_heat_w: np.ndarray,
    calibrators: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply per-slice calibration multipliers to predictions.

    Args:
        df: Scored rows containing ``system_id`` and ``capacity_kw``.
        pred_elec_w: Electricity predictions.
        pred_heat_w: Heat predictions.
        calibrators: Slice multiplier mapping.

    Returns:
        tuple[np.ndarray, np.ndarray]: Calibrated electricity and heat predictions.
    """
    # Apply slice multipliers at scoring time for systematic bias correction.
    elec = np.asarray(pred_elec_w, dtype=float).copy()
    heat = np.asarray(pred_heat_w, dtype=float).copy()

    for pos, row in enumerate(df[["system_id", "capacity_kw"]].itertuples(index=False, name=None)):
        system_id, capacity_kw = row
        key = f"{int(system_id)}|{int(capacity_kw)}"
        mult = calibrators.get(key)
        if mult is None:
            continue
        elec[pos] *= mult["elec_mult"]
        heat[pos] *= mult["heat_mult"]

    return elec, heat


def slice_error_analysis(
    df: pd.DataFrame,
    y_elec_true: pd.Series,
    y_heat_true: pd.Series,
    pred_elec_w: np.ndarray,
    pred_heat_w: np.ndarray,
) -> pd.DataFrame:
    """Compute slice-level performance metrics across system and capacity.

    Args:
        df: Test rows including slice keys.
        y_elec_true: True electricity target values.
        y_heat_true: True heat target values.
        pred_elec_w: Electricity predictions.
        pred_heat_w: Heat predictions.

    Returns:
        pd.DataFrame: Slice metrics sorted by worst heat/electricity MAE.
    """
    # Quantify residual error concentration across operational slices.
    scored = df[["system_id", "capacity_kw"]].copy()
    scored["true_elec"] = y_elec_true.to_numpy(dtype=float)
    scored["true_heat"] = y_heat_true.to_numpy(dtype=float)
    scored["pred_elec"] = np.asarray(pred_elec_w, dtype=float)
    scored["pred_heat"] = np.asarray(pred_heat_w, dtype=float)

    rows = []
    for (system_id, capacity_kw), grp in scored.groupby(["system_id", "capacity_kw"]):
        rows.append(
            {
                "system_id": int(system_id),
                "capacity_kw": int(capacity_kw),
                "rows": int(len(grp)),
                "mae_elec_w": float(mean_absolute_error(grp["true_elec"], grp["pred_elec"])),
                "mae_heat_w": float(mean_absolute_error(grp["true_heat"], grp["pred_heat"])),
                "r2_elec": float(r2_score(grp["true_elec"], grp["pred_elec"])),
                "r2_heat": float(r2_score(grp["true_heat"], grp["pred_heat"])),
            }
        )

    return pd.DataFrame(rows).sort_values(["mae_heat_w", "mae_elec_w"], ascending=False)

def save_pipeline_artifacts(
    run_tag: str,
    elec_imputer,
    heat_imputer,
    runtime_model,
    elec_features: list[str],
    heat_features: list[str],
    selected_strategy: str,
    slice_calibrators: dict,
):
    """Persist fitted preprocessors, runtime model, and metadata artifacts.

    Args:
        run_tag: Artifact version tag.
        elec_imputer: Fitted electricity imputer.
        heat_imputer: Fitted heat imputer.
        runtime_model: Trained ON/OFF classifier.
        elec_features: Electricity feature names.
        heat_features: Heat feature names.
        selected_strategy: Winning target transformation strategy.
        slice_calibrators: Learned per-slice calibration mapping.

    Returns:
        dict: Paths of persisted artifacts.
    """
    runtime_model_path = RUNTIME_MODEL_PATH_TEMPLATE.format(run_tag=run_tag)
    elec_imputer_path = ELEC_IMPUTER_PATH_TEMPLATE.format(run_tag=run_tag)
    heat_imputer_path = HEAT_IMPUTER_PATH_TEMPLATE.format(run_tag=run_tag)
    pipeline_meta_path = PIPELINE_META_PATH_TEMPLATE.format(run_tag=run_tag)
    slice_calibrator_path = SLICE_CALIBRATOR_PATH_TEMPLATE.format(run_tag=run_tag)

    joblib.dump(runtime_model, runtime_model_path)
    joblib.dump(elec_imputer, elec_imputer_path)
    joblib.dump(heat_imputer, heat_imputer_path)

    pipeline_meta = {
        "generated_at": datetime.now().isoformat(),
        "input_path": INPUT_PATH,
        "target_strategy": selected_strategy,
        "slice_calibrator_count": len(slice_calibrators),
        "feature_counts": {
            "electricity": len(elec_features),
            "heat": len(heat_features),
        },
        "features": {
            "electricity": elec_features,
            "heat": heat_features,
        },
        "artifacts": {
            "runtime_model": runtime_model_path,
            "electricity_imputer": elec_imputer_path,
            "heat_imputer": heat_imputer_path,
        },
    }

    with open(pipeline_meta_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_meta, f, indent=2)

    with open(slice_calibrator_path, "w", encoding="utf-8") as f:
        json.dump(slice_calibrators, f, indent=2)

    return {
        "runtime_model": runtime_model_path,
        "electricity_imputer": elec_imputer_path,
        "heat_imputer": heat_imputer_path,
        "pipeline_meta": pipeline_meta_path,
        "slice_calibrators": slice_calibrator_path,
        "target_strategy": selected_strategy,
    }



def train_single_target(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    label: str,
    X_eval: np.ndarray | None = None,
    y_eval: pd.Series | np.ndarray | None = None,
    hyperparams: dict | None = None,
):
    """Train a single XGBoost regressor and return test predictions.

    Args:
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        y_train: Training target.
        y_test: Test target.
        label: Target identifier.
        X_eval: Optional evaluation matrix for early stopping.
        y_eval: Optional evaluation target.
        hyperparams: Optional explicit parameter override.

    Returns:
        tuple[XGBRegressor, np.ndarray]: Fitted model and test predictions.
    """
    params = model_hyperparams(label) if hyperparams is None else hyperparams
    model = XGBRegressor(**params)
    print(f"\nTraining {label} model...")

    eval_X = X_test if X_eval is None else X_eval
    eval_y = y_test if y_eval is None else y_eval
    model.fit(X_train, y_train, eval_set=[(eval_X, eval_y)], verbose=100)
    preds = model.predict(X_test)
    return model, preds


def train_runtime_classifier(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_on: pd.Series,
    y_test_on: pd.Series,
):
    """Train ON/OFF runtime classifier used for standby gating.

    Args:
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        y_train_on: Binary train labels.
        y_test_on: Binary test labels.

    Returns:
        tuple[XGBClassifier, np.ndarray, np.ndarray]: Classifier, hard labels,
        and ON probabilities for the test set.
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
    print("\nTraining runtime ON/OFF classifier...")
    clf.fit(X_train, y_train_on, eval_set=[(X_test, y_test_on)], verbose=False)
    on_proba = clf.predict_proba(X_test)[:, 1]
    on_pred = (on_proba >= 0.5).astype(int)
    return clf, on_pred, on_proba


def classifier_diagnostics(y_true_on: pd.Series, y_pred_on: np.ndarray) -> dict:
    """Compute confusion-matrix and text report diagnostics for runtime labels.

    Args:
        y_true_on: Ground-truth ON/OFF labels.
        y_pred_on: Predicted ON/OFF labels.

    Returns:
        dict: Confusion matrix, classification report, and scalar counts.
    """
    labels = [0, 1]
    cm = confusion_matrix(y_true_on, y_pred_on, labels=labels)
    report = classification_report(
        y_true_on,
        y_pred_on,
        labels=labels,
        target_names=["OFF", "ON"],
        digits=4,
        zero_division=0,
    )

    tn, fp, fn, tp = cm.ravel()
    return {
        "confusion_matrix": cm,
        "classification_report": report,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def apply_cop_guardrail(
    pred_elec_w: np.ndarray,
    pred_heat_w: np.ndarray,
    cop_min: float = 0.0,
    cop_max: float = 8.0,
):
    """Constrain predictions to physically plausible non-negative COP bounds.

    Args:
        pred_elec_w: Electricity predictions in watts.
        pred_heat_w: Heat predictions in watts.
        cop_min: Minimum allowed COP bound.
        cop_max: Maximum allowed COP bound.

    Returns:
        tuple[np.ndarray, np.ndarray, int]: Corrected electricity predictions,
        corrected heat predictions, and number of adjusted rows.
    """
    # Enforce non-negative power first.
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


def evaluate_dual(
    y_elec_test: pd.Series,
    pred_elec_w: np.ndarray,
    y_heat_test: pd.Series,
    pred_heat_w: np.ndarray,
    on_pred: np.ndarray,
    y_on_true: pd.Series | np.ndarray | None = None,
) -> dict:
    """Evaluate dual-target predictions with pointwise and aggregate metrics.

    Args:
        y_elec_test: True electricity targets.
        pred_elec_w: Electricity predictions.
        y_heat_test: True heat targets.
        pred_heat_w: Heat predictions.
        on_pred: Runtime ON/OFF predictions used for standby forcing.
        y_on_true: Optional true ON/OFF labels used for false-negative counting.

    Returns:
        dict: Metrics including MAE, R2, energy error, COP error, and guardrail stats.
    """
    off_mask = on_pred == 0
    pred_elec_w = pred_elec_w.astype(float).copy()
    pred_heat_w = pred_heat_w.astype(float).copy()

    # Electricity in OFF state is treated as standby draw.
    pred_elec_w[off_mask] = 50.0
    pred_heat_w[off_mask] = 0.0

    pred_elec_corr_w, pred_heat_corr_w, cop_adjusted_count = apply_cop_guardrail(
        pred_elec_w,
        pred_heat_w,
        cop_min=0.0,
        cop_max=8.0,
    )

    mae_elec = mean_absolute_error(y_elec_test, pred_elec_corr_w)
    mae_heat = mean_absolute_error(y_heat_test, pred_heat_corr_w)
    r2_elec = r2_score(y_elec_test, pred_elec_corr_w)
    r2_heat = r2_score(y_heat_test, pred_heat_corr_w)

    true_elec_kwh = y_elec_test.sum() / 1000.0
    pred_elec_kwh = pred_elec_corr_w.sum() / 1000.0
    true_heat_kwh = y_heat_test.sum() / 1000.0
    pred_heat_kwh = pred_heat_corr_w.sum() / 1000.0

    energy_err_pct = abs(pred_elec_kwh - true_elec_kwh) / true_elec_kwh * 100
    heat_err_pct = abs(pred_heat_kwh - true_heat_kwh) / true_heat_kwh * 100

    eps = 1e-6
    true_cop = true_heat_kwh / max(true_elec_kwh, eps)
    pred_cop = pred_heat_kwh / max(pred_elec_kwh, eps)
    cop_err_pct = abs(pred_cop - true_cop) / max(true_cop, eps) * 100

    pred_elec_hour_kwh = pred_elec_corr_w / 1000.0
    pred_heat_hour_kwh = pred_heat_corr_w / 1000.0
    pred_cop_hourly = np.divide(
        pred_heat_hour_kwh,
        np.maximum(pred_elec_hour_kwh, eps),
        out=np.full_like(pred_heat_hour_kwh, np.nan),
        where=pred_elec_hour_kwh > eps,
    )

    valid_mask = np.isfinite(pred_cop_hourly)
    if valid_mask.any():
        physical_mask = (pred_cop_hourly[valid_mask] >= 0.0) & (pred_cop_hourly[valid_mask] <= 8.0)
        cop_physical_valid_pct = physical_mask.mean() * 100.0
        cop_physical_invalid_count = int((~physical_mask).sum())
    else:
        cop_physical_valid_pct = 100.0
        cop_physical_invalid_count = 0

    leakage_flag = (r2_elec > 0.97) or (r2_heat > 0.97)

    if y_on_true is None:
        runtime_false_negative_count = int(off_mask.sum())
    else:
        y_on_true_arr = np.asarray(y_on_true, dtype=int)
        runtime_false_negative_count = int(
            np.count_nonzero((y_on_true_arr == 1) & (on_pred == 0))
        )

    return {
        "r2_elec": float(r2_elec),
        "r2_heat": float(r2_heat),
        "mae_elec_w": float(mae_elec),
        "mae_heat_w": float(mae_heat),
        "true_elec_kwh": float(true_elec_kwh),
        "pred_elec_kwh": float(pred_elec_kwh),
        "true_heat_kwh": float(true_heat_kwh),
        "pred_heat_kwh": float(pred_heat_kwh),
        "energy_err_pct": float(energy_err_pct),
        "heat_err_pct": float(heat_err_pct),
        "true_cop": float(true_cop),
        "pred_cop": float(pred_cop),
        "cop_err_pct": float(cop_err_pct),
        "cop_physical_valid_pct": float(cop_physical_valid_pct),
        "cop_physical_invalid_count": cop_physical_invalid_count,
        "cop_adjusted_count": int(cop_adjusted_count),
        "off_forced_count": int(off_mask.sum()),
        "runtime_false_negative_count": runtime_false_negative_count,
        "leakage_flag": bool(leakage_flag),
    }


def threshold_status(metrics: dict) -> dict:
    """Apply science and business thresholds to metric outputs.

    Args:
        metrics: Metrics dictionary returned by ``evaluate_dual``.

    Returns:
        dict: Pass/fail states for science, business, and readiness summaries.
    """
    science_gates = {
        "r2_elec_pass": metrics["r2_elec"] > 0.80,
        "r2_heat_pass": metrics["r2_heat"] > 0.80,
        "mae_elec_pass": metrics["mae_elec_w"] < 90.0,
        "mae_heat_pass": metrics["mae_heat_w"] < 120.0,
    }

    business_gates = {
        "energy_err_pass": metrics["energy_err_pct"] < 5.0,
        "heat_err_pass": metrics["heat_err_pct"] < 5.0,
        "cop_err_pass": metrics["cop_err_pct"] < 6.0,
        "cop_physical_pass": metrics["cop_physical_invalid_count"] == 0,
        "leakage_pass": not metrics["leakage_flag"],
        "runtime_fn_pass": metrics["runtime_false_negative_count"] == 0,
    }

    summary = {
        "science_overall_pass": all(science_gates.values()),
        "business_overall_pass": all(business_gates.values()),
        "production_ready_pass": all(business_gates.values()),
        "research_ready_pass": all(science_gates.values()) and all(business_gates.values()),
    }

    return {
        "science_gates": science_gates,
        "business_gates": business_gates,
        "summary": summary,
    }



def build_report(
    elec_features: list[str],
    heat_features: list[str],
    elec_model: XGBRegressor,
    heat_model: XGBRegressor,
    metrics: dict,
    gates: dict,
    classifier_stats: dict,
    elec_model_path: str,
    heat_model_path: str,
    pipeline_artifacts: dict,
    feature_schema_path: str,
    selected_strategy: str,
    strategy_scores: list[dict],
    tuning_rows: list[dict],
    slice_metrics_df: pd.DataFrame,
) -> str:
    """Render a human-readable training report.

    Args:
        elec_features: Electricity feature list.
        heat_features: Heat feature list.
        elec_model: Trained electricity model.
        heat_model: Trained heat model.
        metrics: Final test metrics.
        gates: Threshold gate status object.
        classifier_stats: Runtime classifier diagnostics.
        elec_model_path: Saved electricity model path.
        heat_model_path: Saved heat model path.
        pipeline_artifacts: Persisted artifact path mapping.
        feature_schema_path: Path to saved feature schema.
        selected_strategy: Winning target strategy.
        strategy_scores: Strategy leaderboard entries.
        tuning_rows: Walk-forward tuning results.
        slice_metrics_df: Slice-level diagnostics table.

    Returns:
        str: Formatted report content.
    """
    elec_importance = pd.Series(elec_model.feature_importances_, index=elec_features).sort_values(ascending=False)
    heat_importance = pd.Series(heat_model.feature_importances_, index=heat_features).sort_values(ascending=False)

    science_gates = gates["science_gates"]
    business_gates = gates["business_gates"]
    summary = gates["summary"]


    lines = [
        "=" * 64,
        "DAIKIN HEAT PUMP - DUAL TARGET TRAINING REPORT V4",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 64,
        "",
        "SCIENCE METRICS",
        "TARGET 1: ELECTRICITY (heatpump_elec)",
        f"R2: {metrics['r2_elec']:.4f} | Threshold > 0.80 | PASS={science_gates['r2_elec_pass']}",
        f"MAE: {metrics['mae_elec_w']:.1f} W | Threshold < 90W | PASS={science_gates['mae_elec_pass']}",
        "",
        "TARGET 2: HEAT OUTPUT (heatpump_heat)",
        f"R2: {metrics['r2_heat']:.4f} | Threshold > 0.80 | PASS={science_gates['r2_heat_pass']}",
        f"MAE: {metrics['mae_heat_w']:.1f} W | Threshold < 120W | PASS={science_gates['mae_heat_pass']}",
        "",
        "BUSINESS METRICS",
        f"Electricity true/pred kWh: {metrics['true_elec_kwh']:.2f} / {metrics['pred_elec_kwh']:.2f}",
        f"Heat true/pred kWh: {metrics['true_heat_kwh']:.2f} / {metrics['pred_heat_kwh']:.2f}",
        f"Energy Error: {metrics['energy_err_pct']:.2f}% | Threshold < 5% | PASS={business_gates['energy_err_pass']}",
        f"Heat Error: {metrics['heat_err_pct']:.2f}% | Threshold < 5% | PASS={business_gates['heat_err_pass']}",
        f"True COP: {metrics['true_cop']:.3f}",
        f"Pred COP: {metrics['pred_cop']:.3f}",
        f"COP Error: {metrics['cop_err_pct']:.2f}% | Threshold < 6% | PASS={business_gates['cop_err_pass']}",
        f"Physical COP valid rate: {metrics['cop_physical_valid_pct']:.2f}%",
        f"Physical COP invalid count: {metrics['cop_physical_invalid_count']} | Must be 0 | PASS={business_gates['cop_physical_pass']}",
        f"Leakage flag (R2 > 0.97): {metrics['leakage_flag']} | PASS={business_gates['leakage_pass']}",
        f"Runtime false-negative count: {metrics['runtime_false_negative_count']} | Must be 0 | PASS={business_gates['runtime_fn_pass']}",
        "",
        "DEPLOYMENT STATUS",
        f"Business gate: {'PASS' if summary['business_overall_pass'] else 'FAIL'}",
        f"Science gate: {'PASS' if summary['science_overall_pass'] else 'FAIL'}",
        f"Production ready: {'PASS' if summary['production_ready_pass'] else 'FAIL'}",
        f"Research ready: {'PASS' if summary['research_ready_pass'] else 'FAIL'}",
        "",
        "TARGET STRATEGY TESTING",
        f"Selected strategy: {selected_strategy}",
        "",
        "ON/OFF CLASSIFIER DIAGNOSTICS",
        f"Confusion matrix [ [TN, FP], [FN, TP] ]: {classifier_stats['confusion_matrix'].tolist()}",
        f"True-ON predicted OFF (FN): {classifier_stats['fn']}",
        "",
        f"Saved feature schema: {feature_schema_path}",
        f"Saved electricity model: {elec_model_path}",
        f"Saved heat model: {heat_model_path}",
        "",
        f"Saved runtime classifier: {pipeline_artifacts['runtime_model']}",
        f"Saved electricity imputer: {pipeline_artifacts['electricity_imputer']}",
        f"Saved heat imputer: {pipeline_artifacts['heat_imputer']}",
        f"Saved pipeline metadata: {pipeline_artifacts['pipeline_meta']}",
        f"Saved slice calibrators: {pipeline_artifacts['slice_calibrators']}",

        "TOP 10 ELECTRICITY FEATURES",
    ]

    lines.append("Strategy leaderboard (lower total MAE is better):")
    for row in sorted(strategy_scores, key=lambda x: x["score_total_mae"]):
        lines.append(
            f"- {row['strategy']}: total_mae={row['score_total_mae']:.2f}, "
            f"mae_elec={row['mae_elec_w']:.2f}, mae_heat={row['mae_heat_w']:.2f}"
        )

    if tuning_rows:
        lines.append("")
        lines.append("Walk-forward tuning summary:")
        for row in tuning_rows:
            params_short = {
                "n_estimators": row["params"]["n_estimators"],
                "learning_rate": round(row["params"]["learning_rate"], 6),
                "max_depth": row["params"]["max_depth"],
                "min_child_weight": row["params"]["min_child_weight"],
            }
            lines.append(
                f"- {row['label']} candidate {row['candidate']}: "
                f"avg_mae={row['avg_mae_w']:.2f}W params={params_short}"
            )

    if not slice_metrics_df.empty:
        lines.append("")
        lines.append("Slice error analysis (top 6 worst heat MAE):")
        top_slices = slice_metrics_df.head(6)
        for _, s in top_slices.iterrows():
            lines.append(
                f"- system={int(s['system_id'])} capacity={int(s['capacity_kw'])}kW "
                f"rows={int(s['rows'])} mae_elec={s['mae_elec_w']:.1f}W "
                f"mae_heat={s['mae_heat_w']:.1f}W r2_elec={s['r2_elec']:.3f} "
                f"r2_heat={s['r2_heat']:.3f}"
            )


    for feat, score in elec_importance.head(10).items():
        lines.append(f"- {feat}: {score:.4f}")

    lines.append("")
    lines.append("TOP 10 HEAT FEATURES")
    for feat, score in heat_importance.head(10).items():
        lines.append(f"- {feat}: {score:.4f}")

    lines.append("=" * 64)
    return "\n".join(lines)


def main():
    """Execute full training pipeline and persist model artifacts.

    This includes feature schema generation, chronological splitting, optional
    walk-forward tuning, target-strategy search, slice calibration, evaluation,
    and report/manifest writing.
    """
    run_tag = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{FEATURE_POLICY_MODE}"
    df, elec_features, heat_features = load_data()
    print(f"Electricity feature count: {len(elec_features)}")
    print(f"Heat feature count: {len(heat_features)}")

    feature_schema = build_feature_schema(df, elec_features, heat_features)
    feature_schema_path = FEATURE_SCHEMA_PATH_TEMPLATE.format(run_tag=run_tag)

    with open(feature_schema_path, "w", encoding="utf-8") as f:
        json.dump(feature_schema, f, indent=2)


    train_df, test_df = time_split(df)
    split_diagnostics = validate_split_integrity(train_df, test_df)


    X_elec_train = train_df[elec_features]
    X_elec_test = test_df[elec_features]
    X_heat_train = train_df[heat_features]
    X_heat_test = test_df[heat_features]

    y_elec_train = train_df[TARGET_ELEC].astype(float)
    y_elec_test = test_df[TARGET_ELEC].astype(float)
    y_heat_train = train_df[TARGET_HEAT].astype(float)
    y_heat_test = test_df[TARGET_HEAT].astype(float)
    y_on_train = train_df["heating_on"].astype(int)
    y_on_test = test_df["heating_on"].astype(int)

    on_train_mask = y_on_train == 1
    on_test_mask = y_on_test == 1

    X_elec_train_imp, X_elec_test_imp, elec_imputer = impute_train_test(X_elec_train, X_elec_test)
    X_heat_train_imp, X_heat_test_imp, heat_imputer = impute_train_test(X_heat_train, X_heat_test)

    X_elec_train_on = X_elec_train_imp[on_train_mask.to_numpy()]
    X_heat_train_on = X_heat_train_imp[on_train_mask.to_numpy()]
    X_elec_eval_on = X_elec_test_imp[on_test_mask.to_numpy()]
    X_heat_eval_on = X_heat_test_imp[on_test_mask.to_numpy()]
    y_elec_train_on = y_elec_train[on_train_mask]
    y_heat_train_on = y_heat_train[on_train_mask]
    y_elec_eval_on = y_elec_test[on_test_mask]
    y_heat_eval_on = y_heat_test[on_test_mask]

    elec_tuned_params, elec_tuning_rows = walk_forward_tune_target(
        X_elec_train_on,
        y_elec_train_on,
        "electricity",
    )
    heat_tuned_params, heat_tuning_rows = walk_forward_tune_target(
        X_heat_train_on,
        y_heat_train_on,
        "heat",
    )
    tuning_rows = elec_tuning_rows + heat_tuning_rows

    strategy_runs = []
    for strategy in TARGET_STRATEGY_CANDIDATES:
        print(f"\nEvaluating target strategy: {strategy}")
        run = strategy_train_predict(
            strategy,
            X_elec_train_on,
            X_elec_test_imp,
            X_heat_train_on,
            X_heat_test_imp,
            y_elec_train_on,
            y_heat_train_on,
            y_elec_test,
            y_heat_test,
            X_elec_eval_on,
            X_heat_eval_on,
            y_elec_eval_on,
            y_heat_eval_on,
            elec_tuned_params,
            heat_tuned_params,
        )

        strategy_metrics = evaluate_dual(
            y_elec_test,
            run["pred_elec_test"],
            y_heat_test,
            run["pred_heat_test"],
            np.ones(len(y_elec_test), dtype=int),
            y_on_true=y_on_test,
        )
        run["strategy_metrics"] = strategy_metrics
        run["score_total_mae"] = strategy_metrics["mae_elec_w"] + strategy_metrics["mae_heat_w"]
        strategy_runs.append(run)

    strategy_runs = sorted(strategy_runs, key=lambda x: x["score_total_mae"])
    selected_run = strategy_runs[0]
    selected_strategy = selected_run["strategy"]
    print(f"\nSelected target strategy: {selected_strategy}")

    elec_model = selected_run["elec_model"]
    heat_model = selected_run["heat_model"]
    pred_elec_w = selected_run["pred_elec_test"].copy()
    pred_heat_w = selected_run["pred_heat_test"].copy()

    slice_calibrators = {}
    if ENABLE_SLICE_CALIBRATION:
        slice_calibrators = build_slice_calibrators(
            train_df.loc[on_train_mask, ["system_id", "capacity_kw", TARGET_ELEC, TARGET_HEAT]],
            selected_run["pred_elec_train_on"],
            selected_run["pred_heat_train_on"],
        )
        pred_elec_w, pred_heat_w = apply_slice_calibrators(
            test_df,
            pred_elec_w,
            pred_heat_w,
            slice_calibrators,
        )

    runtime_model, on_pred, on_proba = train_runtime_classifier(
        X_elec_train_imp,
        X_elec_test_imp,
        y_on_train,
        y_on_test,
    )

    classifier_stats = classifier_diagnostics(y_on_test, on_pred)
    print("\n=== ON/OFF CLASSIFIER REPORT (TEST SET) ===")
    print(classifier_stats["classification_report"])
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(classifier_stats["confusion_matrix"])
    print(f"True-ON predicted OFF (FN): {classifier_stats['fn']}")
    print("=== END CLASSIFIER REPORT ===\n")

    metrics = evaluate_dual(
        y_elec_test,
        pred_elec_w,
        y_heat_test,
        pred_heat_w,
        on_pred,
        y_on_true=y_on_test,
    )
    gates = threshold_status(metrics)

    slice_metrics_df = slice_error_analysis(
        test_df,
        y_elec_test,
        y_heat_test,
        pred_elec_w,
        pred_heat_w,
    )

    elec_model_path = f"{MODEL_DIR}/xgb_elec_v{run_tag}.json"
    heat_model_path = f"{MODEL_DIR}/xgb_heat_v{run_tag}.json"

    elec_model.save_model(elec_model_path)
    heat_model.save_model(heat_model_path)
    pipeline_artifacts = save_pipeline_artifacts(
            run_tag=run_tag,
            elec_imputer=elec_imputer,
            heat_imputer=heat_imputer,
            runtime_model=runtime_model,
            elec_features=elec_features,
            heat_features=heat_features,
            selected_strategy=selected_strategy,
            slice_calibrators=slice_calibrators,
            )


    report = build_report(
        elec_features,
        heat_features,
        elec_model,
        heat_model,
        metrics,
        gates,
        classifier_stats,
        elec_model_path,
        heat_model_path,
        pipeline_artifacts,
        feature_schema_path,
        selected_strategy,
        [
            {
                "strategy": r["strategy"],
                "score_total_mae": r["score_total_mae"],
                "mae_elec_w": r["strategy_metrics"]["mae_elec_w"],
                "mae_heat_w": r["strategy_metrics"]["mae_heat_w"],
            }
            for r in strategy_runs
        ],
        tuning_rows,
        slice_metrics_df,
    )
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "input_path": INPUT_PATH,
        "feature_policy_mode": FEATURE_POLICY_MODE,
        "selected_target_strategy": selected_strategy,
        "walk_forward_tuned_params": {
            "electricity": elec_tuned_params,
            "heat": heat_tuned_params,
        },
        "split_diagnostics": split_diagnostics,
        "feature_schema_path": feature_schema_path,
        "feature_schema": feature_schema,


        "features": {
            "electricity": elec_features,
            "heat": heat_features,
        },
        "targets": [TARGET_ELEC, TARGET_HEAT],
        "models": {
            "electricity": elec_model_path,
            "heat": heat_model_path,
        },
        "metrics": metrics,
        "strategy_scores": [
            {
                "strategy": r["strategy"],
                "score_total_mae": r["score_total_mae"],
                "mae_elec_w": r["strategy_metrics"]["mae_elec_w"],
                "mae_heat_w": r["strategy_metrics"]["mae_heat_w"],
            }
            for r in strategy_runs
        ],
        "slice_error_analysis_top10": slice_metrics_df.head(10).to_dict(orient="records"),
        "slice_calibrator_count": len(slice_calibrators),
        "classifier": {
            "confusion_matrix": classifier_stats["confusion_matrix"].tolist(),
            "classification_report": classifier_stats["classification_report"],
            "tn": classifier_stats["tn"],
            "fp": classifier_stats["fp"],
            "fn": classifier_stats["fn"],
            "tp": classifier_stats["tp"],
        },
        "gates": gates,
        "pipeline_artifacts": pipeline_artifacts,
        "imputer_strategy": {
            "electricity": "median_fit_train_only",
            "heat": "median_fit_train_only",
        },

    }
    manifest_path = f"{MODEL_DIR}/run_manifest_{run_tag}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + report)
    print(f"\nSaved report: {REPORT_PATH}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
