"""
FILE 4 - 04_train_model.py  [VERSION 3 - DUAL TARGET]
========================================================
PURPOSE : Train dual XGBoost models for electricity and heat output.
INPUT   : data/processed/daikin_clean.csv
OUTPUT  : models/xgb_elec_v{date}.json
          models/xgb_heat_v{date}.json
          models/training_report.txt
          models/run_manifest_{date}.json
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, r2_score
from xgboost import XGBClassifier, XGBRegressor

INPUT_PATH = "data/processed/daikin_clean.csv"
MODEL_DIR = "models"
REPORT_PATH = f"{MODEL_DIR}/training_report.txt"

TARGET_ELEC = "heatpump_elec"
TARGET_HEAT = "heatpump_heat"

COMMON_FEATURES = [
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
    "elec_lag1",
    "flowT_lag1",
    "returnT_lag1",
    "heating_on_lag1",
    "elec_lag24",
    "elec_lag168",
    "elec_lag1_pct",
    "heating_on_lag24",
    "run_hours",
    "cop_lag1",
    "was_defrost_lag1",
    "deltaT_house_lag1",
    "deltaT_lift_lag1",
    "lift_per_kw_lag1",
    "elec_lag2",
    "elec_lag3",
    "elec_lag4",
    "elec_lag6",
]

ELEC_ONLY_FEATURES = [
]

HEAT_ONLY_FEATURES = [
    "heat_lag1",
    "heat_lag24",
    "heat_lag168",
    "flowrate_lag1",
]

REQUIRED_ELEC_FEATURES = ["elec_lag1_pct"]

os.makedirs(MODEL_DIR, exist_ok=True)


def model_hyperparams(label: str) -> dict:
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


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])
    df = df.sort_values(["timestamp", "system_id"]).reset_index(drop=True)

    required = ["timestamp", "system_id", TARGET_ELEC, TARGET_HEAT] + COMMON_FEATURES + ELEC_ONLY_FEATURES + HEAT_ONLY_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for dual-target training: {missing}")

    before = len(df)
    df = df.dropna(subset=[TARGET_ELEC, TARGET_HEAT])
    print(f"Loaded rows: {before} -> {len(df)} after label checks")
    print(f"Systems: {sorted(df['system_id'].unique())}")
    print(f"Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    return df


def build_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    meta_cols = []
    for c in df.columns:
        if (c.startswith("emitter_") or c.startswith("zone_")) and pd.api.types.is_numeric_dtype(df[c]):
            meta_cols.append(c)
    meta_cols = sorted(meta_cols)

    elec_features = COMMON_FEATURES + ELEC_ONLY_FEATURES + meta_cols
    heat_features = COMMON_FEATURES + HEAT_ONLY_FEATURES + meta_cols

    # Safety check: ensure known high-value electricity feature is always present.
    for feature in REQUIRED_ELEC_FEATURES:
        if feature not in elec_features:
            elec_features.append(feature)

    return elec_features, heat_features


def time_split(df: pd.DataFrame, train_ratio: float = 0.8):
    split_idx = int(len(df) * train_ratio)
    split_ts = df["timestamp"].iloc[split_idx]

    print(f"Time split at: {split_ts}")
    print(f"Train rows: {split_idx}")
    print(f"Test rows: {len(df) - split_idx}")

    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def impute_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame):
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    return X_train_imp, X_test_imp, imputer


def train_single_target(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    label: str,
    X_eval: np.ndarray | None = None,
    y_eval: pd.Series | np.ndarray | None = None,
):
    model = XGBRegressor(**model_hyperparams(label))
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
) -> dict:
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
        "leakage_flag": bool(leakage_flag),
    }


def threshold_status(metrics: dict) -> dict:
    return {
        "r2_elec_pass": metrics["r2_elec"] > 0.80,
        "r2_heat_pass": metrics["r2_heat"] > 0.80,
        "mae_elec_pass": metrics["mae_elec_w"] < 90.0,
        "mae_heat_pass": metrics["mae_heat_w"] < 120.0,
        "energy_err_pass": metrics["energy_err_pct"] < 5.0,
        "heat_err_pass": metrics["heat_err_pct"] < 5.0,
        "cop_err_pass": metrics["cop_err_pct"] < 6.0,
        "cop_physical_pass": metrics["cop_physical_invalid_count"] == 0,
        "leakage_pass": not metrics["leakage_flag"],
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
) -> str:
    elec_importance = pd.Series(elec_model.feature_importances_, index=elec_features).sort_values(ascending=False)
    heat_importance = pd.Series(heat_model.feature_importances_, index=heat_features).sort_values(ascending=False)

    all_pass = all(gates.values())

    lines = [
        "=" * 64,
        "DAIKIN HEAT PUMP - DUAL TARGET TRAINING REPORT V3",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 64,
        "",
        "TARGET 1: ELECTRICITY (heatpump_elec)",
        f"R2: {metrics['r2_elec']:.4f} | Threshold > 0.80 | PASS={gates['r2_elec_pass']}",
        f"MAE: {metrics['mae_elec_w']:.1f} W | Threshold < 90W | PASS={gates['mae_elec_pass']}",
        "",
        "TARGET 2: HEAT OUTPUT (heatpump_heat)",
        f"R2: {metrics['r2_heat']:.4f} | Threshold > 0.80 | PASS={gates['r2_heat_pass']}",
        f"MAE: {metrics['mae_heat_w']:.1f} W | Threshold < 120W | PASS={gates['mae_heat_pass']}",
        "",
        "AGGREGATE ENERGY METRICS",
        f"Electricity true/pred kWh: {metrics['true_elec_kwh']:.2f} / {metrics['pred_elec_kwh']:.2f}",
        f"Heat true/pred kWh: {metrics['true_heat_kwh']:.2f} / {metrics['pred_heat_kwh']:.2f}",
        f"Energy Error: {metrics['energy_err_pct']:.2f}% | Threshold < 5% | PASS={gates['energy_err_pass']}",
        f"Heat Error: {metrics['heat_err_pct']:.2f}% | Threshold < 5% | PASS={gates['heat_err_pass']}",
        "",
        "COP VALIDATION (predicted heat / predicted electricity)",
        f"True COP: {metrics['true_cop']:.3f}",
        f"Pred COP: {metrics['pred_cop']:.3f}",
        f"COP Error: {metrics['cop_err_pct']:.2f}% | Threshold < 6% | PASS={gates['cop_err_pass']}",
        f"Physical COP valid rate: {metrics['cop_physical_valid_pct']:.2f}%",
        f"Physical COP invalid count: {metrics['cop_physical_invalid_count']} | Must be 0 | PASS={gates['cop_physical_pass']}",
        f"COP guardrail adjustments: {metrics['cop_adjusted_count']}",
        f"OFF-state forced-zero count: {metrics['off_forced_count']}",
        "",
        "ON/OFF CLASSIFIER DIAGNOSTICS",
        f"Confusion matrix [ [TN, FP], [FN, TP] ]: {classifier_stats['confusion_matrix'].tolist()}",
        f"True-ON predicted OFF (FN): {classifier_stats['fn']}",
        "",
        "LEAKAGE GUARD",
        f"Leakage flag (R2 > 0.97): {metrics['leakage_flag']} | PASS={gates['leakage_pass']}",
        "",
        f"OVERALL GATE: {'PASS' if all_pass else 'FAIL'}",
        f"Saved electricity model: {elec_model_path}",
        f"Saved heat model: {heat_model_path}",
        "",
        "TOP 10 ELECTRICITY FEATURES",
    ]

    for feat, score in elec_importance.head(10).items():
        lines.append(f"- {feat}: {score:.4f}")

    lines.append("")
    lines.append("TOP 10 HEAT FEATURES")
    for feat, score in heat_importance.head(10).items():
        lines.append(f"- {feat}: {score:.4f}")

    lines.append("=" * 64)
    return "\n".join(lines)


def main():
    df = load_data()
    elec_features, heat_features = build_feature_lists(df)
    print(f"Electricity feature count: {len(elec_features)}")
    print(f"Heat feature count: {len(heat_features)}")

    train_df, test_df = time_split(df)

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

    elec_model, pred_elec_w = train_single_target(
        X_elec_train_imp[on_train_mask.to_numpy()],
        X_elec_test_imp,
        y_elec_train[on_train_mask],
        y_elec_test,
        "electricity",
        X_eval=X_elec_test_imp[on_test_mask.to_numpy()],
        y_eval=y_elec_test[on_test_mask],
    )
    heat_model, pred_heat_w = train_single_target(
        X_heat_train_imp[on_train_mask.to_numpy()],
        X_heat_test_imp,
        y_heat_train[on_train_mask],
        y_heat_test,
        "heat",
        X_eval=X_heat_test_imp[on_test_mask.to_numpy()],
        y_eval=y_heat_test[on_test_mask],
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

    metrics = evaluate_dual(y_elec_test, pred_elec_w, y_heat_test, pred_heat_w, on_pred)
    gates = threshold_status(metrics)

    run_tag = datetime.now().strftime("%Y%m%d")
    elec_model_path = f"{MODEL_DIR}/xgb_elec_v{run_tag}.json"
    heat_model_path = f"{MODEL_DIR}/xgb_heat_v{run_tag}.json"

    elec_model.save_model(elec_model_path)
    heat_model.save_model(heat_model_path)

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
    )
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "input_path": INPUT_PATH,
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
        "classifier": {
            "confusion_matrix": classifier_stats["confusion_matrix"].tolist(),
            "classification_report": classifier_stats["classification_report"],
            "tn": classifier_stats["tn"],
            "fp": classifier_stats["fp"],
            "fn": classifier_stats["fn"],
            "tp": classifier_stats["tp"],
        },
        "gates": gates,
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
