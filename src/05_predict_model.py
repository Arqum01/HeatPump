"""Serving stage for dual-target heat pump forecasting.

Concepts:
- Schema-first validation to keep train/serve parity.
- Strategy-aware inverse transforms for target-space consistency.
- Optional slice calibrators and COP guardrails at inference time.
"""

import json
import os
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from monitoring_model_06 import summarize_batch_monitoring, save_monitoring_summary



ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = "models/production" if os.getenv("VERCEL") else "models"
MODEL_DIR = Path(os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR))
if not MODEL_DIR.is_absolute():
    MODEL_DIR = ROOT_DIR / MODEL_DIR


def _resolve_artifact_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def resolve_latest_run_tag() -> str:
    """Resolve the newest available manifest version tag in the model folder.

    Returns:
        str: Lexicographically latest run tag from manifest filenames.

    Raises:
        FileNotFoundError: If model directory or manifest files are missing.
    """
    prefix = "run_manifest_"
    suffix = ".json"
    candidates = []

    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    for path in MODEL_DIR.glob("run_manifest_*.json"):
        name = path.name
        if name.startswith(prefix) and name.endswith(suffix):
            tag = name[len(prefix) : -len(suffix)]
            if tag:
                candidates.append(tag)

    if not candidates:
        raise FileNotFoundError(
            f"No manifest files found in {MODEL_DIR}. Run training first."
        )

    return sorted(candidates)[-1]


def apply_cop_guardrail(
    pred_elec_w: np.ndarray,
    pred_heat_w: np.ndarray,
    cop_min: float = 0.0,
    cop_max: float = 8.0,
):
    """Enforce non-negative and bounded-COP constraints on predictions.

    Args:
        pred_elec_w: Electricity predictions in watts.
        pred_heat_w: Heat predictions in watts.
        cop_min: Minimum allowed COP.
        cop_max: Maximum allowed COP.

    Returns:
        tuple[np.ndarray, np.ndarray, int]: Corrected electricity predictions,
        corrected heat predictions, and adjustment count.
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


def validate_inference_frame(
    df: pd.DataFrame,
    feature_schema: dict,
    allow_extra_columns: bool = True,
):
    """Validate and type-coerce serving data against a stored schema.

    Args:
        df: Input scoring DataFrame.
        feature_schema: Schema saved at training time.
        allow_extra_columns: Whether to tolerate non-required columns.

    Returns:
        dict: Validated frame plus electricity/heat feature views and column diagnostics.
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


@lru_cache(maxsize=8)
def load_bundle(run_tag: str):
    """Load all trained artifacts needed for inference.

    Args:
        run_tag: Version tag identifying the manifest and artifacts.

    Returns:
        dict: Bundle containing models, preprocessors, schema, and strategy metadata.
    """
    # Load all artifacts required to reproduce training-time behavior.
    manifest_path = MODEL_DIR / f"run_manifest_{run_tag}.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    feature_schema_path = _resolve_artifact_path(manifest["feature_schema_path"])
    with open(feature_schema_path, "r", encoding="utf-8") as f:
        feature_schema = json.load(f)

    elec_model = XGBRegressor()
    elec_model.load_model(str(_resolve_artifact_path(manifest["models"]["electricity"])))

    heat_model = XGBRegressor()
    heat_model.load_model(str(_resolve_artifact_path(manifest["models"]["heat"])))

    runtime_model = joblib.load(_resolve_artifact_path(manifest["pipeline_artifacts"]["runtime_model"]))
    elec_imputer = joblib.load(_resolve_artifact_path(manifest["pipeline_artifacts"]["electricity_imputer"]))
    heat_imputer = joblib.load(_resolve_artifact_path(manifest["pipeline_artifacts"]["heat_imputer"]))

    slice_calibrators = {}
    calibrator_path = manifest["pipeline_artifacts"].get("slice_calibrators")
    resolved_calibrator_path = _resolve_artifact_path(calibrator_path) if calibrator_path else None
    if resolved_calibrator_path and resolved_calibrator_path.exists():
        with open(resolved_calibrator_path, "r", encoding="utf-8") as f:
            slice_calibrators = json.load(f)

    target_strategy = manifest.get("selected_target_strategy", "none")

    return {
        "manifest": manifest,
        "feature_schema": feature_schema,
        "elec_model": elec_model,
        "heat_model": heat_model,
        "runtime_model": runtime_model,
        "elec_imputer": elec_imputer,
        "heat_imputer": heat_imputer,
        "slice_calibrators": slice_calibrators,
        "target_strategy": target_strategy,
    }


def apply_slice_calibrators(df: pd.DataFrame, pred_elec: np.ndarray, pred_heat: np.ndarray, calibrators: dict):
    """Apply optional per-slice multipliers to serving predictions.

    Args:
        df: Scored DataFrame with ``series_id`` and ``capacity_kw`` columns.
        pred_elec: Electricity predictions.
        pred_heat: Heat predictions.
        calibrators: Mapping of ``system|capacity`` keys to multipliers.

    Returns:
        tuple[np.ndarray, np.ndarray]: Calibrated electricity and heat predictions.
    """
    # Apply per-slice multipliers learned from training residual structure.
    if not calibrators:
        return pred_elec, pred_heat

    # Single-row/manual serving can provide only required model features,
    # which may omit slice identifiers used by calibrators.
    if "series_id" not in df.columns or "capacity_kw" not in df.columns:
        return pred_elec, pred_heat

    elec = pred_elec.astype(float).copy()
    heat = pred_heat.astype(float).copy()

    for pos, row in enumerate(df[["series_id", "capacity_kw"]].itertuples(index=False, name=None)):
        series_id, capacity_kw = row
        key = f"{int(series_id)}|{int(capacity_kw)}"
        mult = calibrators.get(key)
        if mult is None:
            continue
        elec[pos] *= float(mult.get("elec_mult", 1.0))
        heat[pos] *= float(mult.get("heat_mult", 1.0))

    return elec, heat


def predict_bundle(df: pd.DataFrame, run_tag: str) -> pd.DataFrame:
    """Run end-to-end inference for a data batch and a model run tag.

    Args:
        df: Raw input DataFrame containing all required serving features.
        run_tag: Version tag used to resolve model artifacts.

    Returns:
        pd.DataFrame: Scored DataFrame with runtime and prediction outputs.
    """
    bundle = load_bundle(run_tag)
    feature_schema = bundle["feature_schema"]

    checked = validate_inference_frame(df, feature_schema, allow_extra_columns=True)
    scored = checked["validated_frame"].copy()

    X_elec = checked["electricity_frame"]
    X_heat = checked["heat_frame"]

    X_elec_imp = bundle["elec_imputer"].transform(X_elec)
    X_heat_imp = bundle["heat_imputer"].transform(X_heat)

    on_proba = bundle["runtime_model"].predict_proba(X_elec_imp)[:, 1]
    on_pred = (on_proba >= 0.5).astype(int)

    strategy = bundle.get("target_strategy", "none")

    # Keep prediction reconstruction aligned with the selected training strategy.
    if strategy == "log1p_both":
        pred_elec_w = np.expm1(bundle["elec_model"].predict(X_elec_imp))
        pred_heat_w = np.expm1(bundle["heat_model"].predict(X_heat_imp))
    elif strategy == "cop_ratio_heat":
        pred_elec_w = bundle["elec_model"].predict(X_elec_imp)
        pred_ratio = np.maximum(bundle["heat_model"].predict(X_heat_imp), 0.0)
        pred_heat_w = pred_ratio * np.maximum(pred_elec_w.astype(float), 1e-6)
    else:
        pred_elec_w = bundle["elec_model"].predict(X_elec_imp)
        pred_heat_w = bundle["heat_model"].predict(X_heat_imp)

    pred_elec_w, pred_heat_w = apply_slice_calibrators(
        scored,
        np.maximum(pred_elec_w.astype(float), 0.0),
        np.maximum(pred_heat_w.astype(float), 0.0),
        bundle.get("slice_calibrators", {}),
    )

    off_mask = on_pred == 0
    pred_elec_w = pred_elec_w.astype(float).copy()
    pred_heat_w = pred_heat_w.astype(float).copy()

    pred_elec_w[off_mask] = 50.0
    pred_heat_w[off_mask] = 0.0

    pred_elec_corr_w, pred_heat_corr_w, cop_adjusted_count = apply_cop_guardrail(
        pred_elec_w,
        pred_heat_w,
        cop_min=0.0,
        cop_max=8.0,
    )

    scored["runtime_on_proba"] = on_proba
    scored["runtime_on_pred"] = on_pred
    scored["pred_heatpump_elec"] = pred_elec_corr_w
    scored["pred_heatpump_heat"] = pred_heat_corr_w
    scored["pred_cop"] = np.divide(
        pred_heat_corr_w,
        np.maximum(pred_elec_corr_w, 1e-6),
        out=np.full_like(pred_heat_corr_w, np.nan, dtype=float),
        where=pred_elec_corr_w > 1e-6,
    )
    scored["cop_guardrail_adjusted"] = 0
    if cop_adjusted_count > 0:
        raw_pred_heat_nonneg = np.maximum(pred_heat_w.astype(float), 0.0)
        changed_mask = np.abs(pred_heat_corr_w - raw_pred_heat_nonneg) > 1e-6
        scored.loc[changed_mask, "cop_guardrail_adjusted"] = 1

    return scored


if __name__ == "__main__":
    run_tag = resolve_latest_run_tag()
    sample_path = ROOT_DIR / "data" / "processed" / "daikin_clean.csv"
    df = pd.read_csv(sample_path, parse_dates=["timestamp"]).head(100)

    preds = predict_bundle(df, run_tag=run_tag)
    preds.to_csv(ROOT_DIR / "data" / "processed" / "sample_predictions.csv", index=False)

    monitoring = summarize_batch_monitoring(
        input_df=df,
        prediction_df=preds,
        reference_df=df,
    )
    save_monitoring_summary(monitoring, str(MODEL_DIR / "latest_monitoring_summary.json"))

    print(preds.head())
    print(json.dumps(monitoring, indent=2))


