"""Batch monitoring utilities for production scoring diagnostics.

Concepts:
- Data quality checks using null-rate and schema-level summaries.
- Prediction health checks using range and guardrail statistics.
- Distribution-shift screening using simple z-score drift signals.
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd


def summarize_batch_monitoring(
    input_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
) -> dict:
    """Build a compact monitoring summary for a scored batch.

    Args:
        input_df: Input features used for scoring.
        prediction_df: Model outputs for the same batch.
        reference_df: Optional baseline dataset for drift comparison.

    Returns:
        dict: Monitoring payload containing quality stats, ranges, drift, and alerts.
    """
    # Summarize serving health in a compact machine-readable payload.
    numeric_input_cols = input_df.select_dtypes(include=[np.number]).columns.tolist()

    null_rates = input_df.isna().mean().sort_values(ascending=False)
    pred_rows = len(prediction_df)
    on_rate = float(prediction_df["runtime_on_pred"].mean()) if "runtime_on_pred" in prediction_df else None

    summary = {
        "generated_at": datetime.now().isoformat(),
        "input_rows": int(len(input_df)),
        "prediction_rows": int(pred_rows),
        "successful_prediction_rate": float(pred_rows / max(len(input_df), 1)),
        "top_null_rate_features": {
            k: float(v) for k, v in null_rates.head(10).to_dict().items()
        },
        "prediction_ranges": {
            "pred_heatpump_elec_min": float(prediction_df["pred_heatpump_elec"].min()),
            "pred_heatpump_elec_max": float(prediction_df["pred_heatpump_elec"].max()),
            "pred_heatpump_heat_min": float(prediction_df["pred_heatpump_heat"].min()),
            "pred_heatpump_heat_max": float(prediction_df["pred_heatpump_heat"].max()),
            "pred_cop_min": float(np.nanmin(prediction_df["pred_cop"])),
            "pred_cop_max": float(np.nanmax(prediction_df["pred_cop"])),
        },
        "runtime_classifier": {
            "predicted_on_rate": on_rate,
            "predicted_off_rate": float(1 - on_rate) if on_rate is not None else None,
        },
        "guardrail_counts": {
            "cop_guardrail_adjusted_count": int(prediction_df.get("cop_guardrail_adjusted", pd.Series(dtype=int)).sum())
        },
        "alerts": [],
    }

    if summary["successful_prediction_rate"] < 0.99:
        summary["alerts"].append("prediction_volume_drop")

    if summary["prediction_ranges"]["pred_cop_max"] > 8.0:
        summary["alerts"].append("cop_above_guardrail")

    if summary["prediction_ranges"]["pred_heatpump_elec_min"] < 0:
        summary["alerts"].append("negative_electricity_prediction")

    if reference_df is not None:
        # Estimate first-order drift against a reference batch.
        ref_numeric = reference_df[numeric_input_cols].select_dtypes(include=[np.number])
        cur_numeric = input_df[numeric_input_cols].select_dtypes(include=[np.number])

        drift = {}
        for col in cur_numeric.columns:
            ref_mean = ref_numeric[col].mean()
            cur_mean = cur_numeric[col].mean()
            ref_std = ref_numeric[col].std()
            if pd.notna(ref_std) and ref_std > 0:
                z = abs(cur_mean - ref_mean) / ref_std
                drift[col] = float(z)

        top_drift = dict(sorted(drift.items(), key=lambda x: x[1], reverse=True)[:10])
        summary["feature_mean_shift_zscore"] = top_drift

        if any(v > 3.0 for v in top_drift.values()):
            summary["alerts"].append("feature_distribution_shift")

    return summary


def save_monitoring_summary(summary: dict, output_path: str):
    """Persist monitoring summary JSON to disk.

    Args:
        summary: Monitoring dictionary returned by ``summarize_batch_monitoring``.
        output_path: Destination JSON path.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
