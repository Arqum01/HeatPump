"""Data quality stage for model-ready heat pump observations.

Concepts:
- Conservative interpolation for short sensor dropouts.
- Physics-constrained filtering to reject implausible values.
- Label-safe missing-data policy that preserves usable systems.
"""

import pandas as pd
import numpy as np
import os
import logging

# File paths and physical limits used by cleaning rules.
INPUT_PATH   = "data/processed/daikin_features.csv"
OUTPUT_PATH  = "data/processed/daikin_clean.csv"
REPORT_PATH  = "data/processed/cleaning_report.csv"

# Physics-informed boundaries used for robust outlier filtering.
RULES = {
    # Energy — only block physically impossible negatives
    # Zero is allowed; negative power is physically implausible.
    "heatpump_elec_min":  0,
    "heatpump_heat_min":  0,

    # COP bounds include defrost behavior and high-efficiency operation.
    "cop_min":  0.0,
    "cop_max":  8.0,

    # Delta-T bounds protect against sensor spikes while retaining startup events.
    "deltaT_house_min":  0.0,
    "deltaT_house_max":  20.0,

    # Outdoor temperature guardrails cover realistic local extremes.
    "outsideT_min": -20,
    "outsideT_max":  40,
}

# Missing-value policy is anchored to target columns only.
CORE_COLS = ["heatpump_elec", "heatpump_heat"]

# Only input telemetry is interpolated; targets remain untouched.
INTERP_COLS = [
    "heatpump_outsideT",
    "heatpump_roomT",
    "heatpump_flowT",
    "heatpump_returnT",
]

os.makedirs("data/processed", exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# Conservative interpolation: recover one-hour gaps without long-range guessing.
def smart_interpolate(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate short one-hour gaps in selected sensor channels.

    Args:
        df: Input DataFrame with per-system ordered telemetry rows.

    Returns:
        pd.DataFrame: DataFrame with limited linear interpolation applied.
    """
    before = df[INTERP_COLS].isna().sum().sum()

    df[INTERP_COLS] = (
        df.groupby("system_id")[INTERP_COLS]
        .transform(lambda x: x.interpolate(method="linear", limit=1))
    )

    after = df[INTERP_COLS].isna().sum().sum()
    logging.info(f"Interpolation: filled {before - after} single-blink NaN values.")
    return df


# Physical plausibility screening for noisy telemetry.
def apply_boundary_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Filter rows outside physics-informed operating bounds.

    Args:
        df: Input DataFrame containing target and physics feature columns.

    Returns:
        pd.DataFrame: Filtered DataFrame that satisfies all boundary rules.
    """
    before = len(df)

    mask = (
        # Block physically impossible negatives
        (df["heatpump_elec"] >= RULES["heatpump_elec_min"]) &
        (df["heatpump_heat"] >= RULES["heatpump_heat_min"]) &

        # COP guardrail retains plausible defrost and efficient operation states.
        (df["cop"].isna() | (
            (df["cop"] >= RULES["cop_min"]) &
            (df["cop"] <= RULES["cop_max"])
        )) &

        # Delta-T guardrail rejects implausible thermal jumps.
        (df["deltaT_house"] >= RULES["deltaT_house_min"]) &
        (df["deltaT_house"] <= RULES["deltaT_house_max"]) &

        # Ambient guardrail removes unrealistic weather measurements.
        (df["heatpump_outsideT"] >= RULES["outsideT_min"]) &
        (df["heatpump_outsideT"] <= RULES["outsideT_max"])
    )

    df = df[mask].copy()
    after = len(df)
    logging.info(f"Boundary rules: removed {before - after} rows ({(before-after)/before*100:.1f}%)")
    return df


# Selective null filtering preserves rows with valid targets.
def selective_dropna(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows only when core target columns are missing.

    Args:
        df: Input DataFrame after interpolation and boundary filtering.

    Returns:
        pd.DataFrame: DataFrame retaining rows with valid electricity and heat labels.
    """
    before = len(df)
    df = df.dropna(subset=CORE_COLS)
    after = len(df)
    logging.info(f"Selective dropna: removed {before - after} rows missing core label data.")
    return df


# Per-system retention and quality summary.
def generate_report(df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
    """Generate per-system retention and quality statistics.

    Args:
        df_before: Dataset before cleaning transforms.
        df_after: Dataset after cleaning transforms.

    Returns:
        pd.DataFrame: Per-system report also saved to ``REPORT_PATH``.
    """
    rows = []
    for sid in df_before["system_id"].unique():
        b = df_before[df_before["system_id"] == sid]
        a = df_after[df_after["system_id"] == sid]
        kept_pct = len(a) / len(b) * 100 if len(b) > 0 else 0
        rows.append({
            "system_id":      sid,
            "capacity_kw":    b["capacity_kw"].iloc[0],
            "rows_before":    len(b),
            "rows_after":     len(a),
            "rows_kept_pct":  round(kept_pct, 1),
            "missing_roomT":  a["heatpump_roomT"].isna().sum(),
            "cop_mean":       round(a["cop"].mean(), 3) if "cop" in a else None,
        })
    report = pd.DataFrame(rows)
    report.to_csv(REPORT_PATH, index=False)
    return report


# Orchestrate cleaning transforms and emit audit outputs.
def main():
    """Execute the full cleaning workflow and emit diagnostic summaries."""
    # Load
    df = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])
    logging.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    df_before = df.copy()

    # Ordered transforms keep quality policy deterministic.
    df = smart_interpolate(df)
    df = apply_boundary_rules(df)
    df = selective_dropna(df)

    # Persist cleaned dataset.
    df.to_csv(OUTPUT_PATH, index=False)

    # Persist and print quality diagnostics.
    report = generate_report(df_before, df)

    total_pct = len(df) / len(df_before) * 100
    logging.info(f"\n✅ Clean dataset saved → {OUTPUT_PATH}")
    logging.info(f"   {len(df_before)} rows → {len(df)} rows ({total_pct:.1f}% retained)")

    print("\n--- Cleaning Report by System ---")
    print(report.to_string(index=False))

    print("\n--- COP Distribution ---")
    print(df["cop"].describe().round(3))

    print("\n--- System 228 Check ---")
    s228 = df[df["system_id"] == 228][["timestamp", "heatpump_elec", "heatpump_heat", "heatpump_roomT"]].head(5)
    print(s228.to_string(index=False))


if __name__ == "__main__":
    main()
