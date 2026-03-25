"""Feature engineering stage for heat pump modeling.

Techniques used in this module:
- Physics-derived variables for thermal demand and lift.
- Cyclical temporal encoding for hour and month seasonality.
- Multi-horizon lag memory for hourly, daily, and weekly behavior.
- Slice-aware metadata expansion for system-level specialization.
"""

import pandas as pd
import numpy as np
import os
import glob
import logging

# Paths and constants shared by all feature transforms.
RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
OUTPUT_PATH   = f"{PROCESSED_DIR}/daikin_features.csv"

STANDBY_THRESHOLD_W = 50    # Below this = standby(10W documented), not real heating

# Heating degree hour baseline used in demand proxy calculations.
BASE_TEMP = 15.5

os.makedirs(PROCESSED_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# Load and merge all per-system raw files into one ordered frame.
def load_all_systems() -> pd.DataFrame:
    """Load and concatenate all raw system files.

    Returns:
        pd.DataFrame: Combined dataset sorted by ``system_id`` and ``timestamp``.

    Raises:
        FileNotFoundError: If no raw system files are found in ``RAW_DIR``.
    """
    files = sorted(glob.glob(f"{RAW_DIR}/system_*_raw.csv"))
    if not files:
        raise FileNotFoundError(
            f"No raw files found in {RAW_DIR}. Run 01_fetch_data.py first."
        )
    frames = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["timestamp"])
        frames.append(df)
        logging.info(f"Loaded {f} â†’ {df.shape[0]} rows")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["system_id", "timestamp"]).reset_index(drop=True)
    logging.info(f"âœ… Combined shape: {combined.shape}")
    return combined


# Standby normalization to suppress non-heating sensor noise.
def apply_standby_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Zero heat and flow-rate during standby power draw.

    Args:
        df: Input feature table containing electricity, heat, and flow columns.

    Returns:
        pd.DataFrame: DataFrame with standby rows normalized.
    """
    standby_mask = df["heatpump_elec"] < STANDBY_THRESHOLD_W
    df.loc[standby_mask, "heatpump_heat"]     = 0
    df.loc[standby_mask, "heatpump_flowrate"] = 0
    logging.info(f"Standby filter: {standby_mask.sum()} rows zeroed out.")
    return df


# Core energy and runtime-state derived metrics.
def add_energy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add core energy and runtime-state metrics.

    Args:
        df: Input DataFrame with raw power columns.

    Returns:
        pd.DataFrame: DataFrame enriched with kWh, COP, and heating state.
    """
    df["elec_kwh"] = df["heatpump_elec"] / 1000.0
    df["heat_kwh"] = df["heatpump_heat"] / 1000.0

    df["cop"] = np.where(
        df["heatpump_elec"] > STANDBY_THRESHOLD_W,
        df["heatpump_heat"] / df["heatpump_elec"].replace(0, np.nan),
        np.nan
    )

    df["heating_on"] = (
        (df["heatpump_heat"] > 0) &
        (df["heatpump_elec"] > STANDBY_THRESHOLD_W)
    ).astype(int)

    return df


# First-order thermodynamic and building-demand signals.
def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create first-order thermodynamic features.

    Args:
        df: Input DataFrame with flow, return, room, and outdoor temperatures.

    Returns:
        pd.DataFrame: DataFrame with thermal gradient and deficit terms.
    """
    df["deltaT_house"] = df["heatpump_flowT"]  - df["heatpump_returnT"]
    df["deltaT_lift"]  = df["heatpump_flowT"]  - df["heatpump_outsideT"]
    df["temp_deficit"] = df["heatpump_roomT"]  - df["heatpump_outsideT"]
    return df


# Scale-aware interaction features and degree-hour demand proxy.
def add_enhanced_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add scale-aware interaction features.

    Args:
        df: DataFrame containing temperature and capacity fields.

    Returns:
        pd.DataFrame: DataFrame with normalized load and HDH features.
    """
    # Capacity-normalized deficit approximates per-kW heating burden.
    df["load_ratio"]   = df["temp_deficit"] / df["capacity_kw"]

    # Heating Degree Hours measures cold-weather demand pressure.
    df["hdh"] = (BASE_TEMP - df["heatpump_outsideT"]).clip(lower=0)

    return df


# Cyclical temporal encodings plus weather-derived season state.
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add raw and cyclical time features.

    Args:
        df: DataFrame with a timezone-aware ``timestamp`` column.

    Returns:
        pd.DataFrame: DataFrame with hour/month harmonics and season flag.
    """
    hour  = df["timestamp"].dt.hour
    month = df["timestamp"].dt.month

    # Raw temporal fields remain useful for diagnostics.
    df["hour"]        = hour
    df["month"]       = month
    df["day_of_week"] = df["timestamp"].dt.dayofweek   # 0=Monday, 6=Sunday

    # Harmonic encoding removes artificial month/hour edge discontinuities.
    df["hour_sin"]  = np.sin(2 * np.pi * hour  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * hour  / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Temperature-driven season flag generalizes across climates.
    df["is_heating_season"] = (df["heatpump_outsideT"] < BASE_TEMP).astype(int)

    return df


# One-step memory terms built independently per system.
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add one-step lag features per system.

    Args:
        df: DataFrame containing base operational variables.

    Returns:
        pd.DataFrame: DataFrame with lag-1 memory terms.
    """
    lag_cols = {
        "heatpump_flowT":    "flowT_lag1",
        "heatpump_returnT":  "returnT_lag1",
        "heatpump_flowrate": "flowrate_lag1",
        "heatpump_elec":     "elec_lag1",
        "heating_on":        "heating_on_lag1",
    }
    for source_col, lag_col in lag_cols.items():
        df[lag_col] = df.groupby("system_id")[source_col].shift(1)

    logging.info("Lag-1 features created.")
    return df


# Multi-scale lag memory and transient-state indicators.
def add_extended_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add multi-horizon lag and transient-state features.

    Args:
        df: DataFrame with base and lag-ready columns.

    Returns:
        pd.DataFrame: DataFrame with daily/weekly lags, run blocks, and COP memory.
    """
    g = df.groupby("system_id")

    # Daily and weekly lags capture repeating load structure.
    df["elec_lag24"]       = g["heatpump_elec"].shift(24)    # Same hour yesterday
    df["elec_lag168"]      = g["heatpump_elec"].shift(168)   # Same hour last week
    df["heat_lag1"]        = g["heatpump_heat"].shift(1)
    df["heat_lag24"]       = g["heatpump_heat"].shift(24)
    df["heat_lag168"]      = g["heatpump_heat"].shift(168)
    df["heating_on_lag24"] = g["heating_on"].shift(24)       # Was pump on yesterday at this hour?

    # Prior-hour house-side delta-T approximates absorbed heat gradient.
    df["deltaT_house_lag1"] = df["flowT_lag1"] - df["returnT_lag1"]

    # Prior-hour flow temperature against current ambient indicates lift pressure.
    df["deltaT_lift_lag1"]  = df["flowT_lag1"] - df["heatpump_outsideT"]

    # Capacity normalization improves transfer across different unit sizes.
    df["lift_per_kw_lag1"]  = df["deltaT_lift_lag1"] / df["capacity_kw"]

    df["elec_lag2"] = g["heatpump_elec"].shift(2)   # 2 hours ago
    df["elec_lag3"] = g["heatpump_elec"].shift(3)   # 3 hours ago
    df["elec_lag4"] = g["heatpump_elec"].shift(4)
    df["elec_lag6"] = g["heatpump_elec"].shift(6)

    # Contiguous ON/OFF block counting captures warm-up and inertia effects.
    df["run_hours"] = (
        g["heating_on"]
        .transform(
            lambda x: x.groupby((x != x.shift()).cumsum()).cumcount()
        )
    )

    # COP persistence terms model short-horizon efficiency continuity.
    df["cop_lag1"]          = g["cop"].shift(1)
    df["was_defrost_lag1"]  = (df["cop_lag1"] < 1.0).astype(float)
    # Keep float dtype so leading NaN rows preserve uncertainty.

    # Lag normalized by nominal capacity for cross-system comparability.
    df["elec_lag1_pct"] = df["elec_lag1"] / (df["capacity_kw"] * 1000)

    logging.info("Extended lag features created.")
    return df


def add_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expand system identity into model-ready indicator columns.

    Args:
        df: DataFrame with system identity columns.

    Returns:
        pd.DataFrame: DataFrame with one-hot encoded system slices.
    """
    system_dummies = pd.get_dummies(df["system_id"], prefix="system", dtype=float)

    df = pd.concat([df, system_dummies], axis=1)
    return df


# Short rolling context on outdoor temperature.
def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add short-window rolling context features.

    Args:
        df: DataFrame containing outdoor temperature by system.

    Returns:
        pd.DataFrame: DataFrame with rolling outdoor temperature averages.
    """
    df["outsideT_3h_avg"] = (
        df.groupby("system_id")["heatpump_outsideT"]
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )
    return df


# Orchestrate feature transformations in dependency-safe order.
def main():
    """Run the full feature-engineering pipeline.

    The routine loads raw telemetry, applies ordered transforms, persists the
    engineered table, and prints a compact verification summary.
    """
    # 1. Load
    df = load_all_systems()

    # Transformation order preserves feature dependencies and avoids leakage.
    df = apply_standby_filter(df)
    df = add_energy_metrics(df)
    df = add_physics_features(df)
    df = add_enhanced_physics_features(df)
    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = add_extended_lag_features(df)
    df = add_rolling_features(df)
    df = add_metadata_features(df)

    # Save the fully engineered training table.
    df.to_csv(OUTPUT_PATH, index=False)

    logging.info(f"\nâœ… Feature dataset saved â†’ {OUTPUT_PATH}")
    logging.info(f"   Shape       : {df.shape}")
    logging.info(f"   Columns ({len(df.columns)}): {df.columns.tolist()}")

    # Quick verification view for debugging and QA.
    print("\n--- Feature Verification (System 615, rows 1â€“4) ---")
    sample = df[df["system_id"] == 615].iloc[1:5]
    print(sample[[
        "timestamp",
        "heatpump_elec", "elec_lag1", "elec_lag24",
        "cop", "cop_lag1", "was_defrost_lag1",
        "run_hours", "hdh", "is_heating_season",
        "hour_sin", "month_sin", "load_ratio"
    ]].to_string(index=False))

    print("\n--- Heating Season Distribution ---")
    print(df["is_heating_season"].value_counts().rename({1: "Heating ON", 0: "Heating OFF"}))

    print("\n--- HDH Distribution (0 = warm day, high = cold day) ---")
    print(df["hdh"].describe().round(2))


if __name__ == "__main__":
    main()
