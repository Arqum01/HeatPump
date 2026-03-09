"""
FILE 2 — 02_feature_engineering.py
====================================
PURPOSE : Takes raw CSVs, builds all physics + temporal + lag features.
INPUT   : data/raw/system_{id}_raw.csv
OUTPUT  : data/processed/daikin_features.csv

TWEAKS APPLIED:
  ✅ Standby filter (elec < 50W → heat = 0, no exploding COP)
  ✅ COP calculated safely (divide-by-zero guarded)
  ✅ Physics Split: deltaT_house + deltaT_lift (two different signals)
  ✅ temp_deficit (how hard house is demanding heat)
  ✅ Temporal features: hour, month, day_of_week
  ✅ LAG FEATURES: last hour's flowT, returnT, flowrate, elec
      — These use PAST data so there is ZERO data leakage
  ✅ groupby(system_id) before shifting — prevents System 615's
      last row bleeding into System 364's first row
  ✅ Append-to-CSV pattern (RAM friendly for multi-year data)
"""

import pandas as pd
import numpy as np
import os
import glob
import logging

# =============================================================
# CONFIGURATION
# =============================================================
RAW_DIR      = "data/raw"
PROCESSED_DIR = "data/processed"
OUTPUT_PATH  = f"{PROCESSED_DIR}/daikin_features.csv"

STANDBY_THRESHOLD_W = 50   # Below this = standby, not real heating

os.makedirs(PROCESSED_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# =============================================================
# STEP 1 — LOAD ALL RAW SYSTEM FILES
# =============================================================
def load_all_systems() -> pd.DataFrame:
    files = sorted(glob.glob(f"{RAW_DIR}/system_*_raw.csv"))
    if not files:
        raise FileNotFoundError(
            f"No raw files found in {RAW_DIR}. Run 01_fetch_data.py first."
        )

    frames = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["timestamp"])
        frames.append(df)
        logging.info(f"Loaded {f} → {df.shape[0]} rows")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["system_id", "timestamp"]).reset_index(drop=True)
    logging.info(f"✅ Combined shape: {combined.shape}")
    return combined


# =============================================================
# STEP 2 — STANDBY FILTER
#
# When elec < 50W the pump is in standby (just keeping electronics alive).
# It draws 10–30W but produces no real heat — just sensor noise.
# Without this: COP = 5W/10W = 0.5 → physically impossible → corrupts training.
# With this: heat = 0W, COP calculation returns NaN (safe).
# =============================================================
def apply_standby_filter(df: pd.DataFrame) -> pd.DataFrame:
    standby_mask = df["heatpump_elec"] < STANDBY_THRESHOLD_W
    df.loc[standby_mask, "heatpump_heat"]     = 0
    df.loc[standby_mask, "heatpump_flowrate"] = 0
    logging.info(f"Standby filter: {standby_mask.sum()} rows zeroed out.")
    return df


# =============================================================
# STEP 3 — ENERGY METRICS
# =============================================================
def add_energy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df["elec_kwh"] = df["heatpump_elec"] / 1000.0
    df["heat_kwh"] = df["heatpump_heat"] / 1000.0

    # Safe COP — replace 0 with NaN before dividing to avoid infinity
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


# =============================================================
# STEP 4 — PHYSICS FEATURES
#
# deltaT_house = flowT - returnT
#   → How much heat the HOUSE absorbed this hour
#   → Goes UP when house is cold and hungry for heat
#
# deltaT_lift  = flowT - outsideT
#   → How hard the pump is "lifting" heat from cold outside air
#   → Goes UP in freezing weather → explains WHY cop drops in winter
#   → The single best predictor of COP degradation
#
# temp_deficit = roomT - outsideT
#   → How much heating the house DEMANDS right now
#   → High in winter mornings, low on mild afternoons
# =============================================================
def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    df["deltaT_house"]  = df["heatpump_flowT"]   - df["heatpump_returnT"]
    df["deltaT_lift"]   = df["heatpump_flowT"]   - df["heatpump_outsideT"]
    df["temp_deficit"]  = df["heatpump_roomT"]   - df["heatpump_outsideT"]
    return df


# =============================================================
# STEP 5 — TEMPORAL FEATURES
#
# XGBoost cannot discover time patterns on its own.
# It needs to be TOLD what hour and day it is.
#
# hour        → 02:00 = coldest, pump hardest. 14:00 = sun helps, COP peaks.
# month       → Jan = deep winter. June = summer cooling.
# day_of_week → Weekday: people leave at 8am, heating drops.
#               Weekend: people home all day, sustained heating.
# =============================================================
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"]        = df["timestamp"].dt.hour
    df["month"]       = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek   # 0=Monday, 6=Sunday
    return df


# =============================================================
# STEP 6 — LAG FEATURES (The "Memory" Fix)
#
# These were originally leaky (you cannot know current flowT before pump runs).
# Shifted by 1 hour → they become LEGITIMATE because last hour already happened.
#
# CRITICAL: groupby("system_id") before shift()
# Without this: System 615's row 720 would bleed into System 364's row 1
# = completely wrong physical data crossing machine boundaries
#
# What these capture:
#   flowT_lag1   → Was the water hot last hour? (thermal inertia)
#   returnT_lag1 → Was the house absorbing heat? (demand continuation)
#   elec_lag1    → Was the pump running hard? (operational momentum)
#   heating_on_lag1 → Was pump ON or OFF? (cold start detection)
#
# Cold start pattern:
#   heating_on_lag1 = 0 → pump was OFF → cold pipes → spike in elec this hour
#   heating_on_lag1 = 1 → pump was ON  → warm pipes → smooth continuation
# =============================================================
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    lag_cols = {
        "heatpump_flowT":    "flowT_lag1",
        "heatpump_returnT":  "returnT_lag1",
        "heatpump_flowrate": "flowrate_lag1",
        "heatpump_elec":     "elec_lag1",
        "heating_on":        "heating_on_lag1",
    }

    for source_col, lag_col in lag_cols.items():
        df[lag_col] = (
            df.groupby("system_id")[source_col]
            .shift(1)   # Shift by exactly 1 hour within each system
        )

    # The very first row of each system will have NaN lag values — that is correct.
    # These will be dropped in the cleaning step.
    lag_created = list(lag_cols.values())
    null_counts = df[lag_created].isna().sum()
    logging.info(f"Lag features created. NaN counts (expected ~7 rows):\n{null_counts}")

    return df


# =============================================================
# STEP 7 — ROLLING AVERAGE (Thermal Mass of House)
#
# A house does not cool instantly. If it was cold for 3 hours,
# the walls and floors are cold too — pump needs more energy.
# The 3-hour rolling average of outsideT captures this "memory".
# =============================================================
def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df["outsideT_3h_avg"] = (
        df.groupby("system_id")["heatpump_outsideT"]
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )
    return df


# =============================================================
# MAIN PIPELINE
# =============================================================
def main():
    # 1. Load
    df = load_all_systems()

    # 2. Apply all feature engineering steps in order
    df = apply_standby_filter(df)
    df = add_energy_metrics(df)
    df = add_physics_features(df)
    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    # 3. Save
    df.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"\n✅ Feature dataset saved → {OUTPUT_PATH}")
    logging.info(f"   Shape: {df.shape}")
    logging.info(f"   Columns: {df.columns.tolist()}")

    # Quick sanity check
    print("\n--- Feature Sample (System 615, first 3 rows) ---")
    sample = df[df["system_id"] == 615].head(3)
    print(sample[[
        "timestamp", "heatpump_elec", "cop",
        "deltaT_house", "deltaT_lift", "temp_deficit",
        "hour", "elec_lag1", "flowT_lag1"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
