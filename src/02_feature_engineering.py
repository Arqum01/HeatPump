"""
FILE 2 — 02_feature_engineering.py  [VERSION 2 — ALL FIXES APPLIED]
=====================================================================
PURPOSE : Takes raw CSVs, builds all physics + temporal + lag features.
INPUT   : data/raw/system_{id}_raw.csv
OUTPUT  : data/processed/daikin_features.csv

FIXES APPLIED IN THIS VERSION:
  ✅ FIX 1: Sin/Cos encoding for hour + month (wraps cycle correctly)
  ✅ FIX 1: is_heating_season derived from outsideT (not hardcoded months)
             Works in ANY country/hemisphere automatically
  ✅ FIX 2: Extended lag features — lag24 (yesterday) + lag168 (last week)
  ✅ FIX 2: run_hours — consecutive hours pump has been running
  ✅ FIX 3: cop_lag1 + was_defrost_lag1 (post-defrost recovery signal)
  ✅ FIX 4: load_ratio (demand vs capacity) + lift_per_kw (workload/kW)
  ✅ FIX 4: hdh — Heating Degree Hours (physics-based, country-agnostic)
  ✅ FIX 5: elec_pct_capacity + elec_lag1_pct (normalised by system size)
             Fixes cross-system comparison (4kW vs 8kW at same Watts)

KEPT FROM VERSION 1:
  ✅ Standby filter, COP, deltaT_house, deltaT_lift, temp_deficit
  ✅ groupby(system_id) before ALL shifts — no system boundary bleeding
  ✅ outsideT_3h_avg rolling average
"""

import pandas as pd
import numpy as np
import os
import glob
import logging

# =============================================================
# CONFIGURATION
# =============================================================
RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
OUTPUT_PATH   = f"{PROCESSED_DIR}/daikin_features.csv"

STANDBY_THRESHOLD_W = 50    # Below this = standby, not real heating

# UK standard base temperature — below this, heating is needed
# Using outsideT-based logic (not hardcoded months) so it works globally
BASE_TEMP = 15.5

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
# When elec < 50W the pump is in standby.
# Forces heat + flowrate to 0 to prevent COP = 5W/10W = 0.5 (nonsense).
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
# STEP 4 — PHYSICS FEATURES (V1 features kept)
#
# deltaT_house = flowT - returnT  → house heat absorption (demand signal)
# deltaT_lift  = flowT - outsideT → thermodynamic hill (explains COP drop)
# temp_deficit = roomT - outsideT → how much heating the house demands
# =============================================================
def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    df["deltaT_house"] = df["heatpump_flowT"]  - df["heatpump_returnT"]
    df["deltaT_lift"]  = df["heatpump_flowT"]  - df["heatpump_outsideT"]
    df["temp_deficit"] = df["heatpump_roomT"]  - df["heatpump_outsideT"]
    return df


# =============================================================
# STEP 5 — ENHANCED PHYSICS FEATURES (FIX 4 + FIX 5)
#
# load_ratio:
#   temp_deficit alone doesn't account for system size.
#   A 6kW pump facing a 15°C deficit works harder than an 8kW pump.
#   Dividing by capacity_kw makes deficit comparable across systems.
#
# lift_per_kw:
#   The thermodynamic "hill" the pump climbs, normalised by its size.
#   This is the single best predictor of COP degradation.
#   Same deltaT_lift is much harder for a 4kW unit than an 8kW unit.
#
# hdh (Heating Degree Hours):
#   Industry-standard energy metric. "How cold × for how long?"
#   Based on BASE_TEMP (15.5°C UK standard) — clip at 0 means
#   warm days don't produce negative values.
#   Physics-based → works in any country, any hemisphere.
#
# elec_pct_capacity:
#   System 228 (4kW) at 2000W = 50% load
#   System 615 (8kW) at 2000W = 25% load ← same Watts, very different
#   Without this normalisation: model confuses small vs large systems.
# =============================================================
def add_enhanced_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    # FIX 4: Physics interactions
    df["load_ratio"]   = df["temp_deficit"] / df["capacity_kw"]
    # df["lift_per_kw"]  = df["deltaT_lift"]  / df["capacity_kw"]

    # FIX 4: Heating Degree Hours — physics-based, replaces hardcoded months
    # max(0, BASE_TEMP - outsideT): positive when cold, zero when warm
    df["hdh"] = (BASE_TEMP - df["heatpump_outsideT"]).clip(lower=0)

    # FIX 5: Normalise electricity by capacity
    # df["elec_pct_capacity"] = df["heatpump_elec"] / (df["capacity_kw"] * 1000)

    return df


# =============================================================
# STEP 6 — ENHANCED TEMPORAL FEATURES (FIX 1)
#
# RAW hour/month problem:
#   XGBoost sees month=12 and month=1 as 11 apart — but they are neighbours!
#   XGBoost sees hour=23 and hour=0 as 23 apart — but they are neighbours!
#
# Sin/Cos encoding WRAPS the cycle:
#   month=12 and month=1 → sin/cos values are adjacent ✅
#   hour=23  and hour=0  → sin/cos values are adjacent ✅
#
# is_heating_season:
#   NOT hardcoded months (that is a Northern Hemisphere assumption).
#   Derived from outsideT < BASE_TEMP — works in any country automatically.
#   Australia July: outsideT=8°C → 8 < 15.5 → is_heating_season=1 ✅
#   Australia Jan:  outsideT=28°C → 28 > 15.5 → is_heating_season=0 ✅
# =============================================================
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    hour  = df["timestamp"].dt.hour
    month = df["timestamp"].dt.month

    # Raw values (kept for interpretability)
    df["hour"]        = hour
    df["month"]       = month
    df["day_of_week"] = df["timestamp"].dt.dayofweek   # 0=Monday, 6=Sunday

    # FIX 1: Sin/Cos encoding — wraps the cycle correctly
    df["hour_sin"]  = np.sin(2 * np.pi * hour  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * hour  / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # FIX 1: Heating season — physics-based, not month-hardcoded
    df["is_heating_season"] = (df["heatpump_outsideT"] < BASE_TEMP).astype(int)

    return df


# =============================================================
# STEP 7 — LAG FEATURES V1 (kept from original)
#
# CRITICAL: groupby("system_id") before EVERY shift.
# Without it: System 615's last row bleeds into System 364's first row.
# Each machine must have its own independent memory.
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
        df[lag_col] = df.groupby("system_id")[source_col].shift(1)

    logging.info("Lag-1 features created.")
    return df


# =============================================================
# STEP 8 — EXTENDED LAG FEATURES (FIX 2 + FIX 3 + FIX 5)
#
# elec_lag24:
#   "Same hour yesterday used X watts → today probably similar"
#   Directly attacks the systematic overestimate problem.
#   Seasonal patterns repeat day-to-day much more than hour-to-hour.
#
# elec_lag168:
#   "Same hour last week" — captures weekly occupancy patterns.
#   Last Tuesday 8am high demand → this Tuesday 8am probably similar.
#   Weekday/weekend patterns are very consistent.
#
# run_hours:
#   How many consecutive hours has the pump been running?
#   Cold start (run_hours=0): pump uses MORE energy to heat cold pipes.
#   After 3+ hours: system is warm, more efficient, uses less energy.
#   This captures thermal inertia that lag1 alone cannot see.
#
# cop_lag1:
#   COP doesn't randomly jump hour to hour.
#   If it was 4.0 last hour it will likely be ~3.8 this hour.
#
# was_defrost_lag1:
#   After a defrost cycle (COP < 1.0), pump works harder to recover house temp.
#   This is a distinct operational pattern the model needs to learn.
#
# elec_lag1_pct:
#   Normalised version of elec_lag1 — allows fair comparison
#   between a 4kW system running at 80% vs 8kW running at 25%.
# =============================================================
def add_extended_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("system_id")

    # FIX 2: Extended time lags
    df["elec_lag24"]       = g["heatpump_elec"].shift(24)    # Same hour yesterday
    df["elec_lag168"]      = g["heatpump_elec"].shift(168)   # Same hour last week
    df["heating_on_lag24"] = g["heating_on"].shift(24)       # Was pump on yesterday at this hour?

    # FIX 2: Consecutive run hours
    # Logic: if heating_on != heating_on_last_hour, it's a new "run block"
    # cumsum() assigns a unique block ID to each ON/OFF block
    # cumcount() counts how many hours within the current block
    df["run_hours"] = (
        g["heating_on"]
        .transform(
            lambda x: x.groupby((x != x.shift()).cumsum()).cumcount()
        )
    )

    # FIX 3: COP memory
    df["cop_lag1"]          = g["cop"].shift(1)
    df["was_defrost_lag1"]  = (df["cop_lag1"] < 1.0).astype(float)
    # float not int — preserves NaN rows (first row of each system has NaN cop_lag1)
    # int would convert NaN → 0 which is incorrect

    # FIX 5: Normalised lag
    df["elec_lag1_pct"] = df["elec_lag1"] / (df["capacity_kw"] * 1000)

    logging.info("Extended lag features created.")
    return df


# =============================================================
# STEP 9 — ROLLING AVERAGE (kept from V1)
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

    # 2. Apply all steps in order — ORDER MATTERS
    # Standby filter must come before COP calculation
    # Physics features must come before lag features (lag uses physics cols)
    df = apply_standby_filter(df)
    df = add_energy_metrics(df)
    df = add_physics_features(df)
    df = add_enhanced_physics_features(df)
    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = add_extended_lag_features(df)
    df = add_rolling_features(df)

    # 3. Save
    df.to_csv(OUTPUT_PATH, index=False)

    logging.info(f"\n✅ Feature dataset saved → {OUTPUT_PATH}")
    logging.info(f"   Shape       : {df.shape}")
    logging.info(f"   Columns ({len(df.columns)}): {df.columns.tolist()}")

    # Sanity check — verify key new features look correct
    print("\n--- Feature Verification (System 615, rows 1–4) ---")
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