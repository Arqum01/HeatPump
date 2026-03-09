"""
FILE 3 — 03_clean_data.py
===========================
PURPOSE : Apply physical boundary rules, interpolation, and selective
          dropna to produce a clean, ML-ready dataset.
INPUT   : data/processed/daikin_features.csv
OUTPUT  : data/processed/daikin_clean.csv
          data/processed/cleaning_report.csv

TWEAKS APPLIED:
  ✅ Smart interpolation (limit=1) — saves single-hour sensor blinks
  ✅ Physical boundary rules (expanded to reflect real-world extremes)
  ✅ COP range 0.0–8.0 (includes defrost cycles + high-efficiency moments)
  ✅ DeltaT ceiling 20°C (captures cold-start high-power events)
  ✅ Outside temp range -20°C to +40°C (UK climate extremes)
  ✅ Selective dropna (only drop if core label columns are missing)
      — Saves System 228 which always has NaN in roomT
  ✅ Data retention report (shows % kept per system)
"""

import pandas as pd
import numpy as np
import os
import logging

# =============================================================
# CONFIGURATION
# =============================================================
INPUT_PATH   = "data/processed/daikin_features.csv"
OUTPUT_PATH  = "data/processed/daikin_clean.csv"
REPORT_PATH  = "data/processed/cleaning_report.csv"

# Physical boundary constants — these are physics limits, not arbitrary
RULES = {
    # Energy — only block physically impossible negatives
    # Note: We allow 0 because standby filter already handled low readings
    "heatpump_elec_min":  0,
    "heatpump_heat_min":  0,

    # COP — 0.0 allows defrost cycles (pump uses elec to melt ice, low heat output)
    #        8.0 allows modern inverter peak efficiency on mild days
    "cop_min":  0.0,
    "cop_max":  8.0,

    # DeltaT_house — 20°C ceiling captures cold-start high-power events
    # Client used 10°C which deleted valid morning startup rows
    "deltaT_house_min":  0.0,
    "deltaT_house_max":  20.0,

    # Outside temperature — UK reality: -18°C cold snaps, 38°C heatwaves
    "outsideT_min": -20,
    "outsideT_max":  40,
}

# Core label columns — ONLY drop row if THESE are missing
# System 228 always has NaN in heatpump_roomT but valid elec/heat data
# dropna() on ALL columns would delete every System 228 row
CORE_COLS = ["heatpump_elec", "heatpump_heat"]

# Temperature columns eligible for interpolation
# We do NOT interpolate electricity or heat — those are our labels
INTERP_COLS = [
    "heatpump_outsideT",
    "heatpump_roomT",
    "heatpump_flowT",
    "heatpump_returnT",
]

os.makedirs("data/processed", exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# =============================================================
# STEP 1 — SMART INTERPOLATION
#
# IoT sensors occasionally drop a single packet ("blink").
# Without this: entire row deleted, valid elec/heat data lost.
# With this: missing temp estimated as midpoint of hour before/after.
#
# limit=1 is the safety guard:
#   1 missing hour → estimate (reliable, straight-line between neighbours)
#   2+ missing hours → keep as NaN (guessing across a 2hr gap is unreliable)
#
# IMPORTANT: interpolate() within each system separately
# =============================================================
def smart_interpolate(df: pd.DataFrame) -> pd.DataFrame:
    before = df[INTERP_COLS].isna().sum().sum()

    df[INTERP_COLS] = (
        df.groupby("system_id")[INTERP_COLS]
        .transform(lambda x: x.interpolate(method="linear", limit=1))
    )

    after = df[INTERP_COLS].isna().sum().sum()
    logging.info(f"Interpolation: filled {before - after} single-blink NaN values.")
    return df


# =============================================================
# STEP 2 — PHYSICAL BOUNDARY RULES
#
# Each rule corresponds to a real physical constraint.
# Rows outside these boundaries are measurement errors or sensor faults.
# =============================================================
def apply_boundary_rules(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    mask = (
        # Block physically impossible negatives
        (df["heatpump_elec"] >= RULES["heatpump_elec_min"]) &
        (df["heatpump_heat"] >= RULES["heatpump_heat_min"]) &

        # COP range — includes defrost (< 1.0) and peak efficiency (> 7.0)
        (df["cop"].isna() | (
            (df["cop"] >= RULES["cop_min"]) &
            (df["cop"] <= RULES["cop_max"])
        )) &

        # DeltaT — 20°C ceiling captures cold-start high-power events
        (df["deltaT_house"] >= RULES["deltaT_house_min"]) &
        (df["deltaT_house"] <= RULES["deltaT_house_max"]) &

        # Outside temperature — real UK weather extremes
        (df["heatpump_outsideT"] >= RULES["outsideT_min"]) &
        (df["heatpump_outsideT"] <= RULES["outsideT_max"])
    )

    df = df[mask].copy()
    after = len(df)
    logging.info(f"Boundary rules: removed {before - after} rows ({(before-after)/before*100:.1f}%)")
    return df


# =============================================================
# STEP 3 — SELECTIVE DROPNA
#
# Client used: clean.dropna()
#   → Deletes ANY row with ANY NaN in ANY column
#   → System 228 has heatpump_roomT = NaN for every single row
#   → Result: entire System 228 dataset wiped ❌
#
# Our fix: only drop if the LABEL columns (elec, heat) are missing
#   → System 228 keeps all its rows ✅
#   → heatpump_roomT stays as NaN → XGBoost handles NaN natively
# =============================================================
def selective_dropna(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=CORE_COLS)
    after = len(df)
    logging.info(f"Selective dropna: removed {before - after} rows missing core label data.")
    return df


# =============================================================
# STEP 4 — DATA QUALITY REPORT
# =============================================================
def generate_report(df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
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


# =============================================================
# MAIN PIPELINE
# =============================================================
def main():
    # Load
    df = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])
    logging.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    df_before = df.copy()

    # Apply cleaning steps in order
    df = smart_interpolate(df)
    df = apply_boundary_rules(df)
    df = selective_dropna(df)

    # Save
    df.to_csv(OUTPUT_PATH, index=False)

    # Report
    report = generate_report(df_before, df)

    total_pct = len(df) / len(df_before) * 100
    logging.info(f"\n✅ Clean dataset saved → {OUTPUT_PATH}")
    logging.info(f"   {len(df_before)} rows → {len(df)} rows ({total_pct:.1f}% retained)")

    print("\n--- Cleaning Report by System ---")
    print(report.to_string(index=False))

    print("\n--- COP Distribution (should include values < 1.0 for defrost) ---")
    print(df["cop"].describe().round(3))

    print("\n--- System 228 Check (roomT should be NaN, elec/heat should have data) ---")
    s228 = df[df["system_id"] == 228][["timestamp", "heatpump_elec", "heatpump_heat", "heatpump_roomT"]].head(5)
    print(s228.to_string(index=False))


if __name__ == "__main__":
    main()
