import pandas as pd
import numpy as np

INPUT_PATH = "data/processed/daikin_combined_01-01-2026_to_30-01-2026_hourly.csv"
OUTPUT_PATH = "data/processed/daikin_features_01-01-2026_to_30-01-2026_hourly.csv"

df = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])

print("Loaded shape:", df.shape)

# ----------------------------
# 1) Energy per hour (kWh)
# ----------------------------
df["elec_kwh"] = df["heatpump_elec"] / 1000.0   # W -> kWh (1 hour interval)
df["heat_kwh"] = df["heatpump_heat"] / 1000.0

# ----------------------------
# 2) Temperature difference
# ----------------------------
df["deltaT"] = df["heatpump_flowT"] - df["heatpump_returnT"]

# ----------------------------
# 3) COP (safe calculation)
# ----------------------------
MIN_ELEC_W = 50  # avoid nonsense COP at standby

df["cop"] = np.where(
    df["heatpump_elec"] > MIN_ELEC_W,
    df["heatpump_heat"] / df["heatpump_elec"],
    np.nan
)

# ----------------------------
# 4) Basic sanity flags
# ----------------------------
df["heating_on"] = (df["heatpump_heat"] > 0) & (df["heatpump_elec"] > MIN_ELEC_W)

# ----------------------------
# Save
# ----------------------------
df.to_csv(OUTPUT_PATH, index=False)

print("✅ Saved feature dataset to:", OUTPUT_PATH)
print("Final shape:", df.shape)
print(df[[
    "timestamp",
    "system_id",
    "heatpump_elec",
    "heatpump_heat",
    "cop",
    "elec_kwh",
    "heat_kwh",
    "deltaT"
]].head(5))
