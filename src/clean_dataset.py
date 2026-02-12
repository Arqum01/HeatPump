import pandas as pd

INPUT_PATH = "data/processed/daikin_features_01-01-2026_to_30-01-2026_hourly.csv"
OUTPUT_PATH = "data/processed/daikin_clean_01-01-2026_to_30-01-2026_hourly.csv"

df = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])

print("Original shape:", df.shape)

# -----------------------
# Cleaning rules
# -----------------------

clean = df[
    (df["heatpump_heat"] > 0) &
    (df["heatpump_elec"] > 50) &
    (df["cop"] >= 1.0) &
    (df["cop"] <= 7.0) &
    (df["deltaT"] >= 0.0) &
    (df["deltaT"] <= 10.0) &
    (df["heatpump_outsideT"] >= -15) &
    (df["heatpump_outsideT"] <= 25)
].copy()

print("Clean shape:", clean.shape)

# -----------------------
# Drop rows with any remaining NaNs
# -----------------------
before = clean.shape[0]
clean = clean.dropna()
after = clean.shape[0]

print(f"Dropped {before - after} rows due to NaNs")

# -----------------------
# Save clean dataset
# -----------------------
clean.to_csv(OUTPUT_PATH, index=False)

print("✅ Saved CLEAN dataset to:", OUTPUT_PATH)

# -----------------------
# Quick sanity stats
# -----------------------
print("\nCOP stats:")
print(clean["cop"].describe())

print("\nDeltaT stats:")
print(clean["deltaT"].describe())
