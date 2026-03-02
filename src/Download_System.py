import pandas as pd
import requests
from datetime import datetime

SYSTEM_ID = 615
FEEDS = [
    "heatpump_elec",
    "heatpump_heat",
    "heatpump_returnT",
    "heatpump_flowT",
    "heatpump_roomT",
    "heatpump_outsideT",
    "heatpump_flowrate",
]

START = "01-03-2025"  # dd-mm-yyyy
END = "01-03-2026"    # dd-mm-yyyy

url = "https://heatpumpmonitor.org/timeseries/data"
params = {
    "id": SYSTEM_ID,
    "feeds": ",".join(FEEDS),
    "start": START,
    "end": END,
    "interval": 3600,
    "average": 1,
    "timeformat": "notime",
}

print("Requesting:", url)
print("Params:", params)

r = requests.get(url, params=params, timeout=60)
r.raise_for_status()

# ✅ Parse as JSON (NOT CSV)
data = r.json()  # dict: {feed_name: [values...]}

# Figure out length (how many hourly points)
lengths = {k: len(v) for k, v in data.items()}
n = max(lengths.values()) if lengths else 0
print("Feed lengths:", lengths)
print("Number of rows (hours):", n)

# Build a timestamp index since timeformat=notime doesn't include timestamps
start_dt = datetime.strptime(START, "%d-%m-%Y")
end_dt = datetime.strptime(END, "%d-%m-%Y")

# This creates hourly timestamps from start up to (but not including) end
time_index = pd.date_range(start=start_dt, end=end_dt, freq="1h", inclusive="left")

# If API length differs slightly from our computed range, trim to match n
time_index = time_index[:n]

# Create dataframe
df = pd.DataFrame({k: v[:len(time_index)] for k, v in data.items()}, index=time_index)
df.index.name = "timestamp"

# Save clean CSV
out_path = f"data/Raw/system_{SYSTEM_ID}_{START}_to_{END}_hourly.csv"
df.to_csv(out_path)

print("Saved CLEAN table to:", out_path)
print("Shape:", df.shape)
print(df.head())
