import pandas as pd
import requests
from datetime import datetime

# Daikin systems (from your list)
SYSTEMS = [
    {"system_id": 615, "capacity_kw": 8},
    {"system_id": 364, "capacity_kw": 8},
    {"system_id": 44,  "capacity_kw": 8},
    {"system_id": 162, "capacity_kw": 6},
    {"system_id": 351, "capacity_kw": 6},
    {"system_id": 587, "capacity_kw": 6},
]

FEEDS = [
    "heatpump_elec",
    "heatpump_heat",
    "heatpump_returnT",
    "heatpump_flowT",
    "heatpump_roomT",
    "heatpump_outsideT",
    "heatpump_flowrate",
]

START = "01-01-2026"         # dd-mm-yyyy
END = "30-01-2026"           # dd-mm-yyyy
INTERVAL_SECONDS = 3600      # hourly
AVERAGE = 1
TIMEFORMAT = "notime"

URL = "https://heatpumpmonitor.org/timeseries/data"


def download_json(system_id: int) -> dict:
    params = {
        "id": system_id,
        "feeds": ",".join(FEEDS),
        "start": START,
        "end": END,
        "interval": INTERVAL_SECONDS,
        "average": AVERAGE,
        "timeformat": TIMEFORMAT,
    }
    r = requests.get(URL, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def json_to_dataframe(data: dict) -> pd.DataFrame:
    # data: {feed: [values...]}
    if not data:
        return pd.DataFrame(columns=FEEDS)

    n = max(len(v) for v in data.values())
    start_dt = datetime.strptime(START, "%d-%m-%Y")

    # Build timestamps based on n (so we never get off-by-one issues)
    idx = pd.date_range(
        start=start_dt,
        periods=n,
        freq=pd.to_timedelta(INTERVAL_SECONDS, unit="s")
    )
    idx.name = "timestamp"

    df = pd.DataFrame({k: v[:n] for k, v in data.items()}, index=idx)

    # Ensure all expected feeds exist (missing feeds become NaN)
    for f in FEEDS:
        if f not in df.columns:
            df[f] = pd.NA

    return df[FEEDS]


all_rows = []

for s in SYSTEMS:
    system_id = s["system_id"]
    capacity_kw = s["capacity_kw"]

    print(f"\n--- System {system_id} (Daikin {capacity_kw}kW) ---")

    try:
        data = download_json(system_id)
        lengths = {k: len(v) for k, v in data.items()}
        print("Feed lengths:", lengths)

        df = json_to_dataframe(data)
        df = df.reset_index()  # timestamp becomes a column

        df["system_id"] = system_id
        df["capacity_kw"] = capacity_kw

        print("Saved rows for system:", df.shape[0])
        print(df.head(2))

        all_rows.append(df)

    except Exception as e:
        print(f"❌ Failed system {system_id}: {e}")

if not all_rows:
    raise RuntimeError("No systems downloaded successfully. Nothing to combine.")

combined = pd.concat(all_rows, ignore_index=True)

out_path = f"data/processed/daikin_combined_{START}_to_{END}_hourly.csv"
combined.to_csv(out_path, index=False)

print("\n✅ Combined saved to:", out_path)
print("Combined shape:", combined.shape)
print("Columns:", combined.columns.tolist())
print(combined.head(5))
