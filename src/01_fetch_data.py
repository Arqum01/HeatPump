"""
FILE 1 — 01_fetch_data.py
==========================
PURPOSE : Download raw hourly data for all Daikin systems from HPM API.
OUTPUTS : data/raw/system_{id}_raw.csv  (one file per system)
          data/raw/fetch_log.csv        (success/fail report)

TWEAKS APPLIED:
  ✅ Async parallel download (httpx + asyncio)
  ✅ Master Calendar with UTC timezone
  ✅ Padding logic (calendar-first, never trim clock to fit data)
  ✅ System 228 included (4kW, missing roomT handled via NaN)
  ✅ Correct END date (01-02-2026 captures full January)
  ✅ Directory auto-creation
  ✅ Persistent fetch log
"""

import pandas as pd
import httpx
import asyncio
from datetime import datetime
import os
import logging

# =============================================================
# CONFIGURATION — Edit this block to change systems or dates
# =============================================================
SYSTEMS = [
    {"system_id": 615, "capacity_kw": 8},
    {"system_id": 364, "capacity_kw": 8},
    {"system_id": 44,  "capacity_kw": 8},
    {"system_id": 162, "capacity_kw": 6},
    {"system_id": 351, "capacity_kw": 6},
    {"system_id": 587, "capacity_kw": 6},
    {"system_id": 228, "capacity_kw": 4},   # Missing roomT — handled via NaN padding
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

# NOTE: END = "01-02-2026" NOT "30-01-2026"
# With inclusive="left", end is EXCLUSIVE.
# "30-01-2026" would stop at Jan 29 23:00 — losing 24 hours!
START = "01-01-2025"
END   = "01-02-2026"

RAW_DIR = "data/raw"
API_URL = "https://heatpumpmonitor.org/timeseries/data"

# =============================================================
# SETUP
# =============================================================
os.makedirs(RAW_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# =============================================================
# STEP 1 — ASYNC API FETCHER
# Fires all 7 requests in parallel. Total time = slowest single request.
# =============================================================
async def fetch_system(client: httpx.AsyncClient, system: dict):
    sid = system["system_id"]
    params = {
        "id":         sid,
        "feeds":      ",".join(FEEDS),
        "start":      START,
        "end":        END,
        "interval":   3600,
        "average":    1,
        "timeformat": "notime",
    }
    try:
        resp = await client.get(API_URL, params=params, timeout=60.0)
        resp.raise_for_status()
        raw_json = resp.json()   # {feed_name: [values...]}
        logging.info(f"✅ System {sid} — downloaded successfully.")
        return sid, system["capacity_kw"], raw_json, None
    except Exception as e:
        logging.error(f"❌ System {sid} — FAILED: {e}")
        return sid, system["capacity_kw"], None, str(e)


# =============================================================
# STEP 2 — MASTER CALENDAR + PADDING
#
# The calendar is always authoritative.
# If API sends too little data → pad with None (becomes NaN).
# If API sends too much data   → trim the excess.
# This prevents "Time Drift" where 4PM data shifts into a 2PM slot.
# =============================================================
def build_dataframe(sid: int, capacity: int, raw_json: dict) -> pd.DataFrame:
    # Build the perfect hourly calendar in UTC
    # UTC is CRITICAL — prevents DST ambiguity in spring/autumn
    start_dt = datetime.strptime(START, "%d-%m-%Y")
    end_dt   = datetime.strptime(END,   "%d-%m-%Y")
    time_index = pd.date_range(
        start=start_dt,
        end=end_dt,
        freq="1h",
        inclusive="left",   # end is EXCLUSIVE — see comment at top
        tz="UTC"            # Always UTC — standardises "12:00" across all systems
    )
    expected_rows = len(time_index)

    # Pad/trim each feed to match the calendar exactly
    df_dict = {}
    for feed in FEEDS:
        values = raw_json.get(feed, [])
        trimmed = values[:expected_rows]
        while len(trimmed) < expected_rows:
            trimmed.append(None)          # NaN for missing sensor data
        df_dict[feed] = trimmed

    df = pd.DataFrame(df_dict, index=time_index)
    df.index.name = "timestamp"
    df = df.reset_index()

    # Add metadata columns
    df["system_id"]   = sid
    df["capacity_kw"] = capacity

    return df


# =============================================================
# STEP 3 — MAIN CONTROLLER
# =============================================================
async def main():
    fetch_log = []

    async with httpx.AsyncClient() as client:
        tasks = [fetch_system(client, s) for s in SYSTEMS]
        logging.info(f"🚀 Launching parallel download for {len(SYSTEMS)} systems...")
        results = await asyncio.gather(*tasks)

    for sid, capacity, raw_json, error in results:
        if raw_json is not None:
            df = build_dataframe(sid, capacity, raw_json)

            # Save one CSV per system for easy debugging
            out_path = f"{RAW_DIR}/system_{sid}_raw.csv"
            df.to_csv(out_path, index=False)
            logging.info(f"💾 System {sid} saved → {out_path} | Shape: {df.shape}")

            # Log quality
            missing_pct = df["heatpump_elec"].isna().mean() * 100
            fetch_log.append({
                "system_id":   sid,
                "capacity_kw": capacity,
                "rows":        len(df),
                "missing_elec_pct": round(missing_pct, 2),
                "status":      "SUCCESS",
                "error":       ""
            })
        else:
            fetch_log.append({
                "system_id":   sid,
                "capacity_kw": capacity,
                "rows":        0,
                "missing_elec_pct": 100,
                "status":      "FAILED",
                "error":       error
            })

    # Save the fetch log — persistent audit trail
    log_df = pd.DataFrame(fetch_log)
    log_df.to_csv(f"{RAW_DIR}/fetch_log.csv", index=False)
    logging.info(f"\n📋 Fetch log saved to {RAW_DIR}/fetch_log.csv")
    print("\n" + log_df.to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())
