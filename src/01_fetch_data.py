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
from datetime import datetime, timedelta, timezone
import os
import logging

# =============================================================
# CONFIGURATION — Edit this block to change systems or dates
# =============================================================
SYSTEMS = [
    {"system_id": 615, "capacity_kw": 8, "emitter_type": "RADIATORS", "location_zone": "UNKNOWN"},
    {"system_id": 364, "capacity_kw": 8, "emitter_type": "RADIATORS", "location_zone": "UNKNOWN"},
    {"system_id": 44,  "capacity_kw": 8, "emitter_type": "RADIATORS", "location_zone": "UNKNOWN"},
    {"system_id": 162, "capacity_kw": 6, "emitter_type": "RADIATORS", "location_zone": "UNKNOWN"},
    {"system_id": 351, "capacity_kw": 6, "emitter_type": "RADIATORS", "location_zone": "UNKNOWN"},
    {"system_id": 587, "capacity_kw": 6, "emitter_type": "RADIATORS", "location_zone": "UNKNOWN"},
    {
        "system_id": 228,
        "capacity_kw": 4,
        "emitter_type": "RADIATORS",
        "location_zone": "UNKNOWN",
    },  # Missing roomT — handled via NaN padding
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

# Fetch mode:
# - "fixed_window": use START/END exactly as configured
# - "max_available": fetch from VERY_EARLY_START to tomorrow (UTC)
FETCH_MODE = "max_available"
START = "01-01-2025"
END   = "01-02-2026"
VERY_EARLY_START = "01-01-2010"

RAW_DIR = "data/raw"
API_URL = "https://heatpumpmonitor.org/timeseries/data"

# =============================================================
# SETUP
# =============================================================
os.makedirs(RAW_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def resolve_date_window() -> tuple[str, str]:
    if FETCH_MODE == "fixed_window":
        return START, END

    if FETCH_MODE == "max_available":
        end_dt = datetime.now(timezone.utc).date() + timedelta(days=1)
        return VERY_EARLY_START, end_dt.strftime("%d-%m-%Y")

    raise ValueError("FETCH_MODE must be 'fixed_window' or 'max_available'.")


RESOLVED_START, RESOLVED_END = resolve_date_window()


# =============================================================
# STEP 1 — ASYNC API FETCHER
# Fires all 7 requests in parallel. Total time = slowest single request.
# =============================================================
async def fetch_system(client: httpx.AsyncClient, system: dict):
    sid = system["system_id"]
    params = {
        "id":         sid,
        "feeds":      ",".join(FEEDS),
        "start":      RESOLVED_START,
        "end":        RESOLVED_END,
        "interval":   3600,
        "average":    1,
        "timeformat": "notime",
    }
    try:
        resp = await client.get(API_URL, params=params, timeout=60.0)
        resp.raise_for_status()
        raw_json = resp.json()   # {feed_name: [values...]}
        logging.info(f"✅ System {sid} — downloaded successfully.")
        return sid, system, raw_json, None
    except Exception as e:
        logging.error(f"❌ System {sid} — FAILED: {e}")
        return sid, system, None, str(e)


# =============================================================
# STEP 2 — MASTER CALENDAR + PADDING
#
# The calendar is always authoritative.
# If API sends too little data → pad with None (becomes NaN).
# If API sends too much data   → trim the excess.
# This prevents "Time Drift" where 4PM data shifts into a 2PM slot.
# =============================================================
def build_dataframe(system: dict, raw_json: dict) -> pd.DataFrame:
    sid = system["system_id"]
    # Build the perfect hourly calendar in UTC
    # UTC is CRITICAL — prevents DST ambiguity in spring/autumn
    start_dt = datetime.strptime(RESOLVED_START, "%d-%m-%Y")
    end_dt   = datetime.strptime(RESOLVED_END,   "%d-%m-%Y")
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
    df["capacity_kw"] = system["capacity_kw"]
    df["emitter_type"] = system.get("emitter_type", "UNKNOWN")
    df["location_zone"] = system.get("location_zone", "UNKNOWN")

    if FETCH_MODE == "max_available":
        has_signal = df[["heatpump_elec", "heatpump_heat"]].notna().any(axis=1)
        if has_signal.any():
            first_idx = has_signal.idxmax()
            df = df.loc[first_idx:].reset_index(drop=True)

    return df


# =============================================================
# STEP 3 — MAIN CONTROLLER
# =============================================================
async def main():
    fetch_log = []

    logging.info(
        "Date window resolved: %s → %s (mode=%s)",
        RESOLVED_START,
        RESOLVED_END,
        FETCH_MODE,
    )

    async with httpx.AsyncClient() as client:
        tasks = [fetch_system(client, s) for s in SYSTEMS]
        logging.info(f"🚀 Launching parallel download for {len(SYSTEMS)} systems...")
        results = await asyncio.gather(*tasks)

    for sid, system, raw_json, error in results:
        if raw_json is not None:
            df = build_dataframe(system, raw_json)

            # Save one CSV per system for easy debugging
            out_path = f"{RAW_DIR}/system_{sid}_raw.csv"
            df.to_csv(out_path, index=False)
            logging.info(f"💾 System {sid} saved → {out_path} | Shape: {df.shape}")

            # Log quality
            missing_pct = df["heatpump_elec"].isna().mean() * 100
            fetch_log.append({
                "system_id":   sid,
                "capacity_kw": system["capacity_kw"],
                "emitter_type": system.get("emitter_type", "UNKNOWN"),
                "location_zone": system.get("location_zone", "UNKNOWN"),
                "rows":        len(df),
                "missing_elec_pct": round(missing_pct, 2),
                "status":      "SUCCESS",
                "error":       ""
            })
        else:
            fetch_log.append({
                "system_id":   sid,
                "capacity_kw": system["capacity_kw"],
                "emitter_type": system.get("emitter_type", "UNKNOWN"),
                "location_zone": system.get("location_zone", "UNKNOWN"),
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
