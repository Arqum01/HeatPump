"""
Fetch Daikin heat pump data from HeatPumpMonitor API and write a data quality report.

Outputs:
- data/raw/daikin_systems_from_api.csv
- data/raw/system_<id>_raw.csv
- data/raw/fetch_quality_report.csv
- data/raw/fetch_quality_audit.csv
- data/raw/fetch_log.csv
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any

import httpx
import pandas as pd
import requests

PUBLIC_SYSTEMS_URL = "https://heatpumpmonitor.org/system/list/public.json"
TARGET_MANUFACTURER = "daikin"

# Set to None to include all Daikin capacities
TARGET_CAPACITY_BUCKETS = {4, 6, 8}

FEEDS = [
    "heatpump_elec",
    "heatpump_heat",
    "heatpump_returnT",
    "heatpump_flowT",
    "heatpump_roomT",
    "heatpump_outsideT",
    "heatpump_flowrate",
]

API_URL = "https://heatpumpmonitor.org/timeseries/data"
RAW_DIR = "data/raw"

START = "01-01-2024"
END = "01-02-2026"
INTERVAL = 3600

os.makedirs(RAW_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def bucket_capacity(hp_output):
    if hp_output is None:
        return None
    try:
        return int(round(float(hp_output)))
    except (TypeError, ValueError):
        return None


def manufacturer_matches(row_manufacturer: str) -> bool:
    text = str(row_manufacturer or "").strip().lower()
    return TARGET_MANUFACTURER in text


def load_systems_from_api():
    response = requests.get(PUBLIC_SYSTEMS_URL, timeout=60)
    response.raise_for_status()
    rows = response.json()

    systems = []
    for row in rows:
        if not manufacturer_matches(row.get("hp_manufacturer")):
            continue

        capacity_kw = bucket_capacity(row.get("hp_output"))
        if TARGET_CAPACITY_BUCKETS is not None and capacity_kw not in TARGET_CAPACITY_BUCKETS:
            continue

        system_id = row.get("id")
        if system_id is None:
            continue

        systems.append(
            {
                "system_id": int(system_id),
                "capacity_kw": capacity_kw,
                "manufacturer": row.get("hp_manufacturer"),
                "hp_output_api": row.get("hp_output"),
                "hp_model": row.get("hp_model"),
                "location": row.get("location"),
            }
        )

    systems.sort(
        key=lambda x: (
            x["capacity_kw"] if x["capacity_kw"] is not None else 999,
            x["system_id"],
        )
    )
    return systems


SYSTEMS = load_systems_from_api()


def coerce_feed_values(values: Any) -> list:
    if values is None:
        return []
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    if isinstance(values, dict):
        return list(values.values())
    return [values]


def normalize_raw_payload(raw_json: Any) -> dict[str, list]:
    if isinstance(raw_json, dict):
        return {feed: coerce_feed_values(raw_json.get(feed, [])) for feed in FEEDS}

    if isinstance(raw_json, list):
        if len(raw_json) == 0:
            return {feed: [] for feed in FEEDS}

        if len(raw_json) == len(FEEDS) and all(isinstance(v, (list, tuple)) for v in raw_json):
            return {
                feed: coerce_feed_values(values)
                for feed, values in zip(FEEDS, raw_json, strict=True)
            }

        if all(isinstance(v, dict) for v in raw_json):
            by_feed = {feed: [] for feed in FEEDS}
            for row in raw_json:
                for feed in FEEDS:
                    by_feed[feed].append(row.get(feed))
            return by_feed

    raise ValueError(f"Unsupported API payload shape: {type(raw_json).__name__}")


def build_time_index(start_date: str, end_date: str) -> pd.DatetimeIndex:
    start_dt = datetime.strptime(start_date, "%d-%m-%Y")
    end_dt = datetime.strptime(end_date, "%d-%m-%Y")
    return pd.date_range(
        start=start_dt,
        end=end_dt,
        freq="1h",
        inclusive="left",
        tz="UTC",
    )


async def fetch_system(client: httpx.AsyncClient, system: dict):
    sid = system["system_id"]
    params = {
        "id": sid,
        "feeds": ",".join(FEEDS),
        "start": START,
        "end": END,
        "interval": INTERVAL,
        "average": 1,
        "timeformat": "notime",
    }

    try:
        response = await client.get(API_URL, params=params, timeout=60.0)
        response.raise_for_status()
        raw_json = response.json()
        logging.info("Fetched system %s successfully.", sid)
        return sid, system, raw_json, None
    except Exception as exc:
        logging.error("Failed system %s: %s", sid, exc)
        return sid, system, None, str(exc)


def build_dataframe(system: dict, raw_json: Any) -> pd.DataFrame:
    time_index = build_time_index(START, END)
    expected_rows = len(time_index)

    normalized_payload = normalize_raw_payload(raw_json)

    df_dict = {}
    for feed in FEEDS:
        values = coerce_feed_values(normalized_payload.get(feed, []))
        values = values[:expected_rows]
        while len(values) < expected_rows:
            values.append(None)
        df_dict[feed] = values

    df = pd.DataFrame(df_dict, index=time_index)
    df.index.name = "timestamp"
    df = df.reset_index()
    df["system_id"] = system["system_id"]
    df["capacity_kw"] = system["capacity_kw"]
    df["manufacturer"] = system.get("manufacturer")
    df["hp_model"] = system.get("hp_model")
    df["location"] = system.get("location")
    return df


def success_quality_row(df: pd.DataFrame, system: dict) -> dict:
    row = {
        "system_id": system["system_id"],
        "capacity_kw": system["capacity_kw"],
        "manufacturer": system.get("manufacturer"),
        "hp_model": system.get("hp_model"),
        "location": system.get("location"),
        "rows": len(df),
    }

    for feed in FEEDS:
        row[f"missing_{feed}_pct"] = round(df[feed].isna().mean() * 100, 2)

    row["overall_missing_pct"] = round(
        df[FEEDS].isna().sum().sum() / (len(df) * len(FEEDS)) * 100, 2
    )
    row["status"] = "SUCCESS"
    row["error"] = ""
    return row


def failed_quality_row(system: dict, error: str) -> dict:
    row = {
        "system_id": system["system_id"],
        "capacity_kw": system["capacity_kw"],
        "manufacturer": system.get("manufacturer"),
        "hp_model": system.get("hp_model"),
        "location": system.get("location"),
        "rows": 0,
        "overall_missing_pct": None,
        "status": "FAILED",
        "error": error,
    }
    for feed in FEEDS:
        row[f"missing_{feed}_pct"] = None
    return row


def build_quality_audit_df(quality_df: pd.DataFrame) -> pd.DataFrame:
    """
    Export compatibility audit format using `series_id` naming.

    This keeps downstream/legacy tooling working while ensuring the audit file
    always reflects the full set of systems processed by main.py.
    """
    missing_cols = [f"missing_{feed}_pct" for feed in FEEDS]
    audit_columns = [
        "series_id",
        "capacity_kw",
        "rows",
        *missing_cols,
        "overall_missing_pct",
        "status",
        "error",
    ]

    if quality_df.empty:
        return pd.DataFrame(columns=audit_columns)

    audit_df = quality_df.rename(columns={"system_id": "series_id"}).copy()

    for col in audit_columns:
        if col not in audit_df.columns:
            audit_df[col] = None

    audit_df = audit_df[audit_columns].sort_values(
        by=["capacity_kw", "series_id"],
        na_position="last",
    )
    return audit_df


async def main():
    print(f"Total systems loaded from API after filtering: {len(SYSTEMS)}")
    for s in SYSTEMS:
        print(s)

    systems_df = pd.DataFrame(SYSTEMS)
    systems_catalog_path = os.path.join(RAW_DIR, "daikin_systems_from_api.csv")
    systems_df.to_csv(systems_catalog_path, index=False)

    quality_report = []
    fetch_log = []

    async with httpx.AsyncClient() as client:
        tasks = [fetch_system(client, system) for system in SYSTEMS]
        results = await asyncio.gather(*tasks)

    for sid, system, raw_json, error in results:
        if raw_json is None:
            quality_report.append(failed_quality_row(system, error))
            fetch_log.append(
                {
                    "system_id": sid,
                    "capacity_kw": system["capacity_kw"],
                    "manufacturer": system.get("manufacturer"),
                    "hp_model": system.get("hp_model"),
                    "rows": 0,
                    "status": "FAILED",
                    "error": error,
                }
            )
            continue

        try:
            df = build_dataframe(system, raw_json)
            out_path = os.path.join(RAW_DIR, f"system_{sid}_raw.csv")
            df.to_csv(out_path, index=False)

            quality_report.append(success_quality_row(df, system))
            fetch_log.append(
                {
                    "system_id": sid,
                    "capacity_kw": system["capacity_kw"],
                    "manufacturer": system.get("manufacturer"),
                    "hp_model": system.get("hp_model"),
                    "rows": len(df),
                    "status": "SUCCESS",
                    "error": "",
                }
            )

            logging.info("Saved %s", out_path)

        except Exception as exc:
            logging.error("Failed to process system %s payload: %s", sid, exc)
            quality_report.append(failed_quality_row(system, str(exc)))
            fetch_log.append(
                {
                    "system_id": sid,
                    "capacity_kw": system["capacity_kw"],
                    "manufacturer": system.get("manufacturer"),
                    "hp_model": system.get("hp_model"),
                    "rows": 0,
                    "status": "FAILED",
                    "error": str(exc),
                }
            )

    quality_df = pd.DataFrame(quality_report)
    log_df = pd.DataFrame(fetch_log)

    if not quality_df.empty:
        quality_df = quality_df.sort_values(
            by=["capacity_kw", "status", "overall_missing_pct", "system_id"],
            na_position="last",
        )

    quality_path = os.path.join(RAW_DIR, "fetch_quality_report.csv")
    audit_path = os.path.join(RAW_DIR, "fetch_quality_audit.csv")
    log_path = os.path.join(RAW_DIR, "fetch_log.csv")

    quality_df.to_csv(quality_path, index=False)
    audit_df = build_quality_audit_df(quality_df)
    audit_df.to_csv(audit_path, index=False)
    log_df.to_csv(log_path, index=False)

    print("\nLoaded systems:", len(SYSTEMS))
    print("Rows in quality report:", len(quality_df))
    print("Rows in fetch log:", len(log_df))

    print(f"\nSaved: {systems_catalog_path}")
    print(f"Saved: {quality_path}")
    print(f"Saved: {audit_path}")
    print(f"Saved: {log_path}")


if __name__ == "__main__":
    asyncio.run(main())