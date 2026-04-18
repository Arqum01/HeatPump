"""Data ingestion stage for the heat pump pipeline.

Concepts:
- Asynchronous collection across multiple systems.
- UTC-aligned hourly calendar for robust time indexing.
- Graceful fallback when max-range API windows exceed provider limits.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd


# Systems and metadata used throughout the pipeline.
# Systems and metadata updated for optimal 2-year data quality.
# Selection criteria: Lowest missing target % and complete roomT/outsideT sensors.
DEFAULT_SYSTEMS = [
    # 8kW Systems
    {"series_id": 44,  "capacity_kw": 8},  # 0.8% overall missing
    {"series_id": 72,  "capacity_kw": 8},  # 1.51% overall missing
    {"series_id": 224, "capacity_kw": 8},  # 18.64% overall missing

    # 6kW Systems
    {"series_id": 162, "capacity_kw": 6},  # 0.72% overall missing
    {"series_id": 117, "capacity_kw": 6},  # 5.62% overall missing
    {"series_id": 810, "capacity_kw": 6},  # 27.1% overall missing

    # 4kW System
    {"series_id": 147, "capacity_kw": 4},  # 10.77% overall missing
]


def parse_system_id_filter_from_env() -> list[int] | None:
    """Parse optional system-id filter from environment.

    Supported variables (first non-empty wins):
    - SYSTEM_IDS
    - SYSTEM_ID_FILTER

    Format:
    - CSV of numeric IDs: 44,72,147
    """
    raw = os.getenv("SYSTEM_IDS", "").strip()
    if not raw:
        raw = os.getenv("SYSTEM_ID_FILTER", "").strip()
    if not raw:
        return None

    ids = []
    seen = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            sid = int(token)
        except ValueError:
            continue
        if sid not in seen:
            seen.add(sid)
            ids.append(sid)

    return ids if ids else None


def parse_systems_from_env() -> list[dict]:
    """Parse optional custom systems from environment variable.

    Supported formats in SYSTEMS_CONFIG:
    - JSON list: [{"series_id": 44, "capacity_kw": 8}, ...]
    - CSV-like: 44:8,72:8,147:4
    """
    raw = os.getenv("SYSTEMS_CONFIG", "").strip()
    id_filter = parse_system_id_filter_from_env()
    if not raw:
        default_capacity_kw = float(os.getenv("DEFAULT_CAPACITY_KW", "6"))
        if not id_filter:
            return DEFAULT_SYSTEMS

        by_id = {item["series_id"]: item for item in DEFAULT_SYSTEMS}
        selected = []
        for sid in id_filter:
            if sid in by_id:
                selected.append(by_id[sid])
            else:
                selected.append({"series_id": sid, "capacity_kw": default_capacity_kw})
        return selected

    try:
        parsed_json = json.loads(raw)
        if isinstance(parsed_json, list):
            systems = []
            for item in parsed_json:
                if not isinstance(item, dict):
                    continue
                sid = int(item["series_id"])
                cap = float(item["capacity_kw"])
                systems.append({"series_id": sid, "capacity_kw": cap})
            if systems:
                if not id_filter:
                    return systems
                allowed = set(id_filter)
                filtered = [s for s in systems if s["series_id"] in allowed]
                return filtered if filtered else systems
    except Exception:
        pass

    systems = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            continue
        sid_str, cap_str = token.split(":", 1)
        try:
            sid = int(sid_str.strip())
            cap = float(cap_str.strip())
            systems.append({"series_id": sid, "capacity_kw": cap})
        except ValueError:
            continue

    if systems:
        if not id_filter:
            return systems
        allowed = set(id_filter)
        filtered = [s for s in systems if s["series_id"] in allowed]
        return filtered if filtered else systems

    default_capacity_kw = float(os.getenv("DEFAULT_CAPACITY_KW", "6"))
    if not id_filter:
        return DEFAULT_SYSTEMS

    by_id = {item["series_id"]: item for item in DEFAULT_SYSTEMS}
    selected = []
    for sid in id_filter:
        if sid in by_id:
            selected.append(by_id[sid])
        else:
            selected.append({"series_id": sid, "capacity_kw": default_capacity_kw})
    return selected


SYSTEMS = parse_systems_from_env()

FEEDS = [
    "heatpump_elec",
    "heatpump_heat",
    "heatpump_returnT",
    "heatpump_flowT",
    "heatpump_roomT",
    "heatpump_outsideT",
    "heatpump_flowrate",
]

# Window strategies:
# - fixed_window: use START and END directly.
# - max_available: attempt a broad historical pull and fallback if needed.
FETCH_MODE = os.getenv("FETCH_MODE", "fixed_window").strip().lower()  # or "max_available"
START = os.getenv("START_DATE", "01-01-2023").strip()
END = os.getenv("END_DATE", "01-02-2026").strip()
VERY_EARLY_START = "01-01-2010"

RAW_DIR = "data/raw"
API_URL = "https://heatpumpmonitor.org/timeseries/data"
LIMIT_ERROR_TOKEN = "request datapoint limit reached"
API_MAX_DATAPOINTS = int(os.getenv("API_MAX_DATAPOINTS", "70000"))

os.makedirs(RAW_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def coerce_feed_values(values) -> list:
    """Normalize arbitrary API feed payloads into a list.

    Args:
        values: Raw feed value from the API response. Can be scalar, tuple,
            dict-like, list-like, or None.

    Returns:
        list: A list representation safe for trimming and padding logic.
    """
    if values is None:
        return []
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    if isinstance(values, dict):
        return list(values.values())
    return [values]


def estimate_requested_datapoints(start_date: str, end_date: str, interval_seconds: int) -> int:
    """Estimate how many interval datapoints a request window will ask for.

    Args:
        start_date: Inclusive start date in ``DD-MM-YYYY`` format.
        end_date: Exclusive end date in ``DD-MM-YYYY`` format.
        interval_seconds: Interval size in seconds.

    Returns:
        int: Estimated datapoint count for one feed in the request window.
    """
    start_dt = datetime.strptime(start_date, "%d-%m-%Y")
    end_dt = datetime.strptime(end_date, "%d-%m-%Y")
    total_seconds = max(0, int((end_dt - start_dt).total_seconds()))
    return total_seconds // max(1, interval_seconds)


def payload_has_explicit_limit_error(raw_json: dict) -> bool:
    """Check for explicit provider limit token in payload values."""
    if not isinstance(raw_json, dict):
        return False

    for feed in FEEDS:
        values = coerce_feed_values(raw_json.get(feed, []))
        for value in values:
            if isinstance(value, str) and LIMIT_ERROR_TOKEN in value.lower():
                return True
    return False


def resolve_date_window() -> tuple[str, str]:
    """Resolve the effective start/end dates based on fetch mode.

    Returns:
        tuple[str, str]: Start and end date strings in ``DD-MM-YYYY`` format.

    Raises:
        ValueError: If ``FETCH_MODE`` is not recognized.
    """
    if FETCH_MODE == "fixed_window":
        return START, END

    if FETCH_MODE == "max_available":
        end_dt = datetime.now(timezone.utc).date() + timedelta(days=1)
        return VERY_EARLY_START, end_dt.strftime("%d-%m-%Y")

    raise ValueError("FETCH_MODE must be 'fixed_window' or 'max_available'.")


RESOLVED_START, RESOLVED_END = resolve_date_window()


async def fetch_system(client: httpx.AsyncClient, system: dict):
    """Fetch telemetry for one system with API-limit fallback handling.

    Args:
        client: Shared async HTTP client.
        system: System metadata dictionary containing at least ``series_id``.

    Returns:
        tuple: ``(series_id, system_meta, raw_json_or_none, effective_start, error_or_none)``.
    """
    sid = system["series_id"]
    params = {
        "id": sid,
        "feeds": ",".join(FEEDS),
        "start": RESOLVED_START,
        "end": RESOLVED_END,
        "interval": 3600,
        "average": 1,
        "timeformat": "notime",
    }
    effective_start = params["start"]

    # Proactively prevent oversized requests by estimating datapoint count.
    estimated_points = estimate_requested_datapoints(
        start_date=params["start"],
        end_date=params["end"],
        interval_seconds=params["interval"],
    )
    if FETCH_MODE == "max_available" and estimated_points > API_MAX_DATAPOINTS:
        effective_start = START
        params["start"] = START
        logging.info(
            "System %s request estimated at %s points (> %s); using fallback start %s.",
            sid,
            estimated_points,
            API_MAX_DATAPOINTS,
            START,
        )

    try:
        resp = await client.get(API_URL, params=params, timeout=60.0)
        resp.raise_for_status()
        raw_json = resp.json()

        # Keep a narrow fallback for explicit provider limit responses.
        if FETCH_MODE == "max_available" and payload_has_explicit_limit_error(raw_json):
            fallback_params = params.copy()
            fallback_params["start"] = START
            effective_start = START
            logging.warning(
                "System %s hit API datapoint limit in max_available mode; retrying from %s.",
                sid,
                START,
            )
            resp = await client.get(API_URL, params=fallback_params, timeout=60.0)
            resp.raise_for_status()
            raw_json = resp.json()

            if payload_has_explicit_limit_error(raw_json):
                raise ValueError("API datapoint limit still triggered after fallback window retry")

        logging.info("System %s downloaded successfully.", sid)
        return sid, system, raw_json, effective_start, None
    except Exception as exc:
        logging.error("System %s failed: %s", sid, exc)
        return sid, system, None, effective_start, str(exc)


def build_dataframe(system: dict, raw_json: dict, data_start: str) -> pd.DataFrame:
    """Build a calendar-anchored DataFrame for a system payload.

    Args:
        system: System metadata dictionary.
        raw_json: API payload containing feed arrays.
        data_start: Effective start date used for this request.

    Returns:
        pd.DataFrame: Hourly UTC-indexed frame with aligned feeds and metadata.
    """
    sid = system["series_id"]
    start_dt = datetime.strptime(data_start, "%d-%m-%Y")
    end_dt = datetime.strptime(RESOLVED_END, "%d-%m-%Y")
    time_index = pd.date_range(
        start=start_dt,
        end=end_dt,
        freq="1h",
        inclusive="left",
        tz="UTC",
    )
    expected_rows = len(time_index)

    df_dict = {}
    for feed in FEEDS:
        values = coerce_feed_values(raw_json.get(feed, []))
        trimmed = values[:expected_rows]
        while len(trimmed) < expected_rows:
            trimmed.append(None)
        df_dict[feed] = trimmed

    df = pd.DataFrame(df_dict, index=time_index)
    df.index.name = "timestamp"
    df = df.reset_index()

    df["series_id"] = sid
    df["capacity_kw"] = system["capacity_kw"]
    if FETCH_MODE == "max_available":
        has_signal = df[["heatpump_elec", "heatpump_heat"]].notna().any(axis=1)
        if has_signal.any():
            first_idx = has_signal.idxmax()
            df = df.loc[first_idx:].reset_index(drop=True)

    return df


async def main():
    """Run full ingestion workflow and persist raw system files.

    The function executes concurrent downloads, applies calendar alignment,
    writes one CSV per system, and emits an audit-style fetch log.
    """
    fetch_log = []
    quality_rows = []

    logging.info(
        "Date window resolved: %s -> %s (mode=%s)",
        RESOLVED_START,
        RESOLVED_END,
        FETCH_MODE,
    )

    async with httpx.AsyncClient() as client:
        tasks = [fetch_system(client, s) for s in SYSTEMS]
        logging.info("Launching parallel download for %s systems...", len(SYSTEMS))
        results = await asyncio.gather(*tasks)

    for sid, system, raw_json, effective_start, error in results:
        if raw_json is not None:
            df = build_dataframe(system, raw_json, effective_start)
            out_path = f"{RAW_DIR}/system_{sid}_raw.csv"
            df.to_csv(out_path, index=False)
            logging.info("System %s saved -> %s | Shape: %s", sid, out_path, df.shape)

            # Compute feed-level missingness for audit visibility before cleaning.
            feed_missing_pct = {}
            for feed in FEEDS:
                feed_missing_pct[f"missing_{feed}_pct"] = round(df[feed].isna().mean() * 100, 2)

            missing_elec_pct = feed_missing_pct["missing_heatpump_elec_pct"]
            missing_heat_pct = feed_missing_pct["missing_heatpump_heat_pct"]

            quality_rows.append(
                {
                    "series_id": sid,
                    "capacity_kw": system["capacity_kw"],
                    "rows": len(df),
                    **feed_missing_pct,
                    "status": "SUCCESS",
                }
            )

            fetch_log.append(
                {
                    "series_id": sid,
                    "capacity_kw": system["capacity_kw"],
                    "rows": len(df),
                    "missing_elec_pct": missing_elec_pct,
                    "missing_heat_pct": missing_heat_pct,
                    "status": "SUCCESS",
                    "error": "",
                }
            )
        else:
            quality_rows.append(
                {
                    "series_id": sid,
                    "capacity_kw": system["capacity_kw"],
                    "rows": 0,
                    **{f"missing_{feed}_pct": 100.0 for feed in FEEDS},
                    "status": "FAILED",
                }
            )

            fetch_log.append(
                {
                    "series_id": sid,
                    "capacity_kw": system["capacity_kw"],
                    "rows": 0,
                    "missing_elec_pct": 100,
                    "missing_heat_pct": 100,

                    "status": "FAILED",
                    "error": error,
                }
            )

    log_df = pd.DataFrame(fetch_log)
    quality_df = pd.DataFrame(quality_rows)
    log_df.to_csv(f"{RAW_DIR}/fetch_log.csv", index=False)
    quality_df.to_csv(f"{RAW_DIR}/fetch_quality_audit.csv", index=False)
    logging.info("Fetch log saved to %s/fetch_log.csv", RAW_DIR)
    logging.info("Fetch quality audit saved to %s/fetch_quality_audit.csv", RAW_DIR)
    print("\n" + log_df.to_string(index=False))
    print("\n--- Data Quality Audit (Missingness % by Feed) ---")
    print(quality_df.to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())

