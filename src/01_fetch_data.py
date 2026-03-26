"""Data ingestion stage for the heat pump pipeline.

Concepts:
- Asynchronous collection across multiple systems.
- UTC-aligned hourly calendar for robust time indexing.
- Graceful fallback when max-range API windows exceed provider limits.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd


# Systems and metadata used throughout the pipeline.
SYSTEMS = [
    {"system_id": 615, "capacity_kw": 8},
    {"system_id": 364, "capacity_kw": 8},
    {"system_id": 44, "capacity_kw": 8},
    {"system_id": 162, "capacity_kw": 6},
    {"system_id": 351, "capacity_kw": 6},
    {"system_id": 587, "capacity_kw": 6},
    {"system_id": 228, "capacity_kw": 4},
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
        system: System metadata dictionary containing at least ``system_id``.

    Returns:
        tuple: ``(system_id, system_meta, raw_json_or_none, effective_start, error_or_none)``.
    """
    sid = system["system_id"]
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
    sid = system["system_id"]
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

    df["system_id"] = sid
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
                    "system_id": sid,
                    "capacity_kw": system["capacity_kw"],
                    "rows": len(df),
                    **feed_missing_pct,
                    "status": "SUCCESS",
                }
            )

            fetch_log.append(
                {
                    "system_id": sid,
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
                    "system_id": sid,
                    "capacity_kw": system["capacity_kw"],
                    "rows": 0,
                    **{f"missing_{feed}_pct": 100.0 for feed in FEEDS},
                    "status": "FAILED",
                }
            )

            fetch_log.append(
                {
                    "system_id": sid,
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
