from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
SYSTEM_LIST_PUBLIC_URL = "https://heatpumpmonitor.org/system/list/public.json"
BASE_TEMP_C = 15.5
DEMO_DB_PATH = PROCESSED_DIR / "demo_heatpump_store.sqlite3"
PREDICTION_HISTORY_RETENTION_DAYS = 180
DEMO_SUPPORTED_CAPACITIES = [4, 6, 8]
DEMO_SYSTEMS_BY_CAPACITY: dict[int, list[int]] = {
    4: [147],
    6: [117, 162, 810],
    8: [44, 72, 224],
}
GEMINI_LAYER2_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "outside_t_3h": {"type": "NUMBER"},
        "elec_lag1": {"type": "NUMBER"},
        "elec_lag24": {"type": "NUMBER"},
        "elec_lag168": {"type": "NUMBER"},
        "heat_lag1": {"type": "NUMBER"},
        "heat_lag24": {"type": "NUMBER"},
        "heat_lag168": {"type": "NUMBER"},
        "cop_lag1": {"type": "NUMBER"},
        "heating_on_lag1": {"type": "BOOLEAN"},
        "heating_on_lag24": {"type": "BOOLEAN"},
        "run_hours": {"type": "INTEGER"},
        "rationale": {"type": "STRING"},
    },
    "required": [
        "outside_t_3h",
        "elec_lag1",
        "elec_lag24",
        "elec_lag168",
        "heat_lag1",
        "heat_lag24",
        "heat_lag168",
        "cop_lag1",
        "heating_on_lag1",
        "heating_on_lag24",
        "run_hours",
    ],
}


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def normalize_capacity_profile(capacity_kw: float) -> int:
    return min(DEMO_SUPPORTED_CAPACITIES, key=lambda option: abs(float(capacity_kw) - float(option)))


def ensure_demo_telemetry_store() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DEMO_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS demo_telemetry_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                captured_at TEXT NOT NULL,
                system_id INTEGER NOT NULL,
                capacity_kw INTEGER NOT NULL,
                outside_t REAL NOT NULL,
                outside_t_3h REAL NOT NULL,
                room_t REAL NOT NULL,
                elec_lag1 REAL NOT NULL,
                elec_lag24 REAL NOT NULL,
                elec_lag168 REAL NOT NULL,
                heat_lag1 REAL NOT NULL,
                heat_lag24 REAL NOT NULL,
                heat_lag168 REAL NOT NULL,
                cop_lag1 REAL NOT NULL,
                heating_on_lag1 INTEGER NOT NULL,
                heating_on_lag24 INTEGER NOT NULL,
                run_hours INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_demo_telemetry_lookup
            ON demo_telemetry_snapshots (capacity_kw, captured_at DESC)
            """
        )
        retention_cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        conn.execute(
            "DELETE FROM demo_telemetry_snapshots WHERE captured_at < ?",
            (retention_cutoff,),
        )


def ensure_prediction_history_store() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DEMO_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                run_tag TEXT NOT NULL,
                source_layer1 TEXT NOT NULL,
                source_layer2 TEXT NOT NULL,
                system_id INTEGER NOT NULL,
                capacity_kw INTEGER NOT NULL,
                outside_t REAL NOT NULL,
                room_t REAL NOT NULL,
                outside_t_3h REAL NOT NULL,
                elec_lag1 REAL NOT NULL,
                elec_lag24 REAL NOT NULL,
                elec_lag168 REAL NOT NULL,
                heat_lag1 REAL NOT NULL,
                heat_lag24 REAL NOT NULL,
                heat_lag168 REAL NOT NULL,
                cop_lag1 REAL NOT NULL,
                heating_on_lag1 INTEGER NOT NULL,
                heating_on_lag24 INTEGER NOT NULL,
                run_hours INTEGER NOT NULL,
                pred_heatpump_elec REAL,
                pred_heatpump_heat REAL,
                pred_cop REAL,
                runtime_on_proba REAL,
                cop_guardrail_adjusted INTEGER,
                input_json TEXT NOT NULL,
                layer1_json TEXT NOT NULL,
                layer2_json TEXT NOT NULL,
                prediction_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_prediction_history_capacity_time
            ON prediction_history (capacity_kw, created_at DESC)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_prediction_history_system_time
            ON prediction_history (system_id, created_at DESC)
            """
        )
        retention_cutoff = (datetime.now(timezone.utc) - timedelta(days=PREDICTION_HISTORY_RETENTION_DAYS)).isoformat()
        conn.execute(
            "DELETE FROM prediction_history WHERE created_at < ?",
            (retention_cutoff,),
        )


def _parse_iso_utc(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
        if math.isnan(numeric):
            return default
        return numeric
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return default


def _pick_lag_row(
    rows: list[sqlite3.Row],
    now_utc: datetime,
    target_hours: float,
    min_hours: float,
    max_hours: float,
) -> sqlite3.Row | None:
    best_row: sqlite3.Row | None = None
    best_delta = float("inf")

    for row in rows:
        created_at = _parse_iso_utc(str(row["created_at"]))
        age_hours = (now_utc - created_at).total_seconds() / 3600.0
        if age_hours < min_hours or age_hours > max_hours:
            continue
        delta = abs(age_hours - target_hours)
        if delta < best_delta:
            best_delta = delta
            best_row = row
    return best_row


def get_layer1_inputs_from_prediction_history(
    capacity_kw: float,
    now_utc: datetime,
) -> tuple[dict[str, Any] | None, str]:
    ensure_prediction_history_store()
    profile_capacity = normalize_capacity_profile(capacity_kw)

    with sqlite3.connect(DEMO_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT *
            FROM prediction_history
            WHERE capacity_kw = ?
            ORDER BY created_at DESC
            LIMIT 1000
            """,
            (profile_capacity,),
        ).fetchall()

    if not rows:
        return None, "No historical predictions available yet for this capacity profile."

    lag1_row = _pick_lag_row(rows, now_utc, target_hours=1.0, min_hours=0.08, max_hours=8.0)
    if lag1_row is None:
        lag1_row = rows[0]

    lag24_row = _pick_lag_row(rows, now_utc, target_hours=24.0, min_hours=6.0, max_hours=48.0)
    lag168_row = _pick_lag_row(rows, now_utc, target_hours=168.0, min_hours=72.0, max_hours=336.0)

    if lag24_row is None:
        lag24_row = lag1_row
    if lag168_row is None:
        lag168_row = lag24_row

    recent_outside_vals: list[float] = []
    for row in rows:
        created_at = _parse_iso_utc(str(row["created_at"]))
        age_hours = (now_utc - created_at).total_seconds() / 3600.0
        if 0.0 <= age_hours <= 3.0:
            recent_outside_vals.append(_safe_float(row["outside_t"], default=0.0))
    outside_t_3h = sum(recent_outside_vals) / len(recent_outside_vals) if recent_outside_vals else _safe_float(lag1_row["outside_t_3h"], 0.0)

    layer1_inputs = {
        "outside_t_3h": float(round(outside_t_3h, 2)),
        "elec_lag1": float(_safe_float(lag1_row["elec_lag1"], 0.0)),
        "elec_lag24": float(_safe_float(lag24_row["elec_lag1"], 0.0)),
        "elec_lag168": float(_safe_float(lag168_row["elec_lag1"], 0.0)),
        "heat_lag1": float(_safe_float(lag1_row["heat_lag1"], 0.0)),
        "heat_lag24": float(_safe_float(lag24_row["heat_lag1"], 0.0)),
        "heat_lag168": float(_safe_float(lag168_row["heat_lag1"], 0.0)),
        "cop_lag1": float(_safe_float(lag1_row["cop_lag1"], 0.0)),
        "heating_on_lag1": bool(_safe_int(lag1_row["heating_on_lag1"], 0)),
        "heating_on_lag24": bool(_safe_int(lag24_row["heating_on_lag1"], 0)),
        "run_hours": int(max(0, min(72, _safe_int(lag1_row["run_hours"], 0)))),
        "system_id": int(_safe_int(lag1_row["system_id"], 0)),
    }

    return layer1_inputs, "Using persisted prediction history for lag simulation."


def save_prediction_history_entry(
    run_tag: str,
    user_inputs: dict[str, Any],
    layer1_inputs: dict[str, Any],
    layer2_inputs: dict[str, Any],
    layer_status: dict[str, str],
    scored_row: dict[str, Any],
) -> None:
    ensure_prediction_history_store()

    created_at = _parse_iso_utc(str(user_inputs.get("timestamp", datetime.now(timezone.utc).isoformat()))).isoformat()
    input_json = json.dumps(user_inputs, default=str)
    layer1_json = json.dumps(layer1_inputs, default=str)
    layer2_json = json.dumps(layer2_inputs, default=str)
    prediction_json = json.dumps(scored_row, default=str)

    with sqlite3.connect(DEMO_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO prediction_history (
                created_at,
                run_tag,
                source_layer1,
                source_layer2,
                system_id,
                capacity_kw,
                outside_t,
                room_t,
                outside_t_3h,
                elec_lag1,
                elec_lag24,
                elec_lag168,
                heat_lag1,
                heat_lag24,
                heat_lag168,
                cop_lag1,
                heating_on_lag1,
                heating_on_lag24,
                run_hours,
                pred_heatpump_elec,
                pred_heatpump_heat,
                pred_cop,
                runtime_on_proba,
                cop_guardrail_adjusted,
                input_json,
                layer1_json,
                layer2_json,
                prediction_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                str(run_tag),
                str(layer_status.get("layer1", "unknown")),
                str(layer_status.get("layer2", "unknown")),
                _safe_int(layer2_inputs.get("system_id", user_inputs.get("system_id", 0)), 0),
                _safe_int(user_inputs.get("capacity_kw", 0), 0),
                _safe_float(user_inputs.get("outside_t", 0.0), 0.0),
                _safe_float(user_inputs.get("room_t", 0.0), 0.0),
                _safe_float(layer2_inputs.get("outside_t_3h", user_inputs.get("outside_t", 0.0)), 0.0),
                _safe_float(layer2_inputs.get("elec_lag1", 0.0), 0.0),
                _safe_float(layer2_inputs.get("elec_lag24", 0.0), 0.0),
                _safe_float(layer2_inputs.get("elec_lag168", 0.0), 0.0),
                _safe_float(layer2_inputs.get("heat_lag1", 0.0), 0.0),
                _safe_float(layer2_inputs.get("heat_lag24", 0.0), 0.0),
                _safe_float(layer2_inputs.get("heat_lag168", 0.0), 0.0),
                _safe_float(layer2_inputs.get("cop_lag1", 0.0), 0.0),
                _safe_int(layer2_inputs.get("heating_on_lag1", 0), 0),
                _safe_int(layer2_inputs.get("heating_on_lag24", 0), 0),
                _safe_int(layer2_inputs.get("run_hours", 0), 0),
                _safe_float(scored_row.get("pred_heatpump_elec", 0.0), 0.0),
                _safe_float(scored_row.get("pred_heatpump_heat", 0.0), 0.0),
                _safe_float(scored_row.get("pred_cop", 0.0), 0.0),
                _safe_float(scored_row.get("runtime_on_proba", 0.0), 0.0),
                _safe_int(scored_row.get("cop_guardrail_adjusted", 0), 0),
                input_json,
                layer1_json,
                layer2_json,
                prediction_json,
            ),
        )


def simulate_demo_snapshot(
    capacity_kw: int,
    outside_t_hint: float | None = None,
    room_t_hint: float | None = None,
) -> dict[str, Any]:
    profile_capacity = normalize_capacity_profile(capacity_kw)
    rng = random.Random()

    if outside_t_hint is None:
        outside_t = rng.uniform(-4.0, 15.0)
    else:
        outside_t = outside_t_hint + rng.uniform(-1.1, 1.1)

    if room_t_hint is None:
        room_t = rng.uniform(19.0, 22.0)
    else:
        room_t = room_t_hint + rng.uniform(-0.4, 0.4)

    outside_t = round(max(-20.0, min(25.0, outside_t)), 1)
    room_t = round(max(12.0, min(28.0, room_t)), 1)
    outside_t_3h = round(max(-25.0, min(30.0, outside_t + rng.uniform(-1.4, 1.4))), 1)

    temp_deficit = max(0.0, room_t - outside_t)
    heating_on = 1 if (outside_t < 15.0 or temp_deficit > 4.0) else 0

    base_elec = 60.0 if heating_on == 0 else (190.0 + temp_deficit * 52.0 * (profile_capacity / 6.0))
    elec_lag1 = max(50.0, base_elec + rng.uniform(-65.0, 65.0))
    elec_lag24 = max(50.0, elec_lag1 * rng.uniform(0.84, 1.07))
    elec_lag168 = max(50.0, elec_lag1 * rng.uniform(0.78, 1.03))

    cop_lag1 = 3.7 - max(0.0, 11.0 - outside_t) * 0.07 + rng.uniform(-0.18, 0.18)
    cop_lag1 = max(1.2, min(4.8, cop_lag1))

    heat_lag1 = 0.0 if heating_on == 0 else elec_lag1 * cop_lag1
    heat_lag24 = 0.0 if heating_on == 0 else heat_lag1 * rng.uniform(0.84, 1.02)
    heat_lag168 = 0.0 if heating_on == 0 else heat_lag1 * rng.uniform(0.75, 0.98)

    run_hours = 0
    if heating_on == 1:
        run_hours = int(max(1, min(72, round(2 + temp_deficit * 0.6 + rng.uniform(-1.0, 2.2)))))

    heating_on_lag24 = heating_on
    if heating_on == 1 and rng.random() < 0.12:
        heating_on_lag24 = 0

    system_id = rng.choice(DEMO_SYSTEMS_BY_CAPACITY[profile_capacity])

    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "system_id": int(system_id),
        "capacity_kw": int(profile_capacity),
        "outside_t": float(round(outside_t, 2)),
        "outside_t_3h": float(round(outside_t_3h, 2)),
        "room_t": float(round(room_t, 2)),
        "elec_lag1": float(round(elec_lag1, 2)),
        "elec_lag24": float(round(elec_lag24, 2)),
        "elec_lag168": float(round(elec_lag168, 2)),
        "heat_lag1": float(round(heat_lag1, 2)),
        "heat_lag24": float(round(heat_lag24, 2)),
        "heat_lag168": float(round(heat_lag168, 2)),
        "cop_lag1": float(round(cop_lag1, 3)),
        "heating_on_lag1": int(heating_on),
        "heating_on_lag24": int(heating_on_lag24),
        "run_hours": int(run_hours),
    }


def save_demo_snapshot(snapshot: dict[str, Any]) -> None:
    ensure_demo_telemetry_store()
    with sqlite3.connect(DEMO_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO demo_telemetry_snapshots (
                captured_at,
                system_id,
                capacity_kw,
                outside_t,
                outside_t_3h,
                room_t,
                elec_lag1,
                elec_lag24,
                elec_lag168,
                heat_lag1,
                heat_lag24,
                heat_lag168,
                cop_lag1,
                heating_on_lag1,
                heating_on_lag24,
                run_hours
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(snapshot["captured_at"]),
                int(snapshot["system_id"]),
                int(snapshot["capacity_kw"]),
                float(snapshot["outside_t"]),
                float(snapshot["outside_t_3h"]),
                float(snapshot["room_t"]),
                float(snapshot["elec_lag1"]),
                float(snapshot["elec_lag24"]),
                float(snapshot["elec_lag168"]),
                float(snapshot["heat_lag1"]),
                float(snapshot["heat_lag24"]),
                float(snapshot["heat_lag168"]),
                float(snapshot["cop_lag1"]),
                int(snapshot["heating_on_lag1"]),
                int(snapshot["heating_on_lag24"]),
                int(snapshot["run_hours"]),
            ),
        )


def _row_to_snapshot(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "captured_at": row["captured_at"],
        "system_id": int(row["system_id"]),
        "capacity_kw": int(row["capacity_kw"]),
        "outside_t": float(row["outside_t"]),
        "outside_t_3h": float(row["outside_t_3h"]),
        "room_t": float(row["room_t"]),
        "elec_lag1": float(row["elec_lag1"]),
        "elec_lag24": float(row["elec_lag24"]),
        "elec_lag168": float(row["elec_lag168"]),
        "heat_lag1": float(row["heat_lag1"]),
        "heat_lag24": float(row["heat_lag24"]),
        "heat_lag168": float(row["heat_lag168"]),
        "cop_lag1": float(row["cop_lag1"]),
        "heating_on_lag1": int(row["heating_on_lag1"]),
        "heating_on_lag24": int(row["heating_on_lag24"]),
        "run_hours": int(row["run_hours"]),
    }


def get_demo_snapshot(
    capacity_kw: float,
    force_refresh: bool = False,
    freshness_minutes: int = 30,
) -> dict[str, Any]:
    ensure_demo_telemetry_store()
    profile_capacity = normalize_capacity_profile(capacity_kw)

    row: sqlite3.Row | None
    with sqlite3.connect(DEMO_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT *
            FROM demo_telemetry_snapshots
            WHERE capacity_kw = ?
            ORDER BY captured_at DESC
            LIMIT 1
            """,
            (profile_capacity,),
        ).fetchone()

    if row is not None and not force_refresh:
        snapshot = _row_to_snapshot(row)
        captured_at = datetime.fromisoformat(str(snapshot["captured_at"]))
        if captured_at.tzinfo is None:
            captured_at = captured_at.replace(tzinfo=timezone.utc)
        age_seconds = (datetime.now(timezone.utc) - captured_at).total_seconds()
        if age_seconds <= float(freshness_minutes * 60):
            return snapshot

    outside_t_hint = float(row["outside_t"]) if row is not None else None
    room_t_hint = float(row["room_t"]) if row is not None else None
    snapshot = simulate_demo_snapshot(
        capacity_kw=profile_capacity,
        outside_t_hint=outside_t_hint,
        room_t_hint=room_t_hint,
    )
    save_demo_snapshot(snapshot)
    return snapshot


def build_layer1_inputs_from_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "outside_t_3h": float(snapshot["outside_t_3h"]),
        "elec_lag1": float(snapshot["elec_lag1"]),
        "elec_lag24": float(snapshot["elec_lag24"]),
        "elec_lag168": float(snapshot["elec_lag168"]),
        "heat_lag1": float(snapshot["heat_lag1"]),
        "heat_lag24": float(snapshot["heat_lag24"]),
        "heat_lag168": float(snapshot["heat_lag168"]),
        "cop_lag1": float(snapshot["cop_lag1"]),
        "heating_on_lag1": bool(int(snapshot["heating_on_lag1"])),
        "heating_on_lag24": bool(int(snapshot["heating_on_lag24"])),
        "run_hours": int(snapshot["run_hours"]),
        "system_id": int(snapshot["system_id"]),
    }


def _extract_gemini_text(data: dict[str, Any]) -> str:
    candidates = data.get("candidates", [])
    if not candidates:
        return ""

    first = candidates[0] if isinstance(candidates[0], dict) else {}
    parts = first.get("content", {}).get("parts", [])
    text = "\n".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
    return text


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(line for line in cleaned.splitlines() if not line.strip().startswith("```"))
        cleaned = cleaned.strip()

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidate = cleaned[start : end + 1]
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj

    raise ValueError("Gemini did not return a valid JSON object.")


def call_gemini_structured(
    api_key: str,
    model_name: str,
    prompt: str,
    response_schema: dict[str, Any],
) -> dict[str, Any]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    def _extract_error(resp: requests.Response) -> str:
        try:
            payload = resp.json()
            err = payload.get("error", {}) if isinstance(payload, dict) else {}
            msg = err.get("message")
            if msg:
                return str(msg)
        except Exception:
            pass
        return (resp.text or "Unknown Gemini API error")[:500]

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "topP": 0.9,
            "responseMimeType": "application/json",
            "responseSchema": response_schema,
        },
    }

    response = requests.post(url, json=payload, timeout=80)
    if response.status_code == 400:
        fallback_payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "topP": 0.9,
                "responseMimeType": "application/json",
            },
        }
        response = requests.post(url, json=fallback_payload, timeout=80)

    if not response.ok:
        details = _extract_error(response)
        raise RuntimeError(f"Gemini request failed ({response.status_code}) for model '{model_name}': {details}")

    data = response.json()
    text = _extract_gemini_text(data)
    if not text:
        raise RuntimeError("Gemini returned an empty structured response.")

    return _extract_json_object(text)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def sanitize_layer2_overrides(raw: dict[str, Any], fallback: dict[str, Any]) -> tuple[dict[str, Any], str]:
    merged = dict(fallback)

    numeric_bounds: dict[str, tuple[float, float]] = {
        "outside_t_3h": (-25.0, 30.0),
        "elec_lag1": (0.0, 5000.0),
        "elec_lag24": (0.0, 5000.0),
        "elec_lag168": (0.0, 5000.0),
        "heat_lag1": (0.0, 9000.0),
        "heat_lag24": (0.0, 9000.0),
        "heat_lag168": (0.0, 9000.0),
        "cop_lag1": (0.0, 8.0),
    }

    for key, (lower, upper) in numeric_bounds.items():
        if key not in raw:
            continue
        try:
            value = float(raw[key])
        except Exception:
            continue
        if math.isnan(value):
            continue
        merged[key] = max(lower, min(upper, value))

    if "heating_on_lag1" in raw:
        merged["heating_on_lag1"] = _coerce_bool(raw["heating_on_lag1"])
    if "heating_on_lag24" in raw:
        merged["heating_on_lag24"] = _coerce_bool(raw["heating_on_lag24"])

    if "run_hours" in raw:
        try:
            merged["run_hours"] = int(max(0, min(72, round(float(raw["run_hours"])))) )
        except Exception:
            pass

    if not bool(merged.get("heating_on_lag1", False)):
        merged["run_hours"] = 0
        merged["heat_lag1"] = 0.0
        merged["heat_lag24"] = 0.0
        merged["heat_lag168"] = 0.0

    elec_lag1 = float(merged.get("elec_lag1", 0.0))
    cop_lag1 = float(merged.get("cop_lag1", 0.0))
    if elec_lag1 > 0.0 and cop_lag1 > 0.0 and bool(merged.get("heating_on_lag1", True)):
        expected_heat = elec_lag1 * cop_lag1
        heat_lag1 = float(merged.get("heat_lag1", expected_heat))
        if expected_heat > 0.0 and abs(heat_lag1 - expected_heat) > (0.35 * expected_heat):
            merged["heat_lag1"] = expected_heat

    rationale = str(raw.get("rationale", "")).strip()
    return merged, rationale[:180]


def apply_layer2_gemini_autofill(
    base_inputs: dict[str, Any],
    layer1_inputs: dict[str, Any],
    layer1_source: str = "simulator_db",
    layer1_note: str = "Using simulator telemetry from the demo database.",
) -> tuple[dict[str, Any], dict[str, str]]:
    merged_inputs = dict(layer1_inputs)
    status = {
        "layer1": layer1_source,
        "layer2": "simulator_only",
        "note": layer1_note,
        "error": "",
    }

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    if not api_key:
        return merged_inputs, status

    prompt_context = {
        "home_inputs": {
            "outside_t": float(base_inputs["outside_t"]),
            "room_t": float(base_inputs["room_t"]),
            "capacity_kw": float(base_inputs["capacity_kw"]),
            "hour": int(base_inputs.get("hour", datetime.now().hour)),
            "month": int(base_inputs.get("month", datetime.now().month)),
            "usage_goal": str(base_inputs.get("usage_goal", "Balanced comfort")),
            "home_type": str(base_inputs.get("home_type", "Semi-detached")),
        },
        "layer1_simulated_values": layer1_inputs,
        "constraints": {
            "cop_min": 0.0,
            "cop_max": 8.0,
            "max_electricity_w": 5000,
            "max_heat_w": 9000,
        },
    }

    prompt = (
        "You are assisting a heat-pump forecasting pipeline. "
        "Return only JSON that conforms to the provided schema. "
        "Task: refine lag features for inference while keeping values physically plausible. "
        "Prefer small adjustments from layer1_simulated_values and avoid large jumps.\n\n"
        f"Context:\n{json.dumps(prompt_context, indent=2)}"
    )

    try:
        raw = call_gemini_structured(
            api_key=api_key,
            model_name=model_name,
            prompt=prompt,
            response_schema=GEMINI_LAYER2_SCHEMA,
        )
        merged_inputs, rationale = sanitize_layer2_overrides(raw, layer1_inputs)
        status["layer2"] = "simulator_plus_gemini_structured"
        status["note"] = rationale or "Structured Gemini autofill applied on top of simulator telemetry."
    except Exception as exc:
        status["error"] = str(exc)

    return merged_inputs, status


def inject_ui(appearance_mode: str = "Auto") -> None:
    light_vars = {
        "--ink": "#112633",
        "--muted": "#527182",
        "--brand": "#0b8a7a",
        "--paper": "rgba(255, 255, 255, 0.9)",
        "--paper-strong": "rgba(255, 255, 255, 0.97)",
        "--line": "rgba(17, 38, 51, 0.14)",
        "--glass": "rgba(255, 255, 255, 0.74)",
        "--bg-a": "#eef6fa",
        "--bg-b": "#e9f2f6",
        "--bg-c": "#e3ecf2",
        "--input-bg": "#fbfeff",
        "--input-border": "rgba(22, 68, 85, 0.24)",
        "--settings-title": "#133747",
        "--settings-sub": "#436676",
        "--tab-bg": "rgba(235, 244, 248, 0.95)",
        "--tab-btn": "rgba(255, 255, 255, 0.7)",
        "--tab-btn-text": "#234b58",
    }

    dark_vars = {
        "--ink": "#e9f2f5",
        "--muted": "#9fb6bf",
        "--brand": "#39b8a5",
        "--paper": "rgba(21, 35, 43, 0.86)",
        "--paper-strong": "rgba(18, 31, 39, 0.94)",
        "--line": "rgba(183, 219, 231, 0.2)",
        "--glass": "rgba(17, 30, 38, 0.72)",
        "--bg-a": "#0d1820",
        "--bg-b": "#12232d",
        "--bg-c": "#17313d",
        "--input-bg": "#142630",
        "--input-border": "rgba(160, 203, 218, 0.35)",
        "--settings-title": "#d9ecf1",
        "--settings-sub": "#9fbdc6",
        "--tab-bg": "rgba(22, 37, 45, 0.9)",
        "--tab-btn": "rgba(32, 50, 60, 0.92)",
        "--tab-btn-text": "#d7ebf1",
    }

    def vars_block(values: dict[str, str]) -> str:
        return "\n".join(f"            {k}: {v};" for k, v in values.items())

    active_vars = light_vars
    auto_dark_override = ""
    if appearance_mode == "Dark":
        active_vars = dark_vars
    elif appearance_mode == "Auto":
        auto_dark_override = f"""
        @media (prefers-color-scheme: dark) {{
            :root {{
{vars_block(dark_vars)}
            }}
        }}
        """

    css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Nunito+Sans:wght@400;600;700&display=swap');

        :root {
__ACTIVE_VARS__
        }

__AUTO_DARK_OVERRIDE__

        html, body, [class*="css"] {
            font-family: 'Nunito Sans', sans-serif;
            color: var(--ink);
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(900px 440px at 8% 6%, rgba(11, 138, 122, 0.22), transparent 55%),
                radial-gradient(900px 460px at 92% 10%, rgba(240, 138, 36, 0.2), transparent 58%),
                radial-gradient(1200px 520px at 50% 88%, rgba(77, 134, 206, 0.14), transparent 65%),
                linear-gradient(165deg, var(--bg-a) 0%, var(--bg-b) 45%, var(--bg-c) 100%);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 0.72rem;
            padding-bottom: 2.2rem;
        }

        .settings-shell {
            border: 1px solid var(--line);
            background: var(--glass);
            backdrop-filter: blur(8px);
            border-radius: 14px;
            padding: 0.75rem 0.9rem 0.15rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 10px 24px rgba(10, 35, 48, 0.09);
        }

        .settings-title {
            font-family: 'Sora', sans-serif;
            font-size: 1.02rem;
            margin-bottom: 0.1rem;
            color: var(--settings-title);
        }

        .settings-sub {
            color: var(--settings-sub);
            font-size: 0.9rem;
            margin-bottom: 0.6rem;
        }

        .hero-panel {
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.35);
            padding: 1.35rem 1.45rem;
            margin-bottom: 0.95rem;
            background:
                linear-gradient(132deg, rgba(9, 53, 69, 0.95), rgba(11, 138, 122, 0.9) 57%, rgba(240, 138, 36, 0.87));
            color: #f4fffc;
            box-shadow: 0 18px 34px rgba(8, 38, 50, 0.16);
            animation: rise-in 600ms ease-out;
        }

        .hero-kicker {
            display: inline-block;
            border: 1px solid rgba(255, 255, 255, 0.45);
            background: rgba(255, 255, 255, 0.16);
            border-radius: 999px;
            padding: 0.2rem 0.72rem;
            font-size: 0.78rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            font-weight: 700;
            margin-bottom: 0.6rem;
        }

        .hero-title {
            font-family: 'Sora', sans-serif;
            margin: 0;
            font-size: clamp(1.5rem, 2.5vw, 2.3rem);
            line-height: 1.18;
        }

        .hero-sub {
            margin: 0.45rem 0 0;
            color: #e2fff5;
            font-size: 0.98rem;
            max-width: 760px;
        }

        .trust-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 0.65rem;
            margin: 0.82rem 0 1.08rem;
        }

        .trust-card {
            background: var(--paper);
            border: 1px solid var(--line);
            border-radius: 13px;
            padding: 0.72rem 0.82rem;
            backdrop-filter: blur(6px);
            animation: rise-in 650ms ease-out;
        }

        .trust-label {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            font-weight: 700;
        }

        .trust-value {
            font-family: 'Sora', sans-serif;
            margin-top: 0.3rem;
            font-size: 1.12rem;
            color: var(--ink);
            font-weight: 700;
        }

        .result-shell {
            border: 1px solid var(--line);
            border-radius: 16px;
            background: var(--paper-strong);
            padding: 0.9rem;
            margin-top: 0.6rem;
            box-shadow: 0 11px 26px rgba(12, 40, 52, 0.1);
            animation: card-pop 420ms ease-out;
        }

        .result-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.7rem;
            margin-top: 0.5rem;
        }

        .result-card {
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 0.68rem 0.75rem;
            background: var(--paper-strong);
        }

        .result-label {
            color: var(--muted);
            font-size: 0.77rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            font-weight: 700;
        }

        .result-value {
            margin-top: 0.25rem;
            font-family: 'Sora', sans-serif;
            font-size: 1.2rem;
            color: var(--ink);
            font-weight: 700;
        }

        .result-note {
            margin-top: 0.7rem;
            border-left: 4px solid var(--brand);
            background: rgba(11, 138, 122, 0.08);
            border-radius: 8px;
            padding: 0.56rem 0.65rem;
            color: var(--ink);
            font-size: 0.92rem;
        }

        div[data-testid="stTabs"] {
            border: 1px solid var(--line);
            border-radius: 16px;
            background: var(--paper);
            padding: 0.84rem;
            box-shadow: 0 16px 28px rgba(13, 43, 58, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.7);
        }

        div[data-baseweb="tab-list"] {
            gap: 0.72rem;
            background: var(--tab-bg);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 0.5rem 0.6rem;
            margin-bottom: 0.82rem;
        }

        button[data-baseweb="tab"] {
            border-radius: 8px;
            min-height: 40px;
            min-width: 150px;
            margin: 0.05rem;
            font-weight: 700;
            border: 1px solid var(--line);
            background: var(--tab-btn);
            color: var(--tab-btn-text);
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            border-color: rgba(11, 138, 122, 0.45);
            background: linear-gradient(180deg, #ffffff, #eefbf8 45%, #e4f8f2);
            color: #173c4d;
            box-shadow: 0 4px 11px rgba(11, 78, 95, 0.12);
        }

        div[data-testid="stForm"],
        div[data-testid="stDataFrame"],
        div[data-testid="stAlert"],
        div[data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 12px;
            background: var(--paper-strong);
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        textarea {
            border-radius: 10px !important;
            border: 1px solid var(--input-border) !important;
            background: var(--input-bg) !important;
        }

        div.stButton > button {
            border-radius: 10px;
            font-weight: 700;
            border: 1px solid rgba(11, 138, 122, 0.35);
            color: var(--ink);
            background: linear-gradient(180deg, #ffffff, #ebfaf6);
        }

        div.stButton > button:hover {
            border-color: rgba(240, 138, 36, 0.6);
            transform: translateY(-1px);
            transition: all 180ms ease;
        }

        @keyframes rise-in {
            0% { opacity: 0; transform: translateY(9px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        @keyframes card-pop {
            0% { opacity: 0; transform: scale(0.98) translateY(6px); }
            100% { opacity: 1; transform: scale(1) translateY(0); }
        }

        @media (max-width: 900px) {
            .block-container {
                padding-top: 0.72rem;
                padding-left: 0.85rem;
                padding-right: 0.85rem;
            }
            .hero-panel {
                padding: 1rem 1rem;
                border-radius: 16px;
            }
            button[data-baseweb="tab"] {
                min-width: 120px;
                font-size: 0.88rem;
            }
        }
        </style>
        """

    css = css.replace("__ACTIVE_VARS__", vars_block(active_vars))
    css = css.replace("__AUTO_DARK_OVERRIDE__", auto_dark_override)
    st.markdown(css, unsafe_allow_html=True)



def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fetch_defaults_source_mtime() -> float:
    source_path = SRC_DIR / "01_fetch_data.py"
    if not source_path.exists():
        return -1.0
    return source_path.stat().st_mtime


@st.cache_data(show_spinner=False)
def load_default_series_ids(source_mtime: float) -> list[int]:
    _ = source_mtime
    try:
        fetch_module = load_module(SRC_DIR / "01_fetch_data.py", "fetch_defaults_customer")
        raw_systems = getattr(fetch_module, "DEFAULT_SYSTEMS", [])
    except Exception:
        return []

    ids: list[int] = []
    seen: set[int] = set()
    if not isinstance(raw_systems, list):
        return ids

    for item in raw_systems:
        if not isinstance(item, dict):
            continue
        try:
            sid = int(item.get("series_id"))
        except Exception:
            continue
        if sid in seen:
            continue
        seen.add(sid)
        ids.append(sid)
    return ids


def default_series_id() -> int:
    ids = load_default_series_ids(_fetch_defaults_source_mtime())
    return ids[0] if ids else 1


@st.cache_resource(show_spinner=False)
def get_predict_module():
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    return load_module(SRC_DIR / "05_predict_model.py", "predict_model_05_customer")


def list_manifest_runs() -> list[dict[str, Any]]:
    manifests = sorted(MODELS_DIR.glob("run_manifest_*.json"), reverse=True)
    rows: list[dict[str, Any]] = []

    for path in manifests:
        run_tag = path.stem.replace("run_manifest_", "")
        try:
            payload = read_json(path)
            rows.append(
                {
                    "run_tag": run_tag,
                    "policy": payload.get("feature_policy_mode", "unknown"),
                    "strategy": payload.get("selected_target_strategy", "unknown"),
                    "generated_at": payload.get("generated_at", ""),
                    "metrics": payload.get("metrics", {}),
                    "manifest": payload,
                    "path": path,
                }
            )
        except Exception:
            rows.append(
                {
                    "run_tag": run_tag,
                    "policy": "unknown",
                    "strategy": "unknown",
                    "generated_at": "",
                    "metrics": {},
                    "manifest": {},
                    "path": path,
                }
            )

    return rows


def select_customer_run(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose the newest model bundle silently for customer mode."""
    return manifests[0]


def resolve_feature_schema(manifest: dict[str, Any]) -> dict[str, Any]:
    schema = manifest.get("feature_schema", {})
    if isinstance(schema, dict) and schema.get("required_serving_columns"):
        return schema

    schema_path = manifest.get("feature_schema_path")
    if not schema_path:
        return {}

    path = Path(schema_path)
    if not path.is_absolute():
        path = ROOT / path

    if not path.exists():
        return {}

    try:
        return read_json(path)
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_public_system_list() -> list[dict[str, Any]]:
    response = requests.get(SYSTEM_LIST_PUBLIC_URL, timeout=40)
    response.raise_for_status()
    payload = response.json()

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        systems = payload.get("systems")
        if isinstance(systems, list):
            return systems

    raise ValueError("Unexpected metadata payload format from HeatPumpMonitor API.")


def find_system_metadata(rows: list[dict[str, Any]], system_id: int) -> dict[str, Any] | None:
    for row in rows:
        try:
            if int(row.get("id")) == int(system_id):
                return row
        except Exception:
            continue
    return None


def infer_default(col: str, dtype_label: str) -> Any:
    name = col.lower()

    if dtype_label == "datetime":
        return datetime.now(timezone.utc).isoformat()
    if dtype_label in {"string", "str"}:
        return ""
    if dtype_label in {"bool", "int"}:
        if "day_of_week" in name:
            return datetime.now().weekday()
        if "month" in name:
            return datetime.now().month
        if "hour" in name:
            return datetime.now().hour
        return 0

    if "outside" in name:
        return 6.0
    if "room" in name:
        return 20.0
    if "flow" in name:
        return 35.0
    if "return" in name:
        return 30.0
    if "capacity_kw" in name:
        return 6.0
    if "cop" in name:
        return 3.0
    return 0.0


def coerce_for_dtype(value: Any, dtype_label: str) -> Any:
    if dtype_label in {"int", "bool"}:
        try:
            return int(round(float(value)))
        except Exception:
            return 0

    if dtype_label == "datetime":
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    if dtype_label in {"string", "str"}:
        return "" if value is None else str(value)

    try:
        numeric = float(value)
        if math.isnan(numeric):
            return 0.0
        return numeric
    except Exception:
        return 0.0


def build_single_row(required_cols: list[str], dtype_map: dict[str, str], user_inputs: dict[str, Any]) -> pd.DataFrame:
    row = {col: infer_default(col, dtype_map.get(col, "float")) for col in required_cols}

    outside_t = float(user_inputs["outside_t"])
    outside_t_3h = float(user_inputs.get("outside_t_3h", outside_t))
    room_t = float(user_inputs["room_t"])
    capacity_kw = max(float(user_inputs["capacity_kw"]), 0.1)

    home_type = str(user_inputs.get("home_type", "Semi-detached"))
    usage_goal = str(user_inputs.get("usage_goal", "Balanced comfort"))

    hour = int(user_inputs.get("hour", datetime.now().hour))
    month = int(user_inputs.get("month", datetime.now().month))
    day_of_week = int(user_inputs.get("day_of_week", datetime.now().weekday()))

    home_factor_map = {
        "Apartment": 0.82,
        "Terraced": 0.95,
        "Semi-detached": 1.05,
        "Detached": 1.2,
        "Rural / larger home": 1.34,
    }
    goal_factor_map = {
        "Save money": 0.9,
        "Balanced comfort": 1.0,
        "Maximum comfort": 1.14,
    }
    home_factor = home_factor_map.get(home_type, 1.0)
    goal_factor = goal_factor_map.get(usage_goal, 1.0)

    temp_deficit = room_t - outside_t
    hdh = max(0.0, BASE_TEMP_C - outside_t)
    load_ratio = temp_deficit / capacity_kw

    default_heating_on = 1 if (outside_t < 16.0 or usage_goal == "Maximum comfort") else 0
    heating_on_lag1 = int(user_inputs.get("heating_on_lag1", default_heating_on))
    heating_on_lag24 = int(user_inputs.get("heating_on_lag24", heating_on_lag1))

    default_run_hours = 5 if heating_on_lag1 else 0
    if usage_goal == "Save money":
        default_run_hours = 3 if heating_on_lag1 else 0
    elif usage_goal == "Maximum comfort":
        default_run_hours = 7 if heating_on_lag1 else 0
    run_hours = int(user_inputs.get("run_hours", default_run_hours))

    base_elec = max(90.0, 220.0 + max(temp_deficit, 0.0) * 58.0 * home_factor * goal_factor)
    if heating_on_lag1 == 0:
        base_elec = 50.0

    elec_lag1 = float(user_inputs.get("elec_lag1", base_elec))
    elec_lag24 = float(user_inputs.get("elec_lag24", elec_lag1 * 0.92))
    elec_lag168 = float(user_inputs.get("elec_lag168", elec_lag1 * 0.88))

    cop_base = 3.6 - max(0.0, (12.0 - outside_t)) * 0.07
    if usage_goal == "Save money":
        cop_base += 0.12
    elif usage_goal == "Maximum comfort":
        cop_base -= 0.15
    cop_base = min(4.8, max(1.25, cop_base))

    cop_lag1 = float(user_inputs.get("cop_lag1", cop_base))
    heat_lag1 = float(user_inputs.get("heat_lag1", max(0.0, elec_lag1 * cop_lag1)))
    heat_lag24 = float(user_inputs.get("heat_lag24", heat_lag1 * 0.9))
    heat_lag168 = float(user_inputs.get("heat_lag168", heat_lag1 * 0.82))

    # Leakage-safe defaults for lag temperatures using current thermal context,
    # without asking the customer for direct flow/return values.
    target_lift = min(35.0, max(14.0, temp_deficit + 16.0))
    flow_t = min(60.0, max(25.0, room_t + target_lift))
    return_t = min(55.0, max(15.0, flow_t - 5.0))
    flowrate = 0.0 if heating_on_lag1 == 0 else min(45.0, max(6.0, capacity_kw * 2.8))

    delta_t_house_lag1 = flow_t - return_t
    delta_t_lift_lag1 = flow_t - outside_t
    lift_per_kw_lag1 = delta_t_lift_lag1 / capacity_kw
    elec_lag1_pct = elec_lag1 / (capacity_kw * 1000.0)

    computed: dict[str, Any] = {
        "timestamp": user_inputs["timestamp"],
        "series_id": int(user_inputs.get("system_id", 0)),
        "heatpump_outsideT": outside_t,
        "outsideT_3h_avg": outside_t_3h,
        "heatpump_roomT": room_t,
        "capacity_kw": capacity_kw,
        "hdh": hdh,
        "temp_deficit": temp_deficit,
        "load_ratio": load_ratio,
        "is_heating_season": int(outside_t < BASE_TEMP_C),
        "hour": hour,
        "month": month,
        "day_of_week": day_of_week,
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
        "elec_lag1": elec_lag1,
        "elec_lag2": elec_lag1,
        "elec_lag3": elec_lag1,
        "elec_lag4": elec_lag1,
        "elec_lag6": elec_lag1,
        "elec_lag24": elec_lag24,
        "elec_lag168": elec_lag168,
        "elec_lag1_pct": elec_lag1_pct,
        "flowT_lag1": flow_t,
        "returnT_lag1": return_t,
        "deltaT_house_lag1": delta_t_house_lag1,
        "deltaT_lift_lag1": delta_t_lift_lag1,
        "lift_per_kw_lag1": lift_per_kw_lag1,
        "flowrate_lag1": flowrate,
        "heating_on_lag1": float(heating_on_lag1),
        "heating_on_lag24": float(heating_on_lag24),
        "run_hours": int(run_hours),
        "cop_lag1": cop_lag1,
        "was_defrost_lag1": float(cop_lag1 < 1.0),
        "heat_lag1": heat_lag1,
        "heat_lag24": heat_lag24,
        "heat_lag168": heat_lag168,
    }

    for col, value in computed.items():
        if col in row:
            row[col] = value

    for col in required_cols:
        row[col] = coerce_for_dtype(row[col], dtype_map.get(col, "float"))

    return pd.DataFrame([{col: row[col] for col in required_cols}])


def cop_band(cop_value: float) -> tuple[str, str]:
    if cop_value >= 4.0:
        return "Excellent", "#0f8a62"
    if cop_value >= 3.0:
        return "Strong", "#1b8f7f"
    if cop_value >= 2.0:
        return "Moderate", "#e78a1b"
    if cop_value >= 1.0:
        return "Low", "#d7682f"
    return "Off or defrost", "#b54d43"


def customer_note(cop_value: float, on_probability: float) -> str:
    if on_probability < 0.5:
        return "System is likely idle right now. Electricity is set to standby behavior in the production model."
    if cop_value >= 3.5:
        return "Efficiency looks healthy for current conditions. Your setup is operating in a strong performance zone."
    if cop_value >= 2.2:
        return "Efficiency is acceptable. If comfort allows, lowering flow temperature can often improve COP."
    return "Efficiency is on the low side. Check recent operating history and defrost conditions."


def safe_mean(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    value = pd.to_numeric(df[column], errors="coerce").mean()
    if pd.isna(value):
        return None
    return float(value)


def summarize_prediction_df(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
    }

    summary["predicted_on_rate"] = safe_mean(df, "runtime_on_pred")
    summary["avg_pred_elec_w"] = safe_mean(df, "pred_heatpump_elec")
    summary["avg_pred_heat_w"] = safe_mean(df, "pred_heatpump_heat")
    summary["avg_pred_cop"] = safe_mean(df, "pred_cop")

    if "cop_guardrail_adjusted" in df.columns:
        summary["cop_guardrail_adjusted_count"] = int(pd.to_numeric(df["cop_guardrail_adjusted"], errors="coerce").fillna(0).sum())
    else:
        summary["cop_guardrail_adjusted_count"] = 0

    sample_cols = [c for c in ["timestamp", "pred_heatpump_elec", "pred_heatpump_heat", "pred_cop", "runtime_on_pred"] if c in df.columns]
    sample = df[sample_cols].head(12) if sample_cols else df.head(12)
    summary["sample"] = sample.to_dict(orient="records")
    return summary


def call_gemini(api_key: str, model_name: str, prompt: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    def _extract_error(resp: requests.Response) -> str:
        try:
            payload = resp.json()
            err = payload.get("error", {}) if isinstance(payload, dict) else {}
            msg = err.get("message")
            if msg:
                return str(msg)
        except Exception:
            pass
        return (resp.text or "Unknown Gemini API error")[:500]

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.25,
            "topP": 0.9,
        },
    }

    response = requests.post(url, json=payload, timeout=80)
    if response.status_code == 400:
        fallback_payload = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(url, json=fallback_payload, timeout=80)

    if not response.ok:
        details = _extract_error(response)
        raise RuntimeError(f"Gemini request failed ({response.status_code}) for model '{model_name}': {details}")

    data = response.json()
    text = _extract_gemini_text(data)
    if not text:
        return "Gemini returned no candidates."
    return text or "Gemini returned an empty response."


def render_hero() -> None:
    st.markdown(
        """
        <section class="hero-panel">
            <div class="hero-kicker">Customer Experience</div>
            <h1 class="hero-title">Heat Pump Energy Planner</h1>
            <p class="hero-sub">
                Estimate your heat pump energy use and COP in under one minute.
                Enter simple home details and get clear cost and efficiency guidance.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_model_trust_strip(selected_run: dict[str, Any]) -> None:
    metrics = selected_run.get("metrics", {})
    cards = {
        "Model Run": selected_run.get("run_tag", "n/a")[:18],
        "Policy": selected_run.get("policy", "unknown"),
        "R2 Electricity": f"{float(metrics.get('r2_elec', 0.0)):.3f}",
        "R2 Heat": f"{float(metrics.get('r2_heat', 0.0)):.3f}",
        "Energy Error": f"{float(metrics.get('energy_err_pct', 0.0)):.2f}%",
    }

    cards_html = "".join(
        (
            "<div class='trust-card'>"
            f"<div class='trust-label'>{label}</div>"
            f"<div class='trust-value'>{value}</div>"
            "</div>"
        )
        for label, value in cards.items()
    )
    st.markdown(f"<div class='trust-grid'>{cards_html}</div>", unsafe_allow_html=True)


def render_single_prediction_result(scored: pd.DataFrame, unit_rate: float, currency_symbol: str) -> None:
    row = scored.iloc[0]
    pred_elec_w = float(row.get("pred_heatpump_elec", 0.0))
    pred_heat_w = float(row.get("pred_heatpump_heat", 0.0))
    pred_cop = float(row.get("pred_cop", 0.0))
    on_probability = float(row.get("runtime_on_proba", 0.0))
    hourly_kwh = pred_elec_w / 1000.0
    monthly_kwh = hourly_kwh * 24.0 * 30.0
    annual_kwh = hourly_kwh * 24.0 * 365.0
    monthly_cost = monthly_kwh * unit_rate
    annual_cost = annual_kwh * unit_rate

    band, band_color = cop_band(pred_cop)
    note = customer_note(pred_cop, on_probability)

    st.markdown(
        (
            "<section class='result-shell'>"
            "<strong style='font-family: Sora, sans-serif; font-size: 1.02rem;'>Your Home Estimate</strong>"
            "<div class='result-grid'>"
            "<div class='result-card'><div class='result-label'>Estimated Monthly Cost</div>"
            f"<div class='result-value'>{currency_symbol}{monthly_cost:,.0f}</div></div>"
            "<div class='result-card'><div class='result-label'>Estimated Annual Cost</div>"
            f"<div class='result-value'>{currency_symbol}{annual_cost:,.0f}</div></div>"
            "<div class='result-card'><div class='result-label'>Expected COP</div>"
            f"<div class='result-value'>{pred_cop:.2f}</div></div>"
            "<div class='result-card'><div class='result-label'>Estimated Annual Usage</div>"
            f"<div class='result-value'>{annual_kwh:,.0f} kWh</div></div>"
            "</div>"
            f"<div class='result-note'>Efficiency band: <strong style='color:{band_color};'>{band}</strong>. {note}</div>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("Likely running now", f"{on_probability * 100:.1f}%")
    k2.metric("Estimated daily cost", f"{currency_symbol}{monthly_cost / 30.0:.2f}")
    k3.metric("Current heat output", f"{pred_heat_w:,.0f} W")

    with st.expander("Technical details", expanded=False):
        t1, t2 = st.columns(2)
        t1.metric("Predicted electricity draw", f"{pred_elec_w:,.0f} W")
        t2.metric("COP guardrail triggered", "Yes" if int(row.get("cop_guardrail_adjusted", 0)) > 0 else "No")



def render_instant_estimate_tab(run_tag: str, selected_manifest: dict[str, Any], unit_rate_default: float, currency_symbol: str) -> None:
    st.subheader("Instant Estimate")
    st.caption("Provide the 8 core inputs below.")

    schema = resolve_feature_schema(selected_manifest)
    required_cols = schema.get("required_serving_columns", [])
    dtype_map = schema.get("feature_dtypes", {})

    if not required_cols:
        st.error("Feature schema not found for this model run.")
        return

    now = datetime.now()

    with st.form("customer_single_prediction_form"):
        st.markdown("#### 1) Home and comfort")
        c1, c2, c3 = st.columns(3)
        outside_t = c1.slider("Current outside temperature (C)", min_value=-20.0, max_value=25.0, value=6.0, step=0.5)
        room_t = c2.slider("Current room temperature (C)", min_value=12.0, max_value=28.0, value=20.0, step=0.5)
        capacity_kw = c3.number_input(
            "Heat pump size (kW)",
            min_value=2.0,
            max_value=30.0,
            value=8.0,
            step=0.5,
            format="%.1f",
        )

        st.markdown("#### 2) Household preferences")
        c4, c5, c6 = st.columns(3)
        unit_rate = c4.number_input(
            f"Electricity price ({currency_symbol.strip() or '$'} per kWh)",
            min_value=0.01,
            max_value=2.0,
            value=float(unit_rate_default),
            step=0.01,
        )
        home_type = c5.selectbox(
            "Home type",
            ["Apartment", "Terraced", "Semi-detached", "Detached", "Rural / larger home"],
            index=2,
        )
        usage_goal = c6.selectbox(
            "Usage goal",
            ["Save money", "Balanced comfort", "Maximum comfort"],
            index=1,
        )

        st.markdown("#### 3) Timing")
        t1, t2 = st.columns(2)
        hour = t1.slider("Time of day", min_value=0, max_value=23, value=now.hour, step=1)
        month = t2.selectbox(
            "Current month",
            options=list(range(1, 13)),
            index=max(0, now.month - 1),
            format_func=lambda m: datetime(2025, m, 1).strftime("%B"),
        )

        st.caption("Additional model context is prepared automatically in the background.")

        submitted = st.form_submit_button("Predict Now", type="primary", use_container_width=True)

    if not submitted:
        return

    user_inputs = {
        "timestamp": now.isoformat(),
        "outside_t": outside_t,
        "room_t": room_t,
        "capacity_kw": capacity_kw,
        "home_type": home_type,
        "usage_goal": usage_goal,
        "hour": hour,
        "month": month,
        "day_of_week": now.weekday(),
    }

    now_utc = datetime.now(timezone.utc)
    demo_snapshot: dict[str, Any] | None = None
    layer1_inputs: dict[str, Any] | None
    layer1_source = "prediction_history"
    layer1_note = "Using persisted prediction history for lag simulation."

    layer1_inputs, history_note = get_layer1_inputs_from_prediction_history(
        capacity_kw=float(capacity_kw),
        now_utc=now_utc,
    )
    if layer1_inputs is None:
        try:
            demo_snapshot = get_demo_snapshot(capacity_kw=float(capacity_kw), force_refresh=False)
            layer1_inputs = build_layer1_inputs_from_snapshot(demo_snapshot)
            layer1_source = "simulator_db"
            layer1_note = history_note + " Falling back to simulator telemetry from the demo database."
        except Exception as exc:
            st.error(f"Could not prepare model context from available history: {exc}")
            return

    layer2_inputs, layer_status = apply_layer2_gemini_autofill(
        user_inputs,
        layer1_inputs,
        layer1_source=layer1_source,
        layer1_note=layer1_note,
    )
    user_inputs.update(layer2_inputs)
    user_inputs["system_id"] = int(layer2_inputs.get("system_id", layer1_inputs.get("system_id", 0)))

    try:
        features_df = build_single_row(required_cols, dtype_map, user_inputs)
        predict_module = get_predict_module()
        with st.spinner("Calculating your estimate..."):
            scored = predict_module.predict_bundle(features_df, run_tag=run_tag)
        st.session_state["customer_latest_scored"] = scored
        st.session_state["customer_layer_status"] = layer_status

        if not scored.empty:
            try:
                save_prediction_history_entry(
                    run_tag=run_tag,
                    user_inputs=user_inputs,
                    layer1_inputs=layer1_inputs,
                    layer2_inputs=layer2_inputs,
                    layer_status=layer_status,
                    scored_row=scored.iloc[0].to_dict(),
                )
            except Exception as save_exc:
                st.caption("Prediction completed, but history persistence failed: " + str(save_exc)[:220])
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    render_single_prediction_result(scored, unit_rate=unit_rate, currency_symbol=currency_symbol)

    if layer_status.get("layer2") == "simulator_plus_gemini_structured":
        st.success("Prediction context enhancement is active.")
    else:
        st.info("Prediction context prepared using available baseline data.")

    note = layer_status.get("note", "").strip()
    if note:
        st.caption("Context note: automatic preprocessing applied.")
    if layer_status.get("error"):
        st.caption("An advanced context enhancement step was unavailable, so baseline context was used.")

    with st.expander("View technical details", expanded=False):
        if demo_snapshot is not None:
            st.markdown("Reference context snapshot (read-only)")
            st.json(demo_snapshot)
        else:
            st.markdown("Reference context from recent historical records")
            st.json(layer1_inputs)

        st.markdown("Prepared model input")
        st.dataframe(features_df, use_container_width=True)

        st.markdown("Raw prediction output")
        st.dataframe(scored, use_container_width=True)


def render_batch_tab(run_tag: str, selected_manifest: dict[str, Any]) -> None:
    st.subheader("Batch Upload")
    st.caption("Upload a CSV of model features, score it, and download customer-ready predictions.")

    schema = resolve_feature_schema(selected_manifest)
    required_cols = schema.get("required_serving_columns", [])
    dtype_map = schema.get("feature_dtypes", {})

    if not required_cols:
        st.error("Feature schema not found for this run tag.")
        return

    template_row = {col: infer_default(col, dtype_map.get(col, "float")) for col in required_cols}
    template_df = pd.DataFrame([template_row])

    st.download_button(
        "Download input template CSV",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="customer_input_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="customer_batch_upload")
    if uploaded is None:
        st.info("Upload a CSV to start batch scoring.")
        return

    try:
        input_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read uploaded CSV: {exc}")
        return

    st.write("Input preview")
    st.dataframe(input_df.head(25), use_container_width=True)

    if not st.button("Score Batch", type="primary", use_container_width=True):
        return

    try:
        predict_module = get_predict_module()
        with st.spinner("Scoring uploaded data with selected model bundle..."):
            scored = predict_module.predict_bundle(input_df, run_tag=run_tag)
        st.session_state["customer_latest_scored"] = scored
    except Exception as exc:
        st.error(f"Batch scoring failed: {exc}")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_name = f"customer_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_path = PROCESSED_DIR / output_name
    scored.to_csv(output_path, index=False)

    st.success(f"Batch scoring complete. Saved {output_path.relative_to(ROOT)}")

    avg_elec = safe_mean(scored, "pred_heatpump_elec")
    avg_heat = safe_mean(scored, "pred_heatpump_heat")
    avg_cop = safe_mean(scored, "pred_cop")

    metrics_cols = st.columns(4)
    metrics_cols[0].metric("Rows", f"{len(scored):,}")
    metrics_cols[1].metric("Avg Electricity (W)", f"{avg_elec:.0f}" if avg_elec is not None else "n/a")
    metrics_cols[2].metric("Avg Heat (W)", f"{avg_heat:.0f}" if avg_heat is not None else "n/a")
    metrics_cols[3].metric("Avg COP", f"{avg_cop:.2f}" if avg_cop is not None else "n/a")

    plot_cols = [c for c in ["pred_heatpump_elec", "pred_heatpump_heat", "pred_cop"] if c in scored.columns]
    if plot_cols:
        if "timestamp" in scored.columns:
            chart_df = scored[["timestamp"] + plot_cols].copy()
            chart_df = chart_df.set_index("timestamp")
            st.line_chart(chart_df)
        else:
            st.line_chart(scored[plot_cols])

    st.dataframe(scored.head(80), use_container_width=True)
    st.download_button(
        "Download scored CSV",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name="customer_scored_predictions.csv",
        mime="text/csv",
    )


def render_system_info_tab() -> None:
    st.subheader("System Info")
    st.caption("Find your public HeatPumpMonitor system details by ID.")

    lookup_default_id = default_series_id()
    c1, c2, c3 = st.columns([2, 1, 1])
    system_id = c1.number_input("System ID", min_value=1, value=lookup_default_id, step=1)
    lookup_clicked = c2.button("Lookup", type="primary", use_container_width=True)
    refresh_clicked = c3.button("Refresh cache", use_container_width=True)

    if refresh_clicked:
        fetch_public_system_list.clear()
        st.success("Metadata cache cleared.")

    if not lookup_clicked:
        st.info("Enter a system ID and click Lookup.")
        return

    try:
        systems = fetch_public_system_list()
        record = find_system_metadata(systems, int(system_id))
    except Exception as exc:
        st.error(f"Could not fetch metadata: {exc}")
        return

    if not record:
        st.warning(f"System ID {int(system_id)} was not found in the public list.")
        return

    last_updated = "Unknown"
    try:
        if record.get("last_updated") is not None:
            last_updated = datetime.fromtimestamp(int(record["last_updated"]), timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        last_updated = str(record.get("last_updated"))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("System", str(record.get("id", "Unknown")))
    k2.metric("Manufacturer", str(record.get("hp_manufacturer", "Unknown")))
    k3.metric("Model", str(record.get("hp_model", "Unknown")))
    k4.metric("Output (kW)", str(record.get("hp_output", "Unknown")))

    left, right = st.columns(2)
    with left:
        st.markdown("#### Property")
        st.write(f"Location: {record.get('location', 'Unknown')}")
        st.write(f"Property type: {record.get('property', 'Unknown')}")
        st.write(f"Floor area: {record.get('floor_area', 'Unknown')}")
        st.write(f"Heat loss: {record.get('heat_loss', 'Unknown')}")
        st.write(f"Design temp: {record.get('design_temp', 'Unknown')}")

    with right:
        st.markdown("#### Installation")
        st.write(f"Hydraulic separation: {record.get('hydraulic_separation', 'Unknown')}")
        st.write(f"Typical flow temperature: {record.get('flow_temp_typical', 'Unknown')}")
        st.write(f"Tariff type: {record.get('electricity_tariff_type', 'Unknown')}")
        st.write(f"Installer: {record.get('installer_name', 'Unknown')}")
        st.write(f"Last updated: {last_updated}")

    lat = record.get("latitude")
    lon = record.get("longitude")
    if lat is not None and lon is not None:
        try:
            st.map(pd.DataFrame({"lat": [float(lat)], "lon": [float(lon)]}), zoom=5)
        except Exception:
            pass

    with st.expander("View raw metadata", expanded=False):
        st.json(record)


def render_settings_bar(manifests: list[dict[str, Any]]) -> tuple[str, dict[str, Any], float, str]:
    st.markdown(
        """
        <section class="settings-shell">
            <div class="settings-title">Quick Start</div>
            <div class="settings-sub">Provide 8 core home inputs below for a fast estimate.</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.selectbox(
        "Appearance",
        ["Auto", "Light", "Dark"],
        key="customer_appearance_mode",
    )

    selected_run = select_customer_run(manifests)
    run_tag = str(selected_run.get("run_tag", ""))
    return run_tag, selected_run, 0.28, "£"


def render_gemini_tab(selected_run: dict[str, Any]) -> None:
    st.subheader("AI Briefing")
    st.caption("Choose a topic or enter a custom question, then get a short practical explanation from Gemini.")

    latest_df = st.session_state.get("customer_latest_scored")
    if not isinstance(latest_df, pd.DataFrame) or latest_df.empty:
        st.info("Run Predict Now first, then AI Briefing will personalize guidance from your latest estimate.")
        return

    latest_summary = summarize_prediction_df(latest_df)

    briefing_options = {
        "Explain my estimate": (
            "Explain this estimate in simple terms: what cost, usage, and COP mean for my home right now."
        ),
        "How to reduce my bill": (
            "Give practical actions to lower electricity cost while keeping comfort acceptable."
        ),
        "How to improve efficiency": (
            "Give practical tips to improve COP and system efficiency based on this estimate."
        ),
        "What to monitor next": (
            "Tell me which values I should monitor over the next week and what warning signs to watch for."
        ),
    }

    topic_options = list(briefing_options.keys()) + ["Custom request"]

    selected_topic = st.selectbox(
        "What do you want help with?",
        options=topic_options,
        key="customer_ai_brief_topic",
    )

    custom_request = ""
    if selected_topic == "Custom request":
        custom_request = st.text_area(
            "Enter your custom question",
            value="",
            key="customer_ai_custom_request",
            placeholder="Example: Should I lower my thermostat overnight to save more energy?",
        )

    extra_note = st.text_input(
        "Optional extra note",
        value="",
        key="customer_ai_brief_note",
        placeholder="Example: We prefer lower bills over maximum comfort.",
    )

    context = {
        "run_tag": selected_run.get("run_tag"),
        "policy": selected_run.get("policy"),
        "strategy": selected_run.get("strategy"),
        "metrics": selected_run.get("metrics", {}),
        "latest_prediction_summary": latest_summary or {},
    }

    if st.button("Get AI Briefing", type="primary", use_container_width=True):
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        if not api_key:
            st.warning("AI briefing is currently unavailable.")
            return

        if selected_topic == "Custom request":
            if not custom_request.strip():
                st.warning("Please enter your custom question.")
                return
            user_request = custom_request.strip()
        else:
            user_request = briefing_options[selected_topic]

        if extra_note.strip():
            user_request += f" Additional user note: {extra_note.strip()}"

        full_prompt = (
            "You are an expert heat-pump performance advisor. "
            "Respond clearly for a homeowner while keeping technical accuracy. "
            "Keep the answer concise (max 120 words), practical, and easy to act on.\n\n"
            f"Context:\n{json.dumps(context, indent=2)}\n\n"
            f"User request:\n{user_request}"
        )

        with st.spinner("Gemini is preparing your briefing..."):
            try:
                answer = call_gemini(api_key=api_key, model_name=model_name, prompt=full_prompt)
            except Exception as exc:
                st.error("Gemini request failed. Please try again shortly.")
                st.caption(str(exc))
                return

        st.markdown("### Your AI Briefing")
        st.write(answer)


def main() -> None:
    load_env_file(ROOT / ".env")
    st.set_page_config(page_title="Heat Pump Energy Planner", layout="wide")
    try:
        ensure_demo_telemetry_store()
        ensure_prediction_history_store()
    except Exception as exc:
        st.warning(f"Demo telemetry store is unavailable right now: {exc}")

    if "customer_appearance_mode" not in st.session_state:
        st.session_state["customer_appearance_mode"] = "Auto"
    inject_ui(appearance_mode=st.session_state.get("customer_appearance_mode", "Auto"))

    manifests = list_manifest_runs()
    if not manifests:
        st.error("No model manifests found in models/. Run training stage 04 before opening this app.")
        return

    render_hero()
    run_tag, selected_run, unit_rate_default, currency_symbol = render_settings_bar(manifests)

    render_instant_estimate_tab(
        run_tag=run_tag,
        selected_manifest=selected_run.get("manifest", {}),
        unit_rate_default=unit_rate_default,
        currency_symbol=currency_symbol,
    )

    render_gemini_tab(selected_run=selected_run)

    staff_password = os.getenv("CUSTOMER_STAFF_PASSWORD", "").strip()
    if staff_password:
        with st.expander("Staff Access", expanded=False):
            access_code = st.text_input("Enter staff access code", type="password", key="customer_staff_access_code")

            if access_code:
                if access_code == staff_password:
                    st.success("Staff access unlocked.")
                    tab_batch, tab_system = st.tabs(["Batch Upload", "System Info"])

                    with tab_batch:
                        render_batch_tab(run_tag=run_tag, selected_manifest=selected_run.get("manifest", {}))

                    with tab_system:
                        render_system_info_tab()
                else:
                    st.error("Incorrect access code.")


if __name__ == "__main__":
    main()
