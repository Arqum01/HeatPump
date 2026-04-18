from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
MODELS_DIR = ROOT / "models"
BASE_TEMP_C = 15.5

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class TechnicalPredictRequest(BaseModel):
    run_tag: str | None = None
    rows: list[dict[str, Any]] = Field(default_factory=list)


class CustomerPredictRequest(BaseModel):
    run_tag: str | None = None
    timestamp: str | None = None
    system_id: int = 0
    outside_t: float = 6.0
    outside_t_3h: float | None = None
    room_t: float = 20.0
    capacity_kw: float = 8.0
    hour: int | None = None
    month: int | None = None
    day_of_week: int | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def get_predict_module():
    return load_module(SRC_DIR / "05_predict_model.py", "predict_model_05_api")


def active_model_dir() -> Path:
    module = get_predict_module()
    model_dir = Path(getattr(module, "MODEL_DIR", MODELS_DIR))
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    return model_dir


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_manifest_runs() -> list[dict[str, Any]]:
    model_dir = active_model_dir()
    manifests = sorted(model_dir.glob("run_manifest_*.json"), reverse=True)
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
                    "manifest": payload,
                }
            )
        except Exception:
            rows.append(
                {
                    "run_tag": run_tag,
                    "policy": "unknown",
                    "strategy": "unknown",
                    "generated_at": "",
                    "manifest": {},
                }
            )

    return rows


def select_run(run_tag: str | None = None) -> dict[str, Any]:
    manifests = list_manifest_runs()
    if not manifests:
        raise HTTPException(status_code=503, detail="No run manifests found. Prepare model bundle first.")

    if run_tag:
        for row in manifests:
            if row["run_tag"] == run_tag:
                return row
        raise HTTPException(status_code=404, detail=f"Unknown run tag: {run_tag}")

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


def infer_default(col: str, dtype_label: str) -> Any:
    name = col.lower()

    if dtype_label == "datetime":
        return datetime.now(UTC).isoformat()
    if dtype_label in {"string", "str"}:
        return ""
    if dtype_label in {"bool", "int"}:
        now = datetime.now()
        if "day_of_week" in name:
            return now.weekday()
        if "month" in name:
            return now.month
        if "hour" in name:
            return now.hour
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
        return 8.0
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


def build_customer_row(required_cols: list[str], dtype_map: dict[str, str], payload: CustomerPredictRequest) -> pd.DataFrame:
    row = {col: infer_default(col, dtype_map.get(col, "float")) for col in required_cols}

    now_local = datetime.now()
    timestamp = payload.timestamp or datetime.now(UTC).isoformat()
    outside_t = float(payload.outside_t)
    outside_t_3h = float(payload.outside_t_3h) if payload.outside_t_3h is not None else outside_t
    room_t = float(payload.room_t)
    capacity_kw = max(float(payload.capacity_kw), 0.1)

    hour = int(payload.hour if payload.hour is not None else now_local.hour)
    month = int(payload.month if payload.month is not None else now_local.month)
    day_of_week = int(payload.day_of_week if payload.day_of_week is not None else now_local.weekday())

    temp_deficit = room_t - outside_t
    hdh = max(0.0, BASE_TEMP_C - outside_t)
    load_ratio = temp_deficit / capacity_kw

    computed: dict[str, Any] = {
        "timestamp": timestamp,
        "series_id": int(payload.system_id),
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
    }

    for key, value in payload.overrides.items():
        computed[key] = value

    for col, value in computed.items():
        if col in row:
            row[col] = value

    for col in required_cols:
        row[col] = coerce_for_dtype(row[col], dtype_map.get(col, "float"))

    return pd.DataFrame([{col: row[col] for col in required_cols}])


def frame_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


app = FastAPI(title="Heat Pump Prediction API", version="1.0.0")

cors_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


@app.get("/")
@app.get("/api")
def root() -> dict[str, Any]:
    run = select_run(None)
    return {
        "service": "heat-pump-prediction-api",
        "status": "ok",
        "active_model_dir": str(active_model_dir()),
        "default_run_tag": run["run_tag"],
        "routes": ["/health", "/runs", "/predict/technical", "/predict/customer"],
    }


@app.get("/health")
@app.get("/api/health")
def health() -> dict[str, Any]:
    run = select_run(None)
    return {
        "status": "ok",
        "run_tag": run["run_tag"],
        "policy": run.get("policy", "unknown"),
        "strategy": run.get("strategy", "unknown"),
    }


@app.get("/runs")
@app.get("/api/runs")
def runs(limit: int = 20) -> dict[str, Any]:
    manifests = list_manifest_runs()
    safe_limit = max(1, min(limit, 100))
    return {
        "count": len(manifests),
        "items": [
            {
                "run_tag": row["run_tag"],
                "policy": row["policy"],
                "strategy": row["strategy"],
                "generated_at": row["generated_at"],
            }
            for row in manifests[:safe_limit]
        ],
    }


@app.post("/predict/technical")
@app.post("/api/predict/technical")
def predict_technical(request: TechnicalPredictRequest) -> dict[str, Any]:
    if not request.rows:
        raise HTTPException(status_code=422, detail="Provide at least one row in rows.")

    selected_run = select_run(request.run_tag)
    predict_module = get_predict_module()

    try:
        input_df = pd.DataFrame(request.rows)
        scored = predict_module.predict_bundle(input_df, run_tag=selected_run["run_tag"])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    return {
        "run_tag": selected_run["run_tag"],
        "rows": int(len(scored)),
        "predictions": frame_to_records(scored),
    }


@app.post("/predict/customer")
@app.post("/api/predict/customer")
def predict_customer(request: CustomerPredictRequest) -> dict[str, Any]:
    selected_run = select_run(request.run_tag)
    schema = resolve_feature_schema(selected_run["manifest"])
    required_cols = schema.get("required_serving_columns", [])
    dtype_map = schema.get("feature_dtypes", {})

    if not required_cols:
        raise HTTPException(status_code=500, detail="Selected run is missing required serving schema.")

    predict_module = get_predict_module()

    try:
        row_df = build_customer_row(required_cols, dtype_map, request)
        scored = predict_module.predict_bundle(row_df, run_tag=selected_run["run_tag"])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    records = frame_to_records(scored)
    return {
        "run_tag": selected_run["run_tag"],
        "prediction": records[0] if records else {},
    }
