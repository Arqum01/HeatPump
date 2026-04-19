from __future__ import annotations

from admin_app import main


if __name__ == "__main__":
    main()
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
SYSTEM_LIST_PUBLIC_URL = "https://heatpumpmonitor.org/system/list/public.json"


STAGE_KIND_OVERRIDES = {
    "monitoring_model_06.py": "callable",
}


def prettify_stage_label(file_name: str) -> str:
    stem = Path(file_name).stem
    cleaned = re.sub(r"^\d+[_-]*", "", stem)
    cleaned = re.sub(r"[_-]+", " ", cleaned).strip()
    if not cleaned:
        cleaned = stem
    return f"{stem.split('_')[0] if stem[:2].isdigit() else ''} {cleaned.title()}".strip()


def stage_sort_key(path: Path) -> tuple[int, str]:
    match = re.match(r"^(\d+)", path.name)
    order = int(match.group(1)) if match else 999
    return order, path.name


def discover_pipeline_stages() -> list[dict[str, str]]:
    if not SRC_DIR.exists():
        return []

    candidates = []
    for py_file in sorted(SRC_DIR.glob("*.py"), key=stage_sort_key):
        is_pipeline_numbered = re.match(r"^\d+_", py_file.name) is not None
        is_special_model_script = py_file.name in {"monitoring_model_06.py", "backtest_model_07.py"}
        if not (is_pipeline_numbered or is_special_model_script):
            continue

        stage_id = py_file.stem.lower()
        kind = STAGE_KIND_OVERRIDES.get(py_file.name, "script")
        candidates.append(
            {
                "id": stage_id,
                "label": prettify_stage_label(py_file.name),
                "kind": kind,
                "path": str((Path("src") / py_file.name).as_posix()),
            }
        )

    return candidates


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


def ensure_state() -> None:
    if "stage_runs" not in st.session_state:
        st.session_state["stage_runs"] = []
    if "latest_scored_df" not in st.session_state:
        st.session_state["latest_scored_df"] = None
    if "single_input_signature" not in st.session_state:
        st.session_state["single_input_signature"] = ""
    if "single_input_row" not in st.session_state:
        st.session_state["single_input_row"] = {}
    if "latest_prediction_summary" not in st.session_state:
        st.session_state["latest_prediction_summary"] = None


def inject_ui(appearance_mode: str = "Auto") -> None:
    light_vars = {
        "--ink": "#10242b",
        "--muted": "#48626b",
        "--teal": "#1f8a70",
        "--ocean": "#0f5f74",
        "--card": "rgba(252, 254, 255, 0.95)",
        "--surface": "#e7edf0",
        "--surface-2": "#dbe5ea",
        "--surface-soft": "rgba(241, 246, 248, 0.58)",
        "--stroke": "rgba(16, 36, 43, 0.16)",
        "--bg-a": "#dfe8ec",
        "--bg-b": "#d7e2e7",
        "--bg-c": "#cedbe1",
        "--widget-bg": "rgba(250, 253, 255, 0.92)",
        "--widget-border": "rgba(12, 46, 57, 0.15)",
        "--input-bg": "#f7fbfd",
        "--input-border": "rgba(13, 53, 66, 0.26)",
        "--tab-shell": "rgba(234, 241, 245, 0.84)",
        "--tab-item": "rgba(237, 244, 247, 0.95)",
        "--tab-active-a": "#ffffff",
        "--tab-active-b": "#f2fbf7",
        "--tab-text": "#22444f",
    }

    dark_vars = {
        "--ink": "#e8f2f5",
        "--muted": "#9fbbc2",
        "--teal": "#49c3a0",
        "--ocean": "#4aa8c7",
        "--card": "rgba(20, 31, 38, 0.92)",
        "--surface": "#1a2a32",
        "--surface-2": "#223741",
        "--surface-soft": "rgba(20, 32, 38, 0.65)",
        "--stroke": "rgba(163, 205, 216, 0.22)",
        "--bg-a": "#0f1a21",
        "--bg-b": "#13232c",
        "--bg-c": "#17313a",
        "--widget-bg": "rgba(22, 34, 42, 0.93)",
        "--widget-border": "rgba(153, 197, 209, 0.22)",
        "--input-bg": "#182933",
        "--input-border": "rgba(142, 191, 204, 0.34)",
        "--tab-shell": "rgba(24, 39, 47, 0.85)",
        "--tab-item": "rgba(31, 48, 58, 0.95)",
        "--tab-active-a": "#1c3138",
        "--tab-active-b": "#224048",
        "--tab-text": "#d3e9ee",
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
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Manrope:wght@400;500;700&display=swap');

        :root {
__ACTIVE_VARS__
        }

__AUTO_DARK_OVERRIDE__

        html, body, [class*="css"] {
            font-family: 'Manrope', sans-serif;
            color: var(--ink);
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 15% 12%, rgba(31, 138, 112, 0.25), transparent 30%),
                radial-gradient(circle at 85% 10%, rgba(15, 95, 116, 0.23), transparent 30%),
                linear-gradient(180deg, var(--bg-a) 0%, var(--bg-b) 45%, var(--bg-c) 100%);
        }

        .block-container {
            max-width: 1320px;
            padding-top: 1.1rem;
            padding-bottom: 3rem;
            background: var(--surface-soft);
            border: 1px solid var(--stroke);
            border-radius: 18px;
        }

        .hero {
            border-radius: 22px;
            padding: 1.4rem 1.5rem;
            margin-bottom: 1rem;
            background: linear-gradient(120deg, rgba(16, 36, 43, 0.95), rgba(15, 95, 116, 0.92));
            color: #f8fffc;
            box-shadow: 0 18px 40px rgba(10, 37, 43, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.1);
            animation: rise 520ms ease-out;
        }

        .hero h1 {
            margin: 0;
            font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(1.7rem, 2vw, 2.35rem);
            letter-spacing: 0.02em;
        }

        .hero p {
            margin: 0.3rem 0 0;
            color: #d9f8ef;
            font-size: 1rem;
        }

        .stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 0.9rem;
            margin: 0.8rem 0 1.2rem;
        }

        .stat-card {
            border-radius: 16px;
            padding: 0.9rem 1rem;
            background: var(--card);
            border: 1px solid var(--stroke);
            backdrop-filter: blur(8px);
            box-shadow: 0 8px 26px rgba(16, 36, 43, 0.08);
            animation: rise 560ms ease-out;
        }

        .stat-label {
            color: var(--muted);
            font-size: 0.84rem;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }

        .stat-value {
            margin-top: 0.32rem;
            font-weight: 700;
            font-size: 1.28rem;
            color: var(--ink);
        }

        .log-card {
            border-radius: 12px;
            background: var(--widget-bg);
            border: 1px solid var(--widget-border);
            padding: 0.75rem 0.85rem;
            margin: 0.35rem 0 0.55rem 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.8rem;
        }

        .log-title {
            font-weight: 700;
            color: var(--ink);
            font-size: 0.95rem;
        }

        .log-sub {
            color: var(--muted);
            font-size: 0.8rem;
        }

        .log-badge {
            border-radius: 999px;
            padding: 0.2rem 0.58rem;
            border: 1px solid transparent;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            white-space: nowrap;
        }

        .log-badge.success {
            background: rgba(40, 167, 69, 0.15);
            border-color: rgba(40, 167, 69, 0.45);
            color: #2ca14a;
        }

        .log-badge.failed {
            background: rgba(220, 53, 69, 0.14);
            border-color: rgba(220, 53, 69, 0.45);
            color: #cf3f53;
        }

        div[data-testid="stTabs"] {
            background: var(--tab-shell);
            border: 1px solid var(--stroke);
            border-radius: 14px;
            padding: 0.7rem 0.8rem 1rem 0.8rem;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7);
            margin-top: 0.35rem;
        }

        div[data-baseweb="tab-list"] {
            background: var(--surface-2);
            border-radius: 10px;
            padding: 0.38rem;
            gap: 0.35rem;
            border: 1px solid var(--stroke);
            margin-bottom: 0.85rem;
            overflow-x: auto;
        }

        button[data-baseweb="tab"] {
            background: var(--tab-item);
            border-radius: 8px;
            color: var(--tab-text);
            border: 1px solid transparent;
            font-weight: 600;
            min-height: 42px;
            min-width: 132px;
            padding: 0.45rem 0.95rem;
            font-size: 0.92rem;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(180deg, var(--tab-active-a), var(--tab-active-b));
            border: 1px solid rgba(31, 138, 112, 0.45);
            color: var(--ink);
            box-shadow: 0 3px 10px rgba(13, 61, 74, 0.14);
        }

        div[data-testid="stForm"],
        div[data-testid="stExpander"],
        div[data-testid="stDataFrame"],
        div[data-testid="stAlert"],
        div[data-testid="stMetric"] {
            background: var(--widget-bg);
            border: 1px solid var(--widget-border);
            border-radius: 12px;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        textarea {
            background: var(--input-bg) !important;
            border: 1px solid var(--input-border) !important;
            border-radius: 10px !important;
            color: var(--ink) !important;
        }

        div[data-testid="stVerticalBlock"] div.stButton > button {
            border-radius: 12px;
            border: 1px solid rgba(31, 138, 112, 0.28);
            background: linear-gradient(180deg, var(--tab-active-a), var(--tab-active-b));
            color: var(--ink);
            font-weight: 600;
        }

        div[data-testid="stVerticalBlock"] div.stButton > button:hover {
            border-color: #1f8a70;
            color: var(--ink);
            transform: translateY(-1px);
            transition: all 180ms ease;
        }

        @keyframes rise {
            0% { opacity: 0; transform: translateY(8px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        div[data-testid="stTabs"] [data-testid="stVerticalBlock"] > div {
            margin-bottom: 0.45rem;
        }
        </style>
        """
    css = css.replace("__ACTIVE_VARS__", vars_block(active_vars))
    css = css.replace("__AUTO_DARK_OVERRIDE__", auto_dark_override)
    st.markdown(css, unsafe_allow_html=True)


def latest_file(path: Path, pattern: str) -> Path | None:
    files = sorted(path.glob(pattern))
    return files[-1] if files else None


def list_manifest_runs() -> list[dict[str, str]]:
    manifests = sorted(MODELS_DIR.glob("run_manifest_*.json"), reverse=True)
    rows: list[dict[str, str]] = []

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
                }
            )
        except Exception:
            rows.append(
                {
                    "run_tag": run_tag,
                    "policy": "unknown",
                    "strategy": "unknown",
                    "generated_at": "",
                }
            )
    return rows


def select_manifest_run(
    key_prefix: str,
    run_label: str = "Model run",
) -> tuple[dict[str, str] | None, dict[str, Any] | None]:
    """Render a run selector so users can choose policy/run instead of always latest."""
    manifests = list_manifest_runs()
    if not manifests:
        st.warning("No run manifests available yet. Run training stage first.")
        return None, None

    policy_mode = st.radio(
        "Feature policy",
        ["enhanced_onestep", "strict_production", "all"],
        horizontal=True,
        key=f"{key_prefix}_policy_mode",
    )

    filtered_manifests = manifests
    if policy_mode != "all":
        filtered_manifests = [m for m in manifests if m.get("policy") == policy_mode]

    if not filtered_manifests:
        st.warning("No runs found for this policy.")
        return None, None

    run_tag = st.selectbox(
        run_label,
        options=[m["run_tag"] for m in filtered_manifests],
        format_func=lambda t: next(
            (
                f"{m['run_tag']} | {m['policy']} | {m['strategy']}"
                for m in filtered_manifests
                if m["run_tag"] == t
            ),
            t,
        ),
        key=f"{key_prefix}_run_tag",
    )

    selected_run = next((m for m in filtered_manifests if m["run_tag"] == run_tag), filtered_manifests[0])

    manifest_path = MODELS_DIR / f"run_manifest_{selected_run['run_tag']}.json"
    try:
        selected_manifest = read_json(manifest_path)
    except Exception as exc:
        st.error(f"Could not load manifest for run '{selected_run['run_tag']}': {exc}")
        return selected_run, None

    return selected_run, selected_manifest


def summarize_prediction_df(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
    }

    def safe_mean(series_name: str) -> float | None:
        if series_name in df.columns:
            return float(pd.to_numeric(df[series_name], errors="coerce").mean())
        return None

    on_col = "runtime_on_pred" if "runtime_on_pred" in df.columns else None
    elec_col = "pred_heatpump_elec" if "pred_heatpump_elec" in df.columns else ("pred_elec_w" if "pred_elec_w" in df.columns else None)
    heat_col = "pred_heatpump_heat" if "pred_heatpump_heat" in df.columns else ("pred_heat_w" if "pred_heat_w" in df.columns else None)
    cop_col = "pred_cop" if "pred_cop" in df.columns else None

    summary["predicted_on_rate"] = safe_mean(on_col) if on_col else None
    summary["avg_pred_elec_w"] = safe_mean(elec_col) if elec_col else None
    summary["avg_pred_heat_w"] = safe_mean(heat_col) if heat_col else None
    summary["avg_pred_cop"] = safe_mean(cop_col) if cop_col else None

    if "cop_guardrail_adjusted" in df.columns:
        summary["cop_guardrail_adjusted_count"] = int(pd.to_numeric(df["cop_guardrail_adjusted"], errors="coerce").fillna(0).sum())
    else:
        summary["cop_guardrail_adjusted_count"] = 0

    sample_cols = [c for c in ["timestamp", on_col, elec_col, heat_col, cop_col] if c and c in df.columns]
    sample_view = df[sample_cols].head(12) if sample_cols else df.head(12)
    summary["sample"] = sample_view.to_dict(orient="records")
    return summary


def build_prediction_expert_prompt(summary: dict[str, Any], user_request: str) -> str:
    return (
        "You are a senior data scientist and heat pump performance analyst. "
        "Analyze prediction outputs as an expert and provide practical, technical guidance.\n\n"
        "Answer in 150 words.\n\n"
        f"Prediction summary context:\n{json.dumps(summary, indent=2)}\n\n"
        f"Additional user request:\n{user_request}"
    )


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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

    def _looks_incomplete(text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        # Common cut-off endings from token limits.
        bad_endings = ("(", "[", "{", ":", ",", "-", "and", "or", "with", "for")
        if t.endswith(bad_endings):
            return True
        # If final character is not terminal punctuation, likely truncated.
        if t[-1] not in ".!?)]}\"'":
            return True
        return False

    def _post_with_optional_fallback(base_prompt: str) -> dict[str, Any]:
        payload = {
            "contents": [{"parts": [{"text": base_prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.9,
            },
        }

        response = requests.post(url, json=payload, timeout=80)
        if response.status_code == 400:
            fallback_payload = {"contents": [{"parts": [{"text": base_prompt}]}]}
            response = requests.post(url, json=fallback_payload, timeout=80)

        if not response.ok:
            details = _extract_error(response)
            raise RuntimeError(
                f"Gemini API request failed ({response.status_code}) for model '{model_name}': {details}"
            )

        return response.json()

    def _extract_text_and_finish_reason(data: dict[str, Any]) -> tuple[str, str]:
        candidates = data.get("candidates", [])
        if not candidates:
            return "", "NO_CANDIDATE"
        first = candidates[0] if isinstance(candidates[0], dict) else {}
        parts = first.get("content", {}).get("parts", [])
        text = "\n".join(p.get("text", "") for p in parts).strip()
        finish_reason = str(first.get("finishReason", "UNKNOWN"))
        return text, finish_reason

    first_data = _post_with_optional_fallback(prompt)
    text, finish_reason = _extract_text_and_finish_reason(first_data)

    if not text:
        return "Gemini returned no candidates."

    # If output is cut by token limit or ends mid-sentence, request continuation(s).
    continuation_attempts = 0
    while continuation_attempts < 2 and (finish_reason == "MAX_TOKENS" or _looks_incomplete(text)):
        continuation_prompt = (
            "Continue the previous answer from exactly where it stopped. "
            "Do not restart and do not repeat. Finish the incomplete sentence first, "
            "then continue briefly in the same structure.\n\n"
            "Previous partial answer:\n"
            f"{text[-1200:]}"
        )
        cont_data = _post_with_optional_fallback(continuation_prompt)
        cont_text, finish_reason = _extract_text_and_finish_reason(cont_data)
        if cont_text:
            text = f"{text}\n{cont_text}".strip()
        continuation_attempts += 1

    return text or "Gemini returned an empty response."


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_public_system_list() -> list[dict[str, Any]]:
    """Fetch the public HeatPumpMonitor system metadata list."""
    response = requests.get(SYSTEM_LIST_PUBLIC_URL, timeout=40)
    response.raise_for_status()
    payload = response.json()

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        rows = payload.get("systems")
        if isinstance(rows, list):
            return rows
    raise ValueError("Unexpected metadata payload format from HeatPumpMonitor API.")


def find_system_metadata(rows: list[dict[str, Any]], system_id: int) -> dict[str, Any] | None:
    """Return one system metadata record by integer id."""
    for row in rows:
        try:
            if int(row.get("id")) == int(system_id):
                return row
        except Exception:
            continue
    return None


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sanitize_default_systems(raw_systems: Any) -> list[dict[str, float]]:
    systems: list[dict[str, float]] = []
    seen: set[int] = set()
    fallback_capacity = float(os.getenv("DEFAULT_CAPACITY_KW", "6"))

    if not isinstance(raw_systems, list):
        return systems

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

        try:
            capacity_kw = float(item.get("capacity_kw"))
        except Exception:
            capacity_kw = fallback_capacity

        systems.append({"series_id": sid, "capacity_kw": capacity_kw})

    return systems


def _format_capacity_kw(capacity_kw: float) -> str:
    if float(capacity_kw).is_integer():
        return str(int(capacity_kw))
    return str(capacity_kw)


def _fetch_defaults_source_mtime() -> float:
    source_path = SRC_DIR / "01_fetch_data.py"
    if not source_path.exists():
        return -1.0
    return source_path.stat().st_mtime


@st.cache_data(show_spinner=False)
def load_fetch_stage_default_systems(source_mtime: float) -> list[dict[str, float]]:
    _ = source_mtime
    try:
        fetch_module = load_module(SRC_DIR / "01_fetch_data.py", "fetch_defaults_streamlit")
        return _sanitize_default_systems(getattr(fetch_module, "DEFAULT_SYSTEMS", []))
    except Exception:
        return []


def get_default_system_context() -> dict[str, Any]:
    defaults = load_fetch_stage_default_systems(_fetch_defaults_source_mtime())

    system_ids_csv = ",".join(str(item["series_id"]) for item in defaults)
    systems_config_csv = ",".join(
        f"{item['series_id']}:{_format_capacity_kw(item['capacity_kw'])}"
        for item in defaults
    )

    train_subset = defaults[:2]
    train_system_ids_csv = ",".join(str(item["series_id"]) for item in train_subset)

    signature = json.dumps(defaults, sort_keys=True)
    default_system_id = int(defaults[0]["series_id"]) if defaults else 1

    return {
        "defaults": defaults,
        "signature": signature,
        "system_ids_csv": system_ids_csv,
        "train_system_ids_csv": train_system_ids_csv,
        "systems_config_csv": systems_config_csv,
        "default_system_id": default_system_id,
    }


def run_python_script(
    script_path: Path,
    timeout_seconds: int,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    start = datetime.now()
    try:
        cmd_env = os.environ.copy()
        if extra_env:
            cmd_env.update(extra_env)

        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
            env=cmd_env,
        )
        ok = proc.returncode == 0
        return {
            "ok": ok,
            "return_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "started": start.isoformat(),
            "finished": datetime.now().isoformat(),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "return_code": -1,
            "stdout": (exc.stdout or ""),
            "stderr": f"Timed out after {timeout_seconds}s\n{exc.stderr or ''}",
            "started": start.isoformat(),
            "finished": datetime.now().isoformat(),
        }


def run_monitoring_stage() -> dict[str, Any]:
    sys.path.insert(0, str(SRC_DIR))
    monitor_module = load_module(SRC_DIR / "monitoring_model_06.py", "monitoring_model_06")

    processed_dir = DATA_DIR / "processed"
    pred_candidates = sorted(processed_dir.glob("*pred*.csv")) if processed_dir.exists() else []
    if not pred_candidates:
        return {
            "ok": False,
            "return_code": 1,
            "stdout": "",
            "stderr": "No prediction CSV found in data/processed. Run stage 05 first.",
            "started": datetime.now().isoformat(),
            "finished": datetime.now().isoformat(),
        }

    pred_path = pred_candidates[-1]
    pred_df = pd.read_csv(pred_path)
    reference_path = DATA_DIR / "processed" / "daikin_clean.csv"
    reference_df = pd.read_csv(reference_path) if reference_path.exists() else pred_df.copy()

    summary = monitor_module.summarize_batch_monitoring(
        input_df=pred_df,
        prediction_df=pred_df,
        reference_df=reference_df,
    )
    output_path = MODELS_DIR / "latest_monitoring_summary.json"
    monitor_module.save_monitoring_summary(summary, str(output_path))

    return {
        "ok": True,
        "return_code": 0,
        "stdout": f"Monitoring summary saved to {output_path}",
        "stderr": "",
        "started": datetime.now().isoformat(),
        "finished": datetime.now().isoformat(),
    }


def execute_stage(
    stage: dict[str, str],
    timeout_seconds: int,
    train_policy_mode: str = "enhanced_onestep",
    custom_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stage_name_for_label = Path(stage.get("path", "")).name.lower()

    if stage["kind"] == "script":
        extra_env: dict[str, str] = {}
        stage_name = stage_name_for_label
        if stage_name == "04_train_model.py":
            extra_env["FEATURE_POLICY_MODE"] = train_policy_mode

        custom_options = custom_options or {}
        system_ids_csv = str(custom_options.get("system_ids", "")).strip()
        train_system_ids_csv = str(custom_options.get("train_system_ids", "")).strip()

        if system_ids_csv:
            extra_env["SYSTEM_IDS"] = system_ids_csv
        if train_system_ids_csv:
            extra_env["TRAIN_SYSTEM_IDS"] = train_system_ids_csv

        if stage_name == "01_fetch_data.py":
            extra_env.update(
                {
                    "FETCH_MODE": str(custom_options.get("fetch_mode", "fixed_window")),
                    "START_DATE": str(custom_options.get("fetch_start", "01-01-2023")),
                    "END_DATE": str(custom_options.get("fetch_end", "01-02-2026")),
                    "API_MAX_DATAPOINTS": str(custom_options.get("fetch_api_max_datapoints", 70000)),
                    "SYSTEMS_CONFIG": str(custom_options.get("systems_config", "")),
                    "DEFAULT_CAPACITY_KW": str(custom_options.get("default_capacity_kw", 6.0)),
                }
            )
        elif stage_name == "02_feature_engineering.py":
            extra_env.update(
                {
                    "FE_STANDBY_THRESHOLD_W": str(custom_options.get("fe_standby_threshold_w", 50.0)),
                    "FE_BASE_TEMP": str(custom_options.get("fe_base_temp", 15.5)),
                }
            )
        elif stage_name == "03_clean_data.py":
            extra_env.update(
                {
                    "CLEAN_COP_MIN": str(custom_options.get("clean_cop_min", 0.0)),
                    "CLEAN_COP_MAX": str(custom_options.get("clean_cop_max", 8.0)),
                    "CLEAN_OUTSIDE_T_MIN": str(custom_options.get("clean_outside_t_min", -20.0)),
                    "CLEAN_OUTSIDE_T_MAX": str(custom_options.get("clean_outside_t_max", 40.0)),
                }
            )
        elif stage_name == "backtest_model_07.py":
            extra_env["FEATURE_POLICY_MODE"] = str(custom_options.get("backtest_policy_mode", train_policy_mode))

        result = run_python_script(ROOT / stage["path"], timeout_seconds, extra_env=extra_env)
    else:
        result = run_monitoring_stage()
    result["stage_id"] = stage["id"]
    result["stage_label"] = (
        f"{stage['label']} [{train_policy_mode}]"
        if stage_name_for_label == "04_train_model.py"
        else stage["label"]
    )
    return result


def run_duration_seconds(run: dict[str, Any]) -> float | None:
    try:
        start = datetime.fromisoformat(str(run.get("started", "")))
        end = datetime.fromisoformat(str(run.get("finished", "")))
        return max(0.0, (end - start).total_seconds())
    except Exception:
        return None


def infer_single_input_default(col: str, dtype_label: str) -> Any:
    name = col.lower()

    if dtype_label == "datetime":
        return datetime.now().isoformat()
    if dtype_label in {"string"}:
        return ""
    if dtype_label in {"bool", "int"}:
        if "heating_on" in name:
            return 1
        if "system_" in name:
            return 0
        if "day_of_week" in name:
            return 1
        if "month" in name:
            return 1
        if "hour" in name:
            return 12
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
    if "month" in name:
        return 1.0
    if "hour" in name:
        return 12.0
    return 0.0


def render_hero() -> None:
    st.markdown(
        """
        <section class="hero">
            <h1>Heat Pump AI Ops Console</h1>
            <p>Production-style command center for data ingestion, feature engineering, training, scoring, monitoring, and Gemini-assisted analysis.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_top_stats() -> None:
    manifest_path = latest_file(MODELS_DIR, "run_manifest_*.json")
    monitor_path = MODELS_DIR / "latest_monitoring_summary.json"
    backtest_path = MODELS_DIR / "walk_forward_summary.json"

    metric_rows = {
        "Latest Manifest": manifest_path.name if manifest_path else "Not available",
        "Monitoring": "Ready" if monitor_path.exists() else "Not available",
        "Backtest": "Ready" if backtest_path.exists() else "Not available",
        "Python": Path(sys.executable).name,
    }

    cards_html = "".join(
        (
            "<div class='stat-card'>"
            f"<div class='stat-label'>{label}</div>"
            f"<div class='stat-value'>{value}</div>"
            "</div>"
        )
        for label, value in metric_rows.items()
    )
    st.markdown(f"<div class='stat-grid'>{cards_html}</div>", unsafe_allow_html=True)


def render_pipeline_tab() -> None:
    st.subheader("Pipeline Automation")
    st.caption("Run every src stage directly from this UI. No manual terminal workflow needed.")

    pipeline_stages = discover_pipeline_stages()
    if not pipeline_stages:
        st.error("No pipeline stages discovered in src/. Add numbered scripts like 01_*.py.")
        return

    defaults_ctx = get_default_system_context()
    if st.session_state.get("cfg_default_systems_signature") != defaults_ctx["signature"]:
        st.session_state["cfg_systems_config"] = defaults_ctx["systems_config_csv"]
        st.session_state["cfg_default_systems_signature"] = defaults_ctx["signature"]

    timeout_seconds = st.slider("Stage timeout (seconds)", min_value=120, max_value=7200, value=1800, step=60)
    stop_on_error = st.checkbox("Stop full pipeline on first error", value=True)
    train_policy_mode = st.radio(
        "Train model feature policy (for stage 04)",
        ["enhanced_onestep", "strict_production", "both"],
        horizontal=True,
        key="pipeline_train_policy_mode",
    )

    with st.expander("Custom Pipeline Options", expanded=False):
        st.caption("Set optional stage-level settings before running full pipeline.")

        st.markdown("#### System Selection")
        ss1, ss2 = st.columns(2)
        system_ids = ss1.text_input(
            "SYSTEM_IDS (optional)",
            value="",
            help=(
                "Comma-separated IDs used by fetch + feature engineering + training. "
                f"Current defaults: {defaults_ctx['system_ids_csv'] or 'none found'}"
            ),
            key="cfg_system_ids",
        )
        train_system_ids = ss2.text_input(
            "TRAIN_SYSTEM_IDS (optional)",
            value="",
            help=(
                "Optional training-only subset. "
                f"Suggested from defaults: {defaults_ctx['train_system_ids_csv'] or 'set manually'}"
            ),
            key="cfg_train_system_ids",
        )

        st.markdown("#### 01 Fetch Data")
        fo1, fo2, fo3, fo4 = st.columns(4)
        fetch_mode = fo1.selectbox("Fetch mode", ["fixed_window", "max_available"], key="cfg_fetch_mode")
        fetch_start = fo2.text_input("Start (DD-MM-YYYY)", value="01-01-2023", key="cfg_fetch_start")
        fetch_end = fo3.text_input("End (DD-MM-YYYY)", value="01-02-2026", key="cfg_fetch_end")
        fetch_api_max_datapoints = fo4.number_input(
            "API max datapoints",
            min_value=1000,
            max_value=500000,
            value=70000,
            step=1000,
            key="cfg_fetch_api_max_datapoints",
        )
        systems_config = st.text_area(
            "Systems (system_id:capacity_kw, comma-separated)",
            value=defaults_ctx["systems_config_csv"],
            help=(
                "Auto-loaded from src/01_fetch_data.py DEFAULT_SYSTEMS. "
                "You can also pass a JSON list via env if needed."
            ),
            key="cfg_systems_config",
        )
        default_capacity_kw = st.number_input(
            "Default capacity for unknown SYSTEM_IDS (kW)",
            min_value=1.0,
            max_value=30.0,
            value=6.0,
            step=0.5,
            key="cfg_default_capacity_kw",
        )

        st.markdown("#### 02 Feature Engineering")
        fe1, fe2 = st.columns(2)
        fe_standby_threshold_w = fe1.number_input(
            "Standby threshold (W)",
            min_value=0.0,
            max_value=500.0,
            value=50.0,
            step=5.0,
            key="cfg_fe_standby_threshold_w",
        )
        fe_base_temp = fe2.number_input(
            "Base temperature",
            min_value=-10.0,
            max_value=30.0,
            value=15.5,
            step=0.5,
            key="cfg_fe_base_temp",
        )

        st.markdown("#### 03 Clean Data")
        cl1, cl2, cl3, cl4 = st.columns(4)
        clean_cop_min = cl1.number_input("COP min", min_value=0.0, max_value=8.0, value=0.0, step=0.1, key="cfg_clean_cop_min")
        clean_cop_max = cl2.number_input("COP max", min_value=0.1, max_value=12.0, value=8.0, step=0.1, key="cfg_clean_cop_max")
        clean_outside_t_min = cl3.number_input("OutsideT min", min_value=-60.0, max_value=20.0, value=-20.0, step=1.0, key="cfg_clean_outside_t_min")
        clean_outside_t_max = cl4.number_input("OutsideT max", min_value=20.0, max_value=80.0, value=40.0, step=1.0, key="cfg_clean_outside_t_max")

        st.markdown("#### 07 Backtest")
        backtest_policy_mode = st.radio(
            "Backtest feature policy",
            ["enhanced_onestep", "strict_production"],
            horizontal=True,
            key="cfg_backtest_policy_mode",
        )

    custom_options = {
        "system_ids": system_ids,
        "train_system_ids": train_system_ids,
        "fetch_mode": fetch_mode,
        "fetch_start": fetch_start,
        "fetch_end": fetch_end,
        "fetch_api_max_datapoints": fetch_api_max_datapoints,
        "systems_config": systems_config,
        "default_capacity_kw": default_capacity_kw,
        "fe_standby_threshold_w": fe_standby_threshold_w,
        "fe_base_temp": fe_base_temp,
        "clean_cop_min": clean_cop_min,
        "clean_cop_max": clean_cop_max,
        "clean_outside_t_min": clean_outside_t_min,
        "clean_outside_t_max": clean_outside_t_max,
        "backtest_policy_mode": backtest_policy_mode,
    }

    c1, c2 = st.columns([1, 1])
    if c1.button("Run Full Pipeline", type="primary", width="stretch"):
        for stage in pipeline_stages:
            stage_name = Path(stage.get("path", "")).name.lower()
            train_modes = [train_policy_mode]
            if stage_name == "04_train_model.py" and train_policy_mode == "both":
                train_modes = ["enhanced_onestep", "strict_production"]

            for mode in train_modes:
                with st.spinner(f"Running {stage['label']} ({mode})..."):
                    run = execute_stage(
                        stage,
                        timeout_seconds,
                        train_policy_mode=mode,
                        custom_options=custom_options,
                    )
                    st.session_state["stage_runs"].append(run)
                if not run["ok"] and stop_on_error:
                    st.error(f"{run['stage_label']} failed. Full pipeline stopped.")
                    return

    if c2.button("Clear Run History", width="stretch"):
        st.session_state["stage_runs"] = []
        st.rerun()

    st.markdown("### Run Individual Stage")
    col_count = min(4, max(1, len(pipeline_stages)))
    stage_cols = st.columns(col_count)
    for idx, stage in enumerate(pipeline_stages):
        col = stage_cols[idx % col_count]
        if col.button(stage["label"], key=f"run_{stage['id']}", width="stretch"):
            stage_name = Path(stage.get("path", "")).name.lower()
            train_modes = [train_policy_mode]
            if stage_name == "04_train_model.py" and train_policy_mode == "both":
                train_modes = ["enhanced_onestep", "strict_production"]

            for mode in train_modes:
                with st.spinner(f"Running {stage['label']} ({mode})..."):
                    run = execute_stage(
                        stage,
                        timeout_seconds,
                        train_policy_mode=mode,
                        custom_options=custom_options,
                    )
                    st.session_state["stage_runs"].append(run)
            st.rerun()

    st.markdown("### Execution Log")
    runs = list(reversed(st.session_state["stage_runs"]))
    if not runs:
        st.info("No runs yet.")
        return

    success_count = sum(1 for r in runs if r.get("ok"))
    fail_count = len(runs) - success_count
    most_recent_duration = run_duration_seconds(runs[0])

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Runs", len(runs))
    s2.metric("Succeeded", success_count)
    s3.metric("Failed", fail_count)
    s4.metric("Latest Duration", f"{most_recent_duration:.1f}s" if most_recent_duration is not None else "-")

    filter_col, download_col = st.columns([1, 1])
    status_filter = filter_col.selectbox("Filter", ["All", "Success", "Failed"], key="log_filter")

    combined_log = "\n\n".join(
        [
            (
                f"Stage: {r.get('stage_label')}\n"
                f"Status: {'SUCCESS' if r.get('ok') else 'FAILED'}\n"
                f"Return code: {r.get('return_code')}\n"
                f"Started: {r.get('started')}\n"
                f"Finished: {r.get('finished')}\n\n"
                f"STDOUT:\n{r.get('stdout', '')}\n\nSTDERR:\n{r.get('stderr', '')}"
            )
            for r in runs
        ]
    )
    download_col.download_button(
        "Download Logs",
        data=combined_log.encode("utf-8"),
        file_name=f"pipeline_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
    )

    filtered_runs = runs
    if status_filter == "Success":
        filtered_runs = [r for r in runs if r.get("ok")]
    elif status_filter == "Failed":
        filtered_runs = [r for r in runs if not r.get("ok")]

    if not filtered_runs:
        st.info("No runs for selected filter.")
        return

    for idx, run in enumerate(filtered_runs, start=1):
        status = "SUCCESS" if run.get("ok") else "FAILED"
        badge_class = "success" if run.get("ok") else "failed"
        duration_seconds = run_duration_seconds(run)
        duration_text = f"{duration_seconds:.1f}s" if duration_seconds is not None else "-"

        st.markdown(
            (
                "<div class='log-card'>"
                "<div>"
                f"<div class='log-title'>{idx}. {run.get('stage_label', 'Unknown Stage')}</div>"
                f"<div class='log-sub'>Return code: {run.get('return_code')} | Duration: {duration_text}</div>"
                "</div>"
                f"<span class='log-badge {badge_class}'>{status}</span>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        with st.expander(f"Details: {run.get('stage_label', 'Stage')}", expanded=False):
            meta1, meta2, meta3 = st.columns(3)
            meta1.caption(f"Started: {run.get('started', '-')}")
            meta2.caption(f"Finished: {run.get('finished', '-')}")
            meta3.caption(f"Duration: {duration_text}")

            out_tab, err_tab = st.tabs(["stdout", "stderr"])
            with out_tab:
                st.code(run.get("stdout", "") or "<empty>", language="bash")
            with err_tab:
                st.code(run.get("stderr", "") or "<empty>", language="bash")


def render_artifacts_tab() -> None:
    st.subheader("Data and Artifact Explorer")

    patterns = [
        DATA_DIR / "raw",
        DATA_DIR / "processed",
        MODELS_DIR,
    ]

    allowed_suffixes = {".csv", ".json", ".txt", ".parquet", ".md"}

    files: list[Path] = []
    for folder in patterns:
        if folder.exists():
            files.extend([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in allowed_suffixes])

    files = sorted(files)
    if not files:
        st.info("No artifacts found yet. Run pipeline stages first.")
        return

    selected = st.selectbox("Select artifact", files, format_func=lambda p: str(p.relative_to(ROOT)))
    suffix = selected.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(selected)
        st.write(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
        st.dataframe(df.head(200), width="stretch")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            default_cols = [c for c in ["pred_heatpump_elec", "pred_heatpump_heat", "heatpump_elec", "heatpump_heat"] if c in numeric_cols]
            chart_cols = st.multiselect(
                "Numeric columns to visualize",
                options=numeric_cols,
                default=default_cols[:2] if default_cols else numeric_cols[:2],
            )
            if chart_cols:
                if "timestamp" in df.columns:
                    chart_df = df[["timestamp"] + chart_cols].copy()
                    chart_df = chart_df.set_index("timestamp")
                    st.line_chart(chart_df)
                else:
                    st.line_chart(df[chart_cols])

    elif suffix == ".json":
        st.json(read_json(selected))
    else:
        st.code(selected.read_text(encoding="utf-8"), language="text")


def render_model_tab() -> None:
    st.subheader("Model Health")
    st.caption("Choose which model bundle to inspect (enhanced or strict production).")

    selected_run, manifest = select_manifest_run(
        key_prefix="model_health",
        run_label="Model run",
    )
    if not selected_run or not manifest:
        return

    st.caption(
        f"Loaded run: {selected_run['run_tag']} | {selected_run['policy']} | {selected_run['strategy']}"
    )
    metrics = manifest.get("metrics", {})

    row1 = st.columns(4)
    row1[0].metric("R2 Electricity", f"{metrics.get('r2_elec', 0.0):.3f}")
    row1[1].metric("R2 Heat", f"{metrics.get('r2_heat', 0.0):.3f}")
    row1[2].metric("MAE Electricity (W)", f"{metrics.get('mae_elec_w', 0.0):.1f}")
    row1[3].metric("MAE Heat (W)", f"{metrics.get('mae_heat_w', 0.0):.1f}")

    row2 = st.columns(3)
    row2[0].metric("Energy Error %", f"{metrics.get('energy_err_pct', 0.0):.2f}%")
    row2[1].metric("Heat Error %", f"{metrics.get('heat_err_pct', 0.0):.2f}%")
    row2[2].metric("COP Error %", f"{metrics.get('cop_err_pct', 0.0):.2f}%")

    if "strategy_scores" in manifest:
        st.markdown("### Strategy Leaderboard")
        st.dataframe(pd.DataFrame(manifest["strategy_scores"]), width="stretch")

    gates = manifest.get("gates")
    if gates:
        st.markdown("### Gate Status")
        st.json(gates)

    backtest_path = MODELS_DIR / "walk_forward_folds.csv"
    if backtest_path.exists():
        st.markdown("### Walk-Forward Folds")
        backtest_df = pd.read_csv(backtest_path)
        st.dataframe(backtest_df, width="stretch")


def render_predict_tab() -> None:
    st.subheader("Live Batch Scoring")
    st.caption("Upload a feature CSV and run the production prediction bundle from src/05_predict_model.py")

    uploaded = st.file_uploader("Upload inference CSV", type=["csv"], key="predict_upload")

    sys.path.insert(0, str(SRC_DIR))
    predict_module = load_module(SRC_DIR / "05_predict_model.py", "predict_model_05")

    manifests = list_manifest_runs()
    if not manifests:
        st.warning("No run manifests available yet. Run training stage first.")
        return

    policy_mode = st.radio(
        "Feature policy",
        ["enhanced_onestep", "strict_production", "all"],
        horizontal=True,
        key="predict_policy_mode",
    )

    filtered_manifests = manifests
    if policy_mode != "all":
        filtered_manifests = [m for m in manifests if m.get("policy") == policy_mode]

    if not filtered_manifests:
        st.warning("No run tags found for this feature policy. Train a run for the selected mode.")
        return

    run_tag_options = [m["run_tag"] for m in filtered_manifests]
    run_tag = st.selectbox(
        "Run tag",
        options=run_tag_options,
        format_func=lambda t: next(
            (
                f"{m['run_tag']} | {m['policy']} | {m['strategy']}"
                for m in filtered_manifests if m["run_tag"] == t
            ),
            t,
        ),
        key="predict_run_tag",
    )

    if uploaded is None:
        st.info("Upload a CSV to run scoring.")
        return

    input_df = pd.read_csv(uploaded)
    st.write("Input preview")
    st.dataframe(input_df.head(30), width="stretch")

    if st.button("Score Batch", type="primary"):
        with st.spinner("Scoring with bundled artifacts..."):
            scored = predict_module.predict_bundle(input_df, run_tag=run_tag)
            st.session_state["latest_scored_df"] = scored
            st.session_state["latest_prediction_summary"] = summarize_prediction_df(scored)

            output_dir = DATA_DIR / "processed"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_name = f"ui_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            output_path = output_dir / output_name
            scored.to_csv(output_path, index=False)

        st.success(f"Scoring complete. Saved {output_path.relative_to(ROOT)}")
        st.dataframe(scored.head(60), width="stretch")

        m1, m2, m3 = st.columns(3)
        m1.metric("Rows", f"{len(scored):,}")
        m2.metric("Predicted ON Rate", f"{scored['runtime_on_pred'].mean() * 100:.2f}%")
        m3.metric("COP Guardrail Adjustments", int(scored.get("cop_guardrail_adjusted", pd.Series(dtype=int)).sum()))

        plot_cols = [c for c in ["pred_heatpump_elec", "pred_heatpump_heat"] if c in scored.columns]
        if plot_cols:
            if "timestamp" in scored.columns:
                trend = scored[["timestamp"] + plot_cols].set_index("timestamp")
                st.line_chart(trend)
            else:
                st.line_chart(scored[plot_cols])

        st.download_button(
            "Download Scored CSV",
            data=scored.to_csv(index=False).encode("utf-8"),
            file_name="scored_predictions.csv",
            mime="text/csv",
        )


def render_single_input_tab() -> None:
    st.subheader("Single Input Prediction")
    st.caption("Enter one row of feature values and predict instantly.")

    sys.path.insert(0, str(SRC_DIR))
    predict_module = load_module(SRC_DIR / "05_predict_model.py", "predict_model_05_single")

    manifests = list_manifest_runs()
    if not manifests:
        st.warning("No run manifests available yet. Run training stage first.")
        return

    policy_mode = st.radio(
        "Feature policy (single input)",
        ["enhanced_onestep", "strict_production", "all"],
        horizontal=True,
        key="single_policy_mode",
    )

    filtered_manifests = manifests
    if policy_mode != "all":
        filtered_manifests = [m for m in manifests if m.get("policy") == policy_mode]

    if not filtered_manifests:
        st.warning("No run tags found for this feature policy.")
        return

    run_tag_options = [m["run_tag"] for m in filtered_manifests]
    run_tag = st.selectbox(
        "Run tag (single input)",
        options=run_tag_options,
        format_func=lambda t: next(
            (
                f"{m['run_tag']} | {m['policy']} | {m['strategy']}"
                for m in filtered_manifests if m["run_tag"] == t
            ),
            t,
        ),
        key="single_run_tag",
    )

    try:
        bundle = predict_module.load_bundle(run_tag)
    except Exception as exc:
        st.error(f"Could not load model bundle for run tag '{run_tag}': {exc}")
        return

    feature_schema = bundle.get("feature_schema", {})
    required_cols = feature_schema.get("required_serving_columns", [])
    dtype_map = feature_schema.get("feature_dtypes", {})

    if not required_cols:
        st.error("Feature schema is missing required serving columns.")
        return

    signature = f"{run_tag}:{len(required_cols)}"
    if st.session_state["single_input_signature"] != signature:
        defaults = {
            col: infer_single_input_default(col, dtype_map.get(col, "float"))
            for col in required_cols
        }
        st.session_state["single_input_signature"] = signature
        st.session_state["single_input_row"] = defaults

    current_row = st.session_state["single_input_row"]
    editor_df = pd.DataFrame([{col: current_row.get(col) for col in required_cols}])

    st.write(f"Required features: {len(required_cols)}")
    edited_df = st.data_editor(
        editor_df,
        num_rows="fixed",
        width="stretch",
        hide_index=True,
        key=f"single_editor_{signature}",
    )

    if st.button("Predict Single Row", type="primary"):
        try:
            scored = predict_module.predict_bundle(edited_df.copy(), run_tag=run_tag)
            st.session_state["single_input_row"] = edited_df.iloc[0].to_dict()
            st.session_state["latest_scored_df"] = scored
            st.session_state["latest_prediction_summary"] = summarize_prediction_df(scored)

            pred_row = scored.iloc[0]
            p1, p2, p3 = st.columns(3)
            p1.metric("Pred Electricity (W)", f"{float(pred_row.get('pred_heatpump_elec', 0.0)):.2f}")
            p2.metric("Pred Heat (W)", f"{float(pred_row.get('pred_heatpump_heat', 0.0)):.2f}")
            p3.metric("Pred COP", f"{float(pred_row.get('pred_cop', 0.0)):.3f}")

            st.markdown("### Predicted Row")
            st.dataframe(scored, width="stretch")
        except Exception as exc:
            st.error(f"Single row prediction failed: {exc}")


def render_gemini_tab() -> None:
    st.subheader("Gemini Analyst")
    st.caption("The app auto-loads GEMINI_API_KEY from your .env file if present.")

    selected_run, selected_manifest = select_manifest_run(
        key_prefix="gemini_context",
        run_label="Model context run",
    )
    if selected_run:
        st.caption(
            f"Gemini context run: {selected_run['run_tag']} | {selected_run['policy']} | {selected_run['strategy']}"
        )

    default_key = os.getenv("GEMINI_API_KEY", "")
    api_key = st.text_input("Gemini API Key", type="password", value=default_key)
    model_name = st.selectbox(
        "Gemini Model",
        ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
    )

    monitoring = MODELS_DIR / "latest_monitoring_summary.json"
    context = {
        "selected_run": selected_run or {},
        "manifest": selected_manifest or {},
        "monitoring": read_json(monitoring) if monitoring.exists() else {},
    }

    prompt = st.text_area(
        "Prompt",
        value=(
            "Review this heat pump model run and provide:\n"
            "1) top 3 reliability risks,\n"
            "2) top 3 leakage-safe improvements,\n"
            "3) a concrete next experiment plan."
        ),
        height=140,
    )

    st.markdown("### Prediction Output Expert Analysis")
    st.caption("Analyze the latest prediction output with expert-level interpretation.")

    latest_summary = st.session_state.get("latest_prediction_summary")
    expert_request = st.text_area(
        "Prediction analysis request",
        value=(
            "Interpret this latest prediction output like a senior data expert. "
            "Highlight quality, anomalies, and actionable next steps."
        ),
        height=90,
        key="gemini_prediction_analysis_request",
    )

    if latest_summary:
        st.json(latest_summary)
    else:
        st.info("No latest prediction output found yet. Run Live Scoring or Single Input prediction first.")

    if st.button("Analyze Latest Prediction", type="primary"):
        if not api_key:
            st.error("No API key found. Save GEMINI_API_KEY in .env or enter it above.")
            return
        if not latest_summary:
            st.error("No prediction output available. Run a prediction first.")
            return

        full_prompt = build_prediction_expert_prompt(latest_summary, expert_request)
        with st.spinner("Gemini is analyzing latest prediction output..."):
            try:
                expert_answer = call_gemini(api_key=api_key, model_name=model_name, prompt=full_prompt)
            except Exception as exc:
                st.error(
                    "Gemini request failed. Try model 'gemini-2.0-flash' or 'gemini-1.5-flash'."
                )
                st.caption(str(exc))
                return
        st.markdown("#### Expert Analysis")
        st.write(expert_answer)

    if st.button("Ask Gemini", type="primary"):
        if not api_key:
            st.error("No API key found. Save GEMINI_API_KEY in .env or enter it above.")
            return

        full_prompt = (
            "You are a senior ML reliability engineer for heat pump forecasting. "
            "Be concrete, leakage-safe, and physically grounded.\n"
            "Answer in 150 words.\n\n"
            f"Context:\n{json.dumps(context, indent=2)}\n\n"
            f"User request:\n{prompt}"
        )
        with st.spinner("Querying Gemini..."):
            try:
                answer = call_gemini(api_key=api_key, model_name=model_name, prompt=full_prompt)
            except Exception as exc:
                st.error(
                    "Gemini request failed. Try model 'gemini-2.0-flash' or 'gemini-1.5-flash'."
                )
                st.caption(str(exc))
                return
        st.markdown("### Response")
        st.write(answer)


def render_system_metadata_tab() -> None:
    st.subheader("System Metadata Lookup")
    st.caption("Enter a system ID to fetch public metadata from HeatPumpMonitor.")

    default_system_id = get_default_system_context()["default_system_id"]
    c1, c2, c3 = st.columns([2, 1, 1])
    system_id = c1.number_input("System ID", min_value=1, value=default_system_id, step=1)
    lookup_clicked = c2.button("Lookup", type="primary", width="stretch")
    refresh_clicked = c3.button("Refresh API Cache", width="stretch")

    if refresh_clicked:
        fetch_public_system_list.clear()
        st.success("Metadata cache cleared. Next lookup will fetch fresh data.")

    if not lookup_clicked:
        st.info("Enter an ID and click Lookup.")
        return

    with st.spinner("Fetching metadata..."):
        try:
            systems = fetch_public_system_list()
        except Exception as exc:
            st.error(f"Could not fetch metadata from API: {exc}")
            return

    record = find_system_metadata(systems, int(system_id))
    if record is None:
        st.warning(f"System ID {int(system_id)} was not found in the public metadata list.")
        st.caption(f"Total systems fetched: {len(systems)}")
        return

    last_updated_unix = record.get("last_updated")
    last_updated_text = "Unknown"
    try:
        if last_updated_unix is not None:
            last_updated_text = datetime.fromtimestamp(int(last_updated_unix), UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        last_updated_text = str(last_updated_unix)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("System ID", int(record.get("id", system_id)))
    m2.metric("Manufacturer", str(record.get("hp_manufacturer") or "Unknown"))
    m3.metric("Model", str(record.get("hp_model") or "Unknown"))
    m4.metric("Output (kW)", str(record.get("hp_output") or "Unknown"))

    def render_kv_section(title: str, data: dict[str, Any]) -> None:
        st.markdown(f"### {title}")
        # Streamlit Arrow serialization can fail on mixed object types; render values as text.
        rows = [{"Field": str(k), "Value": "" if data[k] is None else str(data[k])} for k in data]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    s1, s2 = st.columns(2)
    with s1:
        render_kv_section(
            "Quick Summary",
            {
                "last_updated": last_updated_text,
                "location": record.get("location"),
                "installer_name": record.get("installer_name"),
                "installer_url": record.get("installer_url"),
                "published": record.get("published"),
            },
        )
    with s2:
        render_kv_section(
            "Heat Pump",
            {
                "hp_manufacturer": record.get("hp_manufacturer"),
                "hp_model": record.get("hp_model"),
                "hp_type": record.get("hp_type"),
                "hp_output": record.get("hp_output"),
                "hp_max_output": record.get("hp_max_output"),
                "refrigerant": record.get("refrigerant"),
            },
        )

    s3, s4 = st.columns(2)
    with s3:
        render_kv_section(
            "Location and Property",
            {
                "property": record.get("property"),
                "age": record.get("age"),
                "floor_area": record.get("floor_area"),
                "heat_loss": record.get("heat_loss"),
                "design_temp": record.get("design_temp"),
                "latitude": record.get("latitude"),
                "longitude": record.get("longitude"),
            },
        )
    with s4:
        render_kv_section(
            "Hydraulics and Emitters",
            {
                "hydraulic_separation": record.get("hydraulic_separation"),
                "flow_temp": record.get("flow_temp"),
                "flow_temp_typical": record.get("flow_temp_typical"),
                "UFH": record.get("UFH"),
                "new_radiators": record.get("new_radiators"),
                "old_radiators": record.get("old_radiators"),
                "fan_coil_radiators": record.get("fan_coil_radiators"),
            },
        )

    s5, s6 = st.columns(2)
    with s5:
        render_kv_section(
            "DHW and Controls",
            {
                "dhw_method": record.get("dhw_method"),
                "dhw_control_type": record.get("dhw_control_type"),
                "dhw_target_temperature": record.get("dhw_target_temperature"),
                "cylinder_volume": record.get("cylinder_volume"),
                "space_heat_control_type": record.get("space_heat_control_type"),
                "zone_number": record.get("zone_number"),
            },
        )
    with s6:
        render_kv_section(
            "Tariff and Solar",
            {
                "electricity_tariff": record.get("electricity_tariff"),
                "electricity_tariff_type": record.get("electricity_tariff_type"),
                "electricity_tariff_unit_rate_all": record.get("electricity_tariff_unit_rate_all"),
                "solar_pv_generation": record.get("solar_pv_generation"),
                "solar_pv_self_consumption": record.get("solar_pv_self_consumption"),
                "battery_storage_capacity": record.get("battery_storage_capacity"),
            },
        )

    s7, s8 = st.columns(2)
    with s7:
        render_kv_section(
            "Metering Boundary",
            {
                "electric_meter": record.get("electric_meter"),
                "heat_meter": record.get("heat_meter"),
                "metering_inc_controls": record.get("metering_inc_controls"),
                "metering_inc_central_heating_pumps": record.get("metering_inc_central_heating_pumps"),
                "metering_inc_secondary_heating_pumps": record.get("metering_inc_secondary_heating_pumps"),
                "boundary_code": record.get("boundary_code"),
            },
        )
    with s8:
        render_kv_section(
            "Data Health",
            {
                "data_flag": record.get("data_flag"),
                "data_flag_note": record.get("data_flag_note"),
                "heatpump_elec_ago": record.get("heatpump_elec_ago"),
                "heatpump_heat_ago": record.get("heatpump_heat_ago"),
                "heatpump_max_age": record.get("heatpump_max_age"),
            },
        )

    boundary = record.get("boundary_metering")
    if isinstance(boundary, dict) and boundary:
        st.markdown("### Boundary Metering Detail")
        st.json(boundary)

    st.markdown("### Raw Metadata JSON")
    st.json(record)


def main() -> None:
    load_env_file(ROOT / ".env")
    ensure_state()

    st.set_page_config(page_title="Heat Pump AI Ops Console", layout="wide")

    with st.sidebar:
        st.markdown("### Appearance")
        appearance_mode = st.radio("Theme", ["Auto", "Light", "Dark"], horizontal=True)
        st.caption("Auto follows your OS/browser preference.")

    inject_ui(appearance_mode=appearance_mode)

    render_hero()
    render_top_stats()

    tab_pipeline, tab_artifacts, tab_model, tab_predict, tab_single, tab_metadata, tab_gemini = st.tabs(
        ["Pipeline", "Artifacts", "Model Health", "Live Scoring", "Single Input", "System Metadata", "Gemini"]
    )

    with tab_pipeline:
        render_pipeline_tab()
    with tab_artifacts:
        render_artifacts_tab()
    with tab_model:
        render_model_tab()
    with tab_predict:
        render_predict_tab()
    with tab_single:
        render_single_input_tab()
    with tab_metadata:
        render_system_metadata_tab()
    with tab_gemini:
        render_gemini_tab()


if __name__ == "__main__":
    main()
