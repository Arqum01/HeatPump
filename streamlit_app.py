import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor


MODEL_DIR = Path("models")
DEFAULT_MANIFEST_GLOB = "run_manifest_*.json"


def latest_file(pattern: str) -> Path | None:
    files = sorted(MODEL_DIR.glob(pattern))
    return files[-1] if files else None


def load_manifest(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_xgb_model(path: str | Path) -> XGBRegressor:
    model = XGBRegressor()
    model.load_model(str(path))
    return model


def apply_cop_guardrail(pred_elec_w: np.ndarray, pred_heat_w: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    eps = 1e-6
    elec = np.maximum(pred_elec_w.astype(float), 0.0)
    heat = np.maximum(pred_heat_w.astype(float), 0.0)

    elec_kwh = elec / 1000.0
    heat_kwh = heat / 1000.0

    valid = elec_kwh > eps
    max_heat = np.zeros_like(heat_kwh)
    max_heat[valid] = elec_kwh[valid] * 8.0

    heat_clipped_kwh = np.clip(heat_kwh, 0.0, max_heat)
    heat_clipped = heat_clipped_kwh * 1000.0
    changed = int(np.count_nonzero(np.abs(heat_clipped - heat) > 1e-6))
    return elec, heat_clipped, changed


def build_metrics(df_out: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}

    if "heatpump_elec" in df_out.columns:
        y_true_elec = df_out["heatpump_elec"].astype(float)
        y_pred_elec = df_out["pred_elec_w"].astype(float)
        metrics["r2_elec"] = float(r2_score(y_true_elec, y_pred_elec))
        metrics["mae_elec_w"] = float(mean_absolute_error(y_true_elec, y_pred_elec))

        true_elec_kwh = float(y_true_elec.sum() / 1000.0)
        pred_elec_kwh = float(y_pred_elec.sum() / 1000.0)
        metrics["energy_err_pct"] = float(abs(pred_elec_kwh - true_elec_kwh) / max(true_elec_kwh, 1e-6) * 100.0)

    if "heatpump_heat" in df_out.columns:
        y_true_heat = df_out["heatpump_heat"].astype(float)
        y_pred_heat = df_out["pred_heat_w"].astype(float)
        metrics["r2_heat"] = float(r2_score(y_true_heat, y_pred_heat))
        metrics["mae_heat_w"] = float(mean_absolute_error(y_true_heat, y_pred_heat))

        true_heat_kwh = float(y_true_heat.sum() / 1000.0)
        pred_heat_kwh = float(y_pred_heat.sum() / 1000.0)
        metrics["heat_err_pct"] = float(abs(pred_heat_kwh - true_heat_kwh) / max(true_heat_kwh, 1e-6) * 100.0)

    if "heatpump_elec" in df_out.columns and "heatpump_heat" in df_out.columns:
        true_elec_kwh = float(df_out["heatpump_elec"].sum() / 1000.0)
        pred_elec_kwh = float(df_out["pred_elec_w"].sum() / 1000.0)
        true_heat_kwh = float(df_out["heatpump_heat"].sum() / 1000.0)
        pred_heat_kwh = float(df_out["pred_heat_w"].sum() / 1000.0)

        true_cop = true_heat_kwh / max(true_elec_kwh, 1e-6)
        pred_cop = pred_heat_kwh / max(pred_elec_kwh, 1e-6)
        metrics["cop_err_pct"] = float(abs(pred_cop - true_cop) / max(true_cop, 1e-6) * 100.0)

    return metrics


def call_gemini(api_key: str, model_name: str, prompt: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt,
                    }
                ]
            }
        ]
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    candidates = data.get("candidates", [])
    if not candidates:
        return "No response candidates returned by Gemini API."

    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        return "Gemini returned an empty content block."
    return "\n".join(p.get("text", "") for p in parts).strip()


st.set_page_config(page_title="Heat Pump Forecast Interface", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1.2rem;}
.metric-card {padding: 0.7rem 0.9rem; border: 1px solid #e3e6eb; border-radius: 10px; background: #f9fbff;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Heat Pump Forecast Interface")
st.caption("Dual-target inference + COP guardrails + Gemini analysis assistant")

manifest_default = latest_file(DEFAULT_MANIFEST_GLOB)
if not manifest_default:
    st.error("No run manifest found in models/. Train a model first with src/04_train_model.py.")
    st.stop()

manifest_files = sorted(MODEL_DIR.glob(DEFAULT_MANIFEST_GLOB))
manifest_choice = st.sidebar.selectbox(
    "Manifest",
    options=manifest_files,
    index=len(manifest_files) - 1,
    format_func=lambda p: p.name,
)

manifest = load_manifest(manifest_choice)
model_paths = manifest.get("models", {})
elec_model_path = model_paths.get("electricity")
heat_model_path = model_paths.get("heat")

if not elec_model_path or not heat_model_path:
    st.error("Manifest is missing model paths for electricity and heat.")
    st.stop()

elec_features = manifest.get("features", {}).get("electricity", [])
heat_features = manifest.get("features", {}).get("heat", [])
if not elec_features or not heat_features:
    st.error("Manifest is missing feature lists. Re-run training to generate a complete manifest.")
    st.stop()

st.sidebar.markdown("### Data Input")
uploaded = st.sidebar.file_uploader("Upload CSV for inference", type=["csv"])
standby_floor = st.sidebar.number_input("Standby floor in OFF state (W)", value=50.0, min_value=0.0, step=5.0)

if uploaded is None:
    st.info("Upload a feature CSV to run inference. Include all manifest features for both electricity and heat models.")
else:
    df = pd.read_csv(uploaded)
    st.write("Input preview", df.head())

    missing_elec = [c for c in elec_features if c not in df.columns]
    missing_heat = [c for c in heat_features if c not in df.columns]
    if missing_elec or missing_heat:
        st.error("Input CSV is missing required features.")
        st.write("Missing electricity features", missing_elec)
        st.write("Missing heat features", missing_heat)
    elif st.button("Run Forecast", type="primary"):
        with st.spinner("Loading models and running predictions..."):
            elec_model = load_xgb_model(elec_model_path)
            heat_model = load_xgb_model(heat_model_path)

            pred_elec = elec_model.predict(df[elec_features])
            pred_heat = heat_model.predict(df[heat_features])

            if "heating_on" in df.columns:
                off_mask = df["heating_on"].astype(float).fillna(0.0).to_numpy() == 0.0
                pred_elec = pred_elec.astype(float)
                pred_heat = pred_heat.astype(float)
                pred_elec[off_mask] = standby_floor
                pred_heat[off_mask] = 0.0

            pred_elec, pred_heat, guardrail_changes = apply_cop_guardrail(pred_elec, pred_heat)

            eps = 1e-6
            pred_cop = np.divide(
                pred_heat / 1000.0,
                np.maximum(pred_elec / 1000.0, eps),
                out=np.full_like(pred_heat, np.nan, dtype=float),
                where=pred_elec > eps,
            )

            df_out = df.copy()
            df_out["pred_elec_w"] = pred_elec
            df_out["pred_heat_w"] = pred_heat
            df_out["pred_elec_kwh"] = pred_elec / 1000.0
            df_out["pred_heat_kwh"] = pred_heat / 1000.0
            df_out["pred_cop"] = pred_cop
            df_out["pred_cop_valid"] = (df_out["pred_cop"] >= 0.0) & (df_out["pred_cop"] <= 8.0)

            metrics = build_metrics(df_out)
            st.session_state["last_metrics"] = metrics
            st.session_state["last_rows"] = len(df_out)
            st.session_state["last_guardrail_changes"] = guardrail_changes

        st.success("Forecast complete.")

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='metric-card'><b>Rows</b><br>{len(df_out):,}</div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><b>COP Guardrail Changes</b><br>{guardrail_changes}</div>", unsafe_allow_html=True)
        c3.markdown(
            f"<div class='metric-card'><b>Physical COP Valid</b><br>{df_out['pred_cop_valid'].mean() * 100:.2f}%</div>",
            unsafe_allow_html=True,
        )

        if metrics:
            st.subheader("Evaluation Metrics")
            st.json(metrics)

        st.subheader("Forecast Output")
        st.dataframe(df_out.head(50), use_container_width=True)

        st.subheader("Predicted Power Trends")
        plot_cols = ["pred_elec_w", "pred_heat_w"]
        if "timestamp" in df_out.columns:
            trend = df_out[["timestamp"] + plot_cols].copy()
            trend = trend.set_index("timestamp")
            st.line_chart(trend)
        else:
            st.line_chart(df_out[plot_cols])

        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Forecast CSV",
            data=csv_bytes,
            file_name="forecast_output.csv",
            mime="text/csv",
        )


st.divider()
st.subheader("Gemini Analysis Assistant")
st.caption("Use Gemini API to explain forecast behavior, diagnose errors, and suggest next experiments.")

api_key = st.text_input("Gemini API Key", type="password")
gemini_model = st.selectbox("Gemini Model", ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-2.5-flash"])
user_prompt = st.text_area(
    "Analysis request",
    value="Review these model results and suggest the top 3 improvements to raise both R2 values above 0.80 without introducing leakage.",
    height=110,
)

if st.button("Ask Gemini"):
    if not api_key:
        st.error("Enter a Gemini API key.")
    else:
        context = {
            "rows": st.session_state.get("last_rows"),
            "guardrail_changes": st.session_state.get("last_guardrail_changes"),
            "metrics": st.session_state.get("last_metrics", {}),
        }
        full_prompt = (
            "You are an ML engineer reviewing a heat pump dual-target model. "
            "Prioritize leakage-safe improvements and physical plausibility.\n\n"
            f"Current context: {json.dumps(context, indent=2)}\n\n"
            f"User request: {user_prompt}"
        )
        try:
            with st.spinner("Querying Gemini API..."):
                response = call_gemini(api_key, gemini_model, full_prompt)
            st.markdown("### Gemini Response")
            st.write(response)
        except Exception as exc:
            st.error(f"Gemini API call failed: {exc}")
