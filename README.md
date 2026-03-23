# Heat Pump Forecast Workspace

This project contains a full pipeline for heat pump forecasting and a Streamlit interface for side-by-side model iteration.

## Run Pipeline

```powershell
.venv\Scripts\python.exe src/01_fetch_data.py
.venv\Scripts\python.exe src/02_feature_engineering.py
.venv\Scripts\python.exe src/03_clean_data.py
.venv\Scripts\python.exe src/04_train_model.py
```

## Run Interface (Streamlit)

```powershell
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

## Interface Features

- Loads latest training manifest from `models/run_manifest_*.json`
- Runs dual-target predictions (`pred_elec_w`, `pred_heat_w`)
- Applies COP physics guardrail (`0 <= COP <= 8`)
- Adds production-friendly outputs (`pred_elec_kwh`, `pred_heat_kwh`, `pred_cop`)
- Computes optional evaluation metrics when actual columns are present
- Integrates Gemini API for model diagnostics and experiment suggestions

## Gemini API Setup

1. Open the Streamlit interface.
2. Enter your Gemini API key in the Gemini section.
3. Select a Gemini model and submit a prompt.

The app sends requests to:

`https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`

