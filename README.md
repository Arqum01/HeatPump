# Heat Pump Forecast Workspace

This project contains a full pipeline for heat pump forecasting and a Streamlit interface for side-by-side model iteration.

## Full Guide

For complete setup, configuration, and troubleshooting instructions, see:

- [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)

## Run Pipeline

```powershell
.venv\Scripts\python.exe src/01_fetch_data.py
.venv\Scripts\python.exe src/02_feature_engineering.py
.venv\Scripts\python.exe src/03_clean_data.py
.venv\Scripts\python.exe src/04_train_model.py
```

## Custom System Selection

You can now run only systems you choose by setting environment variables.

Use the same IDs across fetch, feature engineering, and training:

```powershell
$env:SYSTEM_IDS="615,44,228"
.venv\Scripts\python.exe src/01_fetch_data.py
.venv\Scripts\python.exe src/02_feature_engineering.py
.venv\Scripts\python.exe src/03_clean_data.py
.venv\Scripts\python.exe src/04_train_model.py
```

Train on a smaller subset than the raw data (optional):

```powershell
$env:TRAIN_SYSTEM_IDS="615,44"
.venv\Scripts\python.exe src/04_train_model.py
```

If you want to fetch IDs that are not in the built-in defaults, set a fallback
capacity for unknown IDs:

```powershell
$env:SYSTEM_IDS="999,1001"
$env:DEFAULT_CAPACITY_KW="6"
.venv\Scripts\python.exe src/01_fetch_data.py
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

