---
title: Heat Pump Forecast Workspace
sdk: streamlit
sdk_version: 1.44.0
python_version: 3.10
app_file: app.py
---

# Heat Pump Forecast Workspace

This project contains a full heat pump forecasting pipeline, an admin AI Ops Streamlit app, and a customer-facing Streamlit app.

## Documentation Map

Use these guides for complete setup, operations, and code understanding:

- [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)
- [docs/CUSTOMER_APP_DETAILED_DOCUMENTATION.md](docs/CUSTOMER_APP_DETAILED_DOCUMENTATION.md)
- [docs/SOURCE_CODE_GUIDE.md](docs/SOURCE_CODE_GUIDE.md)
- [docs/UI_INPUT_SPECIFICATIONS.md](docs/UI_INPUT_SPECIFICATIONS.md)
- [docs/PIPELINE_TEACHING_GUIDE.md](docs/PIPELINE_TEACHING_GUIDE.md)
- [PRODUCTION_READINESS_AUDIT.md](PRODUCTION_READINESS_AUDIT.md) (historical snapshot audit)

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
$env:SYSTEM_IDS="<comma-separated series_id values from src/01_fetch_data.py>"
.venv\Scripts\python.exe src/01_fetch_data.py
.venv\Scripts\python.exe src/02_feature_engineering.py
.venv\Scripts\python.exe src/03_clean_data.py
.venv\Scripts\python.exe src/04_train_model.py
```

Train on a smaller subset than the raw data (optional):

```powershell
$env:TRAIN_SYSTEM_IDS="<optional subset of SYSTEM_IDS>"
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
.venv\Scripts\python.exe -m streamlit run app.py
```

- `app.py` is the Hugging Face Spaces entrypoint and defaults to the customer interface.
- Set `SPACE_APP_MODE=admin` to run the admin console through `app.py`.

## Run Admin Interface (Direct)

```powershell
.venv\Scripts\python.exe -m streamlit run admin_app.py
```

## Run Customer Interface (Streamlit)

```powershell
.venv\Scripts\python.exe -m streamlit run customer_app.py
```

- `admin_app.py` is the full admin and operations console.
- `customer_app.py` is the customer-facing interface with guided inputs and a simplified experience.

## Interface Features

### Admin App (`admin_app.py`)

- Pipeline execution with stage-level controls
- Artifact explorer and model health dashboards
- Live batch scoring and single-row scoring tools
- System metadata lookup tab
- Gemini analysis with selectable model run context

### Customer App (`customer_app.py`)

- Guided instant estimate flow
- Customer-safe result cards (cost, usage, COP)
- AI Briefing with broad presets and Custom request option
- Staff tools behind `CUSTOMER_STAFF_PASSWORD`
- Appearance, currency, and default unit-rate controls

## Gemini API Setup

1. Set `GEMINI_API_KEY` in `.env`.
2. Optionally set `GEMINI_MODEL` in `.env`.
3. Launch either Streamlit app and use the Gemini-powered section.

Admin app allows entering key/model in UI as well.

The app sends requests to:

`https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`