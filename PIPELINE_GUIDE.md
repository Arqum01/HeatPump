# Streamlit App Guide (streamlit_app.py)

This guide is only for running and using the Streamlit application in `streamlit_app.py`.

## 1. What This App Does

The app is an AI Ops console for your heat pump project. It provides:

- Pipeline execution (run each stage from UI)
- Artifact explorer (CSV/JSON/TXT previews)
- Model health dashboard
- Live batch scoring
- Single-row prediction editor
- Gemini-assisted analysis

Tabs inside the app:

1. Pipeline
2. Artifacts
3. Model Health
4. Live Scoring
5. Single Input
6. Gemini

## 2. Prerequisites

- Windows PowerShell
- Python virtual environment exists in `.venv`
- Dependencies already installed for Streamlit and project modules
- Open terminal in project root:

```powershell
D:\Client Projects\Heat Pump Monitor\HeatPump
```

## 3. Start the App

Activate environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run Streamlit:

```powershell
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`). Open it in your browser.

## 4. Global UI Behavior

### 4.1 Sidebar theme

In the left sidebar, choose:

- Auto
- Light
- Dark

### 4.2 Session state remembers

The app stores runtime state while the session is open:

- Stage run history
- Latest scored dataframe
- Latest prediction summary
- Single-input editor values

## 5. Pipeline Tab (Main Automation)

Use this tab to run pipeline scripts without manual terminal commands.

### 5.1 Controls

- Stage timeout (seconds)
- Stop full pipeline on first error
- Train policy for stage 04:
  - `enhanced_onestep`
  - `strict_production`
  - `both` (runs stage 04 twice)

### 5.2 Custom Pipeline Options

#### System selection

- `SYSTEM_IDS` (optional): shared filter for fetch/features/training
- `TRAIN_SYSTEM_IDS` (optional): training-only subset filter

Examples:

- `SYSTEM_IDS = 615,44,228`
- `TRAIN_SYSTEM_IDS = 615,44`

#### 01 Fetch Data options

- Fetch mode (`fixed_window` or `max_available`)
- Start date, End date
- API max datapoints
- Systems config (`system_id:capacity_kw` format)
- Default capacity for unknown selected IDs

#### 02 Feature Engineering options

- Standby threshold (W)
- Base temperature

#### 03 Clean Data options

- COP min/max
- Outside temperature min/max

#### 07 Backtest options

- Backtest feature policy mode

### 5.3 Run actions

- Run Full Pipeline
- Run any Individual Stage button
- Clear Run History

### 5.4 Execution log

For each run, you get:

- Status badge (SUCCESS/FAILED)
- Return code and duration
- Expandable stdout/stderr
- Downloadable consolidated logs

## 6. Artifacts Tab

Use this tab to browse generated files from:

- `data/raw`
- `data/processed`
- `models`

Supported previews:

- CSV preview with optional line chart for numeric columns
- JSON pretty view
- Text view for `.txt`/`.md`

## 7. Model Health Tab

Reads latest `models/run_manifest_*.json` and displays:

- R2 and MAE for electricity and heat
- Energy/Heat/COP error percentages
- Strategy leaderboard
- Gate status payload
- Backtest folds if available (`models/walk_forward_folds.csv`)

## 8. Live Scoring Tab

Purpose: score a batch CSV using trained model bundles.

### 8.1 Steps

1. Upload inference CSV.
2. Choose feature policy filter (`enhanced_onestep`, `strict_production`, or `all`).
3. Choose a run tag.
4. Click **Score Batch**.

### 8.2 Output

- Preview of scored rows
- Metrics cards: rows, predicted ON rate, COP guardrail adjustments
- Trend chart when timestamp and prediction columns are available
- Download scored CSV
- Saved file in `data/processed/ui_predictions_YYYYMMDD_HHMMSS.csv`

## 9. Single Input Tab

Purpose: predict one row interactively.

### 9.1 Steps

1. Pick policy filter and run tag.
2. Edit one-row feature table (auto-generated from feature schema).
3. Click **Predict Single Row**.

### 9.2 Output

- Predicted electricity (W)
- Predicted heat (W)
- Predicted COP
- Full scored row table

## 10. Gemini Tab

Purpose: ask LLM-based diagnostics on manifests, monitoring, and predictions.

### 10.1 API key setup

You can set key in `.env`:

```env
GEMINI_API_KEY=your_key_here
```

Or paste key in UI input.

### 10.2 Features

- Ask Gemini with project context (manifest + monitoring)
- Analyze latest prediction output with an expert prompt template

## 11. Recommended Workflow (UI-Only)

1. Open app.
2. Go to **Pipeline** tab.
3. Set `SYSTEM_IDS` and optional `TRAIN_SYSTEM_IDS`.
4. Run full pipeline.
5. Check **Model Health** for metrics and gates.
6. Use **Live Scoring** for batch inference.
7. Use **Gemini** for analysis and next-step suggestions.

## 12. Common File Outputs Used by App

- `models/run_manifest_*.json`
- `models/training_report.txt`
- `models/latest_monitoring_summary.json`
- `models/walk_forward_summary.json`
- `data/processed/ui_predictions_*.csv`

## 13. Troubleshooting (Streamlit App)

### 13.1 App does not start

- Confirm venv activation.
- Run from project root.
- Ensure Streamlit is installed in `.venv`.

### 13.2 No pipeline stages appear

- Confirm `src` exists and has stage files (`01_*.py`, etc.).

### 13.3 Training run tags not visible in scoring tabs

- Run training stage successfully from Pipeline tab first.
- Verify `models/run_manifest_*.json` exists.

### 13.4 "No rows found for selected SYSTEM_IDS"

- Check IDs are valid and comma-separated.
- Remove system filters and rerun to verify base flow.

### 13.5 Gemini does not respond

- Verify `GEMINI_API_KEY` is valid.
- Check internet connection and selected model.

## 14. Quick Start (Copy/Paste)

```powershell
.\.venv\Scripts\Activate.ps1
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Then in app: Pipeline -> set system filters -> Run Full Pipeline.
