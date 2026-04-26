# Complete Project Flow

This document explains the end-to-end flow of the Heat Pump Monitor project, including data ingestion, training, prediction, and Hugging Face Spaces deployment.

## 1. Project Purpose

The project predicts two targets for heat pump operation:

- Electricity demand in watts: heatpump_elec
- Heat output in watts: heatpump_heat

It also predicts runtime ON/OFF state and applies physical safety guardrails to keep predictions realistic.

## 2. High-Level Architecture

Main folders:

- src: pipeline stages and model logic
- data/raw: fetched API snapshots (local working data)
- data/processed: engineered and cleaned datasets plus UI outputs
- models: trained artifacts and manifests used by inference
- docs: operating and implementation documentation
- app.py, customer_app.py, admin_app.py: Streamlit app entrypoints

Pipeline order:

1. src/01_fetch_data.py
2. src/02_feature_engineering.py
3. src/03_clean_data.py
4. src/04_train_model.py
5. src/05_predict_model.py

## 3. Data Ingestion Flow (Stage 01)

File: src/01_fetch_data.py

What it does:

- Pulls hourly telemetry from HeatPumpMonitor API for configured systems.
- Supports fixed date window and max available fallback mode.
- Aligns each system to an hourly UTC calendar.
- Writes one raw CSV per system.
- Writes fetch audit files.

Inputs:

- Environment controls like SYSTEM_IDS, FETCH_MODE, START_DATE, END_DATE

Outputs:

- data/raw/system_<id>_raw.csv
- data/raw/fetch_log.csv
- data/raw/fetch_quality_audit.csv

## 4. Feature Engineering Flow (Stage 02)

File: src/02_feature_engineering.py

What it does:

- Loads all system raw CSV files.
- Adds energy, COP, and heating runtime flags.
- Adds physics-based features (deltaT, lift, demand proxies).
- Adds temporal cyclical features (hour and month sin/cos).
- Adds lag features for short, daily, and weekly memory.
- Adds rolling weather context.

Input:

- data/raw/system_*_raw.csv

Output:

- data/processed/daikin_features.csv

## 5. Data Cleaning Flow (Stage 03)

File: src/03_clean_data.py

What it does:

- Interpolates only selected telemetry columns for short gaps.
- Applies boundary filters using physical ranges (COP, deltaT, outside temperature).
- Drops rows with missing core targets only.
- Generates per-system retention report.

Input:

- data/processed/daikin_features.csv

Outputs:

- data/processed/daikin_clean.csv
- data/processed/cleaning_report.csv

## 6. Training Flow (Stage 04)

File: src/04_train_model.py

### 6.1 Training Setup

- Loads cleaned data and enforces chronological sort.
- Builds two feature sets:
  - electricity feature set
  - heat feature set
- Feature policy modes:
  - strict_production
  - enhanced_onestep

### 6.2 Split and Preprocessing

- Time-based split (no random split).
- Split integrity check prevents timestamp leakage.
- Median imputation fitted on training data only.

### 6.3 Runtime Classifier + Dual Regressors

- Trains runtime ON/OFF classifier (XGBClassifier).
- Trains two target regressors (XGBRegressor):
  - electricity model
  - heat model

### 6.4 Strategy Search

Compares multiple target strategies:

- none
- log1p_both
- cop_ratio_heat

Picks best strategy by total MAE score.

### 6.5 Walk-Forward Tuning

Optional forward-chaining tuning for both targets using TimeSeriesSplit.

### 6.6 Calibration and Guardrails

- Learns per-slice multipliers by system and capacity.
- Applies COP and non-negative power guardrails during evaluation.

### 6.7 Saved Artifacts

The training run writes:

- models/xgb_elec_v<run_tag>.json
- models/xgb_heat_v<run_tag>.json
- models/xgb_runtime_clf_v<run_tag>.joblib
- models/imputer_elec_v<run_tag>.joblib
- models/imputer_heat_v<run_tag>.joblib
- models/feature_schema_v<run_tag>.json
- models/pipeline_meta_v<run_tag>.json
- models/slice_calibrators_v<run_tag>.json
- models/run_manifest_<run_tag>.json
- models/training_report.txt

## 7. Prediction Flow (Stage 05)

File: src/05_predict_model.py

What it does:

1. Resolves latest run tag from models/run_manifest_*.json
2. Loads full artifact bundle:
   - feature schema
   - dual regressors
   - runtime classifier
   - imputers
   - optional slice calibrators
3. Validates incoming inference frame against schema.
4. Applies imputers and runs predictions.
5. Applies selected inverse target strategy.
6. Applies runtime OFF forcing and COP guardrail.
7. Emits final scored dataframe with:
   - runtime_on_proba
   - runtime_on_pred
   - pred_heatpump_elec
   - pred_heatpump_heat
   - pred_cop
8. Writes sample output and monitoring summary when run as script.

Script outputs:

- data/processed/sample_predictions.csv
- models/latest_monitoring_summary.json

## 8. Streamlit App Flow

Entrypoint: app.py

- Default mode routes to customer_app.py
- Set SPACE_APP_MODE=admin to route to admin_app.py

### Customer App

File: customer_app.py

- Guided single prediction UX
- Uses saved model bundle for inference
- Persists prediction history in local SQLite for lag simulation

### Admin App

File: admin_app.py

- Full pipeline control from UI
- Artifact browsing
- Model health and scoring tools
- Single row prediction editor

## 9. Run Commands

From repository root:

PowerShell:

.venv\Scripts\python.exe src/01_fetch_data.py
.venv\Scripts\python.exe src/02_feature_engineering.py
.venv\Scripts\python.exe src/03_clean_data.py
.venv\Scripts\python.exe src/04_train_model.py
.venv\Scripts\python.exe src/05_predict_model.py

Run app:

.venv\Scripts\python.exe -m streamlit run app.py

## 10. Hugging Face Spaces Readiness

This repository is configured for Spaces with:

- README metadata block (title, sdk, app file)
- app.py as runtime entrypoint
- requirements.txt with dependencies
- .github/workflows/sync.yml for automated hub sync
- .gitattributes configured for model files via Git LFS

Current recommended upload set for Spaces:

- app.py, customer_app.py, admin_app.py
- src/*.py
- models/* latest run artifacts
- requirements.txt
- README.md
- docs (optional)

Excluded or local-only data should remain outside pushed Space snapshots.
