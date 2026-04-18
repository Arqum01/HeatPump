# Heat Pump Monitor Source Code Guide

## Purpose
This document explains core Python source files and key functions across the training pipeline and Streamlit apps.
It is designed for fast onboarding, safe edits, and easier debugging.

## Project Flow
1. `01_fetch_data.py`: Pull raw telemetry from API and write per-system raw CSV files.
2. `02_feature_engineering.py`: Build model features from raw telemetry.
3. `03_clean_data.py`: Apply physics and quality filters to produce training-ready data.
4. `04_train_model.py`: Train runtime classifier plus dual regressors, evaluate, and save artifacts.
5. `05_predict_model.py`: Load artifacts and run batch inference.
6. `monitoring_model_06.py`: Build monitoring summaries from prediction runs.
7. `backtest_model_07.py`: Run walk-forward backtesting and slice diagnostics.
8. `streamlit_app.py`: Admin/ops console for running the pipeline and analyzing outputs.
9. `customer_app.py`: Customer-facing estimator with guided inputs and AI briefing.
10. `api/index.py`: Vercel-ready serverless API for production inference.

---

## `src/01_fetch_data.py`
### Module Role
Ingestion module that downloads hourly telemetry for each configured system, aligns data to a strict UTC hourly calendar, and writes a fetch audit log.

### Global Configuration
- `SYSTEMS`: Systems to fetch. Each item includes `system_id` and `capacity_kw`.
- `FEEDS`: Telemetry channels requested from API.
- `FETCH_MODE`: `fixed_window` or `max_available`.
- `START`, `END`, `VERY_EARLY_START`: Date controls.
- `API_MAX_DATAPOINTS`: Datapoint limit guard to avoid oversize requests.

### Functions
#### `coerce_feed_values(values) -> list`
- Purpose: Normalize API feed payload values into list form.
- Input: Any raw payload shape (list, tuple, dict, scalar, `None`).
- Output: A list used by downstream padding/trimming logic.
- Logic:
  1. Convert `None` to empty list.
  2. Pass through list and tuple as list.
  3. Convert dict payloads to list of values.
  4. Wrap scalar values in a single-item list.

#### `estimate_requested_datapoints(start_date, end_date, interval_seconds) -> int`
- Purpose: Estimate how many datapoints a request will ask the API for.
- Input: Date strings in `DD-MM-YYYY` and interval in seconds.
- Output: Integer estimate of datapoints per feed.
- Logic:
  1. Parse start and end to datetime.
  2. Compute positive duration in seconds.
  3. Divide by interval.

#### `payload_has_explicit_limit_error(raw_json) -> bool`
- Purpose: Detect explicit provider limit text in returned payload.
- Input: API response JSON object.
- Output: `True` if payload includes known limit message.
- Logic:
  1. Iterate configured feeds.
  2. Normalize each feed payload to list.
  3. Search string values for the limit token.

#### `resolve_date_window() -> tuple[str, str]`
- Purpose: Compute effective fetch window from current mode.
- Output: `(start, end)` date strings.
- Logic:
  1. Return configured `START/END` for `fixed_window`.
  2. For `max_available`, use `VERY_EARLY_START` and tomorrow UTC.

#### `fetch_system(client, system)`
- Purpose: Fetch one system asynchronously with size-aware fallback.
- Input: Shared `httpx.AsyncClient`, one system config dict.
- Output: Tuple with `(sid, system, payload_or_none, effective_start, error_or_none)`.
- Logic:
  1. Build request params.
  2. Estimate datapoints and fallback to `START` when estimate exceeds API limit.
  3. Request data.
  4. If explicit limit error appears in payload, retry fallback start.
  5. Return payload or captured exception details.

#### `build_dataframe(system, raw_json, data_start) -> pd.DataFrame`
- Purpose: Convert payload to aligned hourly DataFrame.
- Input: System metadata, raw payload, effective start date.
- Output: UTC-indexed DataFrame with all feeds and metadata columns.
- Logic:
  1. Build UTC hourly index from `data_start` to resolved end.
  2. For each feed, trim/pad to exact index length.
  3. Attach `system_id` and `capacity_kw`.
  4. In `max_available` mode, trim leading rows before first useful signal.

#### `main()`
- Purpose: Orchestrate concurrent ingestion and persistence.
- Output: Raw per-system CSV files and `fetch_log.csv`.
- Logic:
  1. Create async tasks for all systems.
  2. Build/serialize per-system frames.
  3. Compute quality stats (`missing_elec_pct`).
  4. Save audit log.

---

## `src/02_feature_engineering.py`
### Module Role
Transforms raw telemetry into model-ready features: energy metrics, thermal physics features, temporal encodings, lag memory, and system slice indicators.

### Functions
#### `load_all_systems() -> pd.DataFrame`
- Loads all `system_*_raw.csv` files.
- Concatenates and sorts by `system_id`, `timestamp`.

#### `apply_standby_filter(df) -> pd.DataFrame`
- Sets `heatpump_heat` and `heatpump_flowrate` to zero where electric draw is below standby threshold.

#### `add_energy_metrics(df) -> pd.DataFrame`
- Adds `elec_kwh`, `heat_kwh`, `cop`, and binary `heating_on`.

#### `add_physics_features(df) -> pd.DataFrame`
- Adds `deltaT_house`, `deltaT_lift`, and `temp_deficit`.

#### `add_enhanced_physics_features(df) -> pd.DataFrame`
- Adds size-normalized load feature `load_ratio`.
- Adds `hdh` (heating degree hours).

#### `add_temporal_features(df) -> pd.DataFrame`
- Adds raw temporal features (`hour`, `month`, `day_of_week`).
- Adds cyclical harmonics (`hour_sin/cos`, `month_sin/cos`).
- Adds `is_heating_season` from outside temperature.

#### `add_lag_features(df) -> pd.DataFrame`
- Adds lag-1 features grouped by `system_id` to prevent cross-system leakage.

#### `add_extended_lag_features(df) -> pd.DataFrame`
- Adds lag-24 and lag-168 memory features.
- Adds thermal lag interactions and run block duration (`run_hours`).
- Adds defrost memory (`cop_lag1`, `was_defrost_lag1`) and normalized lag (`elec_lag1_pct`).

#### `add_metadata_features(df) -> pd.DataFrame`
- One-hot encodes `system_id` into `system_*` features.

#### `add_rolling_features(df) -> pd.DataFrame`
- Adds `outsideT_3h_avg` using per-system rolling mean.

#### `main()`
- Runs all transforms in dependency-safe order and saves `daikin_features.csv`.

---

## `src/03_clean_data.py`
### Module Role
Applies quality-control and physics filters to engineered features, then writes cleaned data and a retention report.

### Functions
#### `smart_interpolate(df) -> pd.DataFrame`
- Performs per-system linear interpolation with `limit=1` on selected sensor columns.

#### `apply_boundary_rules(df) -> pd.DataFrame`
- Enforces non-negative energy, COP range, delta-T range, and outside temperature range.

#### `selective_dropna(df) -> pd.DataFrame`
- Drops rows only when core targets are missing.

#### `generate_report(df_before, df_after) -> pd.DataFrame`
- Creates per-system retention and quality statistics and saves CSV report.

#### `main()`
- Executes interpolation, filters, selective null handling, save, and reporting.

---

## `src/04_train_model.py`
### Module Role
Main training pipeline: feature policy resolution, split integrity checks, tuning, strategy testing, slice calibration, evaluation, and artifact persistence.

### Helper/Configuration Functions
#### `model_hyperparams(label) -> dict`
- Baseline XGBoost hyperparameters per target.

#### `load_data() -> (df, elec_features, heat_features)`
- Loads cleaned data, validates required columns, and returns feature sets.

#### `build_feature_lists(df) -> (elec_features, heat_features)`
- Builds feature sets according to `FEATURE_POLICY_MODE` and available `system_*` metadata columns.

#### `build_feature_schema(df, elec_features, heat_features) -> dict`
- Builds serving schema with ordered features and inferred dtypes.

#### `validate_inference_frame(df, feature_schema, allow_extra_columns=True) -> dict`
- Coerces inference dtypes and returns validated full and target-specific frames.

#### `time_split(df, train_ratio=0.8)`
- Chronological split by timestamp boundary.

#### `validate_split_integrity(train_df, test_df) -> dict`
- Verifies no overlap and strictly ordered train/test periods.

#### `impute_train_test(X_train, X_test)`
- Fits median imputer on train and applies to train/test.

### Tuning and Strategy Functions
#### `tuning_candidates(label) -> list[dict]`
- Generates compact candidate hyperparameter variants around baseline.

#### `walk_forward_tune_target(X_train, y_train, label) -> (best_params, rows)`
- Forward-chaining CV tuning for a single target.

#### `strategy_train_predict(...) -> dict`
- Trains and predicts with one target strategy:
  - `none`
  - `log1p_both`
  - `cop_ratio_heat`

### Slice Calibration and Diagnostics
#### `build_slice_calibrators(train_on_df, pred_elec_on, pred_heat_on) -> dict`
- Learns per `(system_id, capacity_kw)` multiplicative corrections.

#### `apply_slice_calibrators(df, pred_elec_w, pred_heat_w, calibrators) -> (elec, heat)`
- Applies slice multipliers at scoring time.

#### `slice_error_analysis(df, y_elec_true, y_heat_true, pred_elec_w, pred_heat_w) -> pd.DataFrame`
- Computes per-slice MAE and R2 diagnostics.

### Model Training and Evaluation
#### `save_pipeline_artifacts(...) -> dict`
- Saves runtime classifier, imputers, metadata, and slice calibrator files.

#### `train_single_target(...) -> (model, preds)`
- Trains one regressor and returns predictions.

#### `train_runtime_classifier(...) -> (clf, on_pred, on_proba)`
- Trains binary runtime ON/OFF classifier.

#### `classifier_diagnostics(y_true_on, y_pred_on) -> dict`
- Computes confusion matrix and classification report summary.

#### `apply_cop_guardrail(pred_elec_w, pred_heat_w, cop_min=0.0, cop_max=8.0)`
- Enforces physical prediction constraints.

#### `evaluate_dual(y_elec_test, pred_elec_w, y_heat_test, pred_heat_w, on_pred) -> dict`
- Computes pointwise and aggregate business/science metrics.

#### `threshold_status(metrics) -> dict`
- Evaluates metrics against threshold gates.

#### `build_report(...) -> str`
- Creates the human-readable training report text.

#### `main()`
- Full orchestration from loading data to writing models/report/manifest.

---

## `src/05_predict_model.py`
### Module Role
Inference pipeline that loads trained artifacts, validates inputs, reconstructs target-space predictions, applies slice calibration and physical guardrails, and returns scored data.

### Functions
#### `resolve_latest_run_tag() -> str`
- Finds the latest available manifest tag.

#### `apply_cop_guardrail(...)`
- Same physical correction logic used in training evaluation.

#### `validate_inference_frame(df, feature_schema, allow_extra_columns=True) -> dict`
- Schema validation and dtype coercion for serving inputs.

#### `load_bundle(run_tag) -> dict`
- Loads manifest, schema, models, imputers, strategy metadata, and optional slice calibrators.

#### `apply_slice_calibrators(df, pred_elec, pred_heat, calibrators)`
- Applies per-slice multiplier corrections to inference outputs.

#### `predict_bundle(df, run_tag) -> pd.DataFrame`
- End-to-end scoring flow:
  1. Validate schema.
  2. Impute features.
  3. Predict runtime ON/OFF.
  4. Reconstruct targets by selected strategy.
  5. Apply slice calibrators.
  6. Apply standby forcing and COP guardrail.
  7. Append output columns.

---

## `src/monitoring_model_06.py`
### Module Role
Produces post-scoring health summary for prediction batches, with basic drift and alert rules.

### Functions
#### `summarize_batch_monitoring(input_df, prediction_df, reference_df=None) -> dict`
- Creates a monitoring payload with:
  - input/output row stats
  - top null-rate features
  - prediction ranges and runtime rates
  - guardrail adjustments
  - optional drift z-scores
  - triggered alerts

#### `save_monitoring_summary(summary, output_path)`
- Writes monitoring summary JSON to disk.

---

## `src/backtest_model_07.py`
### Module Role
Runs walk-forward backtesting and slice diagnostics to estimate temporal generalization and heterogeneity.

### Functions
#### `model_hyperparams(label) -> dict`
- Baseline model parameters per target.

#### `build_feature_lists(df)`
- Feature policy resolver for backtest setup.

#### `load_data()`
- Loads cleaned data and validates required fields.

#### `apply_cop_guardrail(...)`
- Physical correction on backtest predictions.

#### `train_runtime_classifier(...)`
- Trains fold-specific runtime classifier.

#### `train_single_target(...)`
- Trains fold-specific regressor for one target.

#### `evaluate_fold(test_df, pred_elec_w, pred_heat_w, on_pred)`
- Fold evaluation metrics for pointwise and aggregate behavior.

#### `slice_metrics(test_df, pred_elec_w, pred_heat_w)`
- Per-capacity and per-system slice diagnostics for each fold.

#### `make_walk_forward_folds(df, min_train_ratio=0.50, test_ratio=0.10, step_ratio=0.10)`
- Builds rolling-origin fold boundaries.

#### `run_backtest()`
- Executes fold loop, aggregates outputs, and saves:
  - `walk_forward_folds.csv`
  - `walk_forward_slices.csv`
  - `walk_forward_summary.json`

---

## `streamlit_app.py`
### Module Role
Admin and operations console for end-to-end model lifecycle activities.

### Key Responsibilities
- Discover and execute numbered pipeline stages.
- Provide custom runtime overrides via environment variables.
- Expose artifact browsing for CSV/JSON/text outputs.
- Provide Model Health, Live Scoring, Single Input, System Metadata, and Gemini tabs.
- Allow run-tag selection by feature policy in scoring and diagnostics areas.

### High-Impact Functions
- `discover_pipeline_stages()`: Finds runnable stages dynamically from src.
- `execute_stage(...)`: Applies per-stage env overrides and executes scripts.
- `render_pipeline_tab()`: Main automation workflow and log viewer.
- `render_model_tab()`: Run-selectable model diagnostics.
- `render_predict_tab()` and `render_single_input_tab()`: Batch and single-row inference.
- `render_system_metadata_tab()`: Public metadata lookup by system ID.
- `render_gemini_tab()`: Model-context-aware Gemini analysis.

---

## `customer_app.py`
### Module Role
Customer-facing interface for quick home-level estimates and plain-language AI guidance.

### Key Responsibilities
- Collect simplified inputs and derive full model-serving feature row.
- Execute predictions using the latest run manifest silently.
- Present customer-friendly cost and efficiency outputs.
- Offer AI Briefing with preset topics and a custom request path.
- Keep power-user tools behind staff access controls.

### High-Impact Functions
- `render_instant_estimate_tab(...)`: Main customer form and predict action.
- `build_single_row(...)`: Converts user input into full serving schema row.
- `render_single_prediction_result(...)`: Friendly results and technical details.
- `render_gemini_tab(...)`: AI Briefing with broad options + custom user question.
- `render_settings_bar(...)`: Appearance, currency, and unit-rate controls.
- `main()`: App orchestration and staff-gated feature exposure.

---

## `api/index.py`
### Module Role
Production inference API entrypoint designed for Vercel serverless deployment.

### Key Responsibilities
- Load model manifests from the active model directory.
- Expose health and run inventory endpoints.
- Support both technical row-level inference and simplified customer-style inference.
- Enforce schema-aligned row construction before scoring.

### High-Impact Endpoints
- `GET /health`: runtime status and active run metadata.
- `GET /runs`: available model runs in active model directory.
- `POST /predict/technical`: inference from provided serving feature rows.
- `POST /predict/customer`: simplified inference payload with derived model features.

---

## Recommended Reading Order
1. Start with `01_fetch_data.py` to understand ingestion contracts.
2. Move to `02_feature_engineering.py` and `03_clean_data.py` for dataset construction.
3. Read `04_train_model.py` for core modeling logic.
4. Read `05_predict_model.py` for serving parity.
5. Use `monitoring_model_06.py` and `backtest_model_07.py` for operational and validation context.
6. Read `streamlit_app.py` for admin/ops execution workflow.
7. Read `customer_app.py` with docs/CUSTOMER_APP_DETAILED_DOCUMENTATION.md for customer journey details.
8. Read `api/index.py` with docs/VERCEL_DEPLOYMENT_GUIDE.md for production API deployment.
