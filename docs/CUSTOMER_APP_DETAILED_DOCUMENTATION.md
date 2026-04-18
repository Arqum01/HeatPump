# Customer App Detailed Documentation

## Purpose
This document explains the full customer-facing Streamlit app in customer_app.py.
It covers architecture, runtime flow, function-level behavior, configuration, and operational guidance.

The app is designed to:
- Let a homeowner generate a fast heat-pump estimate.
- Translate model outputs into customer-friendly cost and efficiency guidance.
- Provide optional AI briefing support with broad presets and custom questions.
- Keep internal tooling behind staff-only access.

## Scope
This document focuses on:
- customer_app.py
- Integration points with src/01_fetch_data.py and src/05_predict_model.py
- Environment variables used by the customer UI

It does not re-document the full training pipeline internals in depth. For those, use docs/SOURCE_CODE_GUIDE.md.

## File-Level Overview

### Main File
- customer_app.py: Complete customer UI, prediction orchestration, AI briefing, and staff-gated utilities.

### Referenced Files
- src/01_fetch_data.py: Source of default system IDs through DEFAULT_SYSTEMS.
- src/05_predict_model.py: Prediction bundle loader and scoring logic.
- models/run_manifest_*.json: Model run metadata and feature schema pointers.
- data/processed/: Output folder for scored CSV files.

## Runtime Architecture
The app is organized into five layers.

1. Configuration and bootstrap layer
- Loads .env values.
- Initializes Streamlit page config.
- Applies theme CSS based on appearance mode.

2. Data and model resolution layer
- Locates latest model manifest.
- Resolves feature schema from manifest.
- Dynamically loads prediction module.

3. Input and feature assembly layer
- Collects customer inputs.
- Builds a fully populated single-row feature frame.
- Adds derived thermal and temporal features.

4. Prediction and interpretation layer
- Calls predict_bundle with chosen run tag.
- Displays cost and efficiency metrics.
- Builds summary payload for AI briefing.

5. Optional assistant and staff layer
- AI Briefing section for preset or custom user questions.
- Staff-only area protected by password in environment.

## End-to-End User Flow
1. User opens the app.
2. App auto-loads latest model run and default UI settings.
3. User enters home conditions and optional history.
4. User clicks Predict Now.
5. App generates predictions and displays customer-friendly results.
6. User can ask AI Briefing by choosing a broad topic or custom question.
7. If staff password is configured, internal tabs become accessible after authentication.

## Global Constants
- ROOT: Workspace root path.
- SRC_DIR: Source folder path.
- MODELS_DIR: Model artifact folder path.
- PROCESSED_DIR: Processed output folder path.
- SYSTEM_LIST_PUBLIC_URL: Public metadata API endpoint.
- BASE_TEMP_C: Base temp used in heating degree-hour derivation.

## Environment Variables
The app reads these values from .env if present.

- GEMINI_API_KEY
  - Required for AI Briefing responses.
  - If absent, AI Briefing shows unavailable warning.

- GEMINI_MODEL
  - Optional override for Gemini model.
  - Default fallback: gemini-2.5-flash.

- CUSTOMER_STAFF_PASSWORD
  - If set, shows Staff Access expander.
  - Required to unlock staff tabs.

## Streamlit Session State Keys
- customer_appearance_mode
  - Theme mode for Auto, Light, Dark.

- customer_latest_scored
  - Latest scored dataframe from instant or batch prediction.
  - Used by AI Briefing context.

- customer_ai_brief_topic
  - Selected briefing topic.

- customer_ai_custom_request
  - Custom question text when Custom request topic is selected.

- customer_ai_brief_note
  - Optional user note appended to briefing request.

- customer_currency
  - UI currency selection.

- customer_unit_rate
  - Default electricity unit rate.

- customer_staff_access_code
  - Staff password input.

## Function Reference

### Configuration and Utility Functions

#### load_env_file(env_path: Path) -> None
Reads key-value pairs from .env into process environment if not already present.

#### inject_ui(appearance_mode: str = "Auto") -> None
Injects full custom CSS theme and component styling. Supports Light, Dark, and Auto mode via CSS variables.

#### read_json(path: Path) -> dict[str, Any]
Loads JSON file content from disk.

#### load_module(path: Path, module_name: str)
Dynamically imports a Python module from a file path.

#### _fetch_defaults_source_mtime() -> float
Returns modification timestamp of src/01_fetch_data.py for cache invalidation.

### Default System and Model Resolution

#### load_default_series_ids(source_mtime: float) -> list[int]
Cached loader for default system IDs from DEFAULT_SYSTEMS in src/01_fetch_data.py.

#### default_series_id() -> int
Returns first default ID or fallback 1.

#### get_predict_module()
Cached loader for src/05_predict_model.py module.

#### list_manifest_runs() -> list[dict[str, Any]]
Reads run manifests and returns metadata for each model run.

#### select_customer_run(manifests: list[dict[str, Any]]) -> dict[str, Any]
Chooses newest run manifest for customer mode.

#### resolve_feature_schema(manifest: dict[str, Any]) -> dict[str, Any]
Resolves serving schema from inline manifest field or schema file path.

### External Metadata API Utilities

#### fetch_public_system_list() -> list[dict[str, Any]]
Cached call to public system metadata endpoint.

#### find_system_metadata(rows: list[dict[str, Any]], system_id: int) -> dict[str, Any] | None
Finds one system entry by ID.

### Input Construction and Type Safety

#### infer_default(col: str, dtype_label: str) -> Any
Infers default value by feature name and target dtype.

#### coerce_for_dtype(value: Any, dtype_label: str) -> Any
Coerces value to required schema type and safe fallback.

#### build_single_row(required_cols, dtype_map, user_inputs) -> pd.DataFrame
Builds one complete serving row with:
- User-provided core values.
- Derived thermal features.
- Derived temporal encodings.
- Historical fallback defaults.
- Type-safe coercion for all required columns.

### Output Interpretation Helpers

#### cop_band(cop_value: float) -> tuple[str, str]
Maps COP to customer-friendly performance band and color.

#### customer_note(cop_value: float, on_probability: float) -> str
Returns plain-language recommendation by runtime likelihood and COP.

#### safe_mean(df: pd.DataFrame, column: str) -> float | None
Safe mean helper with missing/NaN handling.

#### summarize_prediction_df(df: pd.DataFrame) -> dict[str, Any]
Creates compact summary payload used by AI Briefing context.

### Gemini Integration

#### call_gemini(api_key: str, model_name: str, prompt: str) -> str
Sends prompt to Gemini API and returns text response.
Includes fallback behavior for 400 responses and robust error messaging.

### UI Rendering Functions

#### render_hero() -> None
Renders top customer hero section.

#### render_model_trust_strip(selected_run: dict[str, Any]) -> None
Legacy model diagnostics strip helper; currently not used in customer path.

#### render_single_prediction_result(scored: pd.DataFrame, unit_rate: float, currency_symbol: str) -> None
Renders customer-facing prediction cards and technical details expander.

#### render_instant_estimate_tab(run_tag, selected_manifest, unit_rate_default, currency_symbol) -> None
Main customer input form and prediction trigger.
Outputs:
- Result cards
- Technical details expander
- Updates customer_latest_scored in session

#### render_batch_tab(run_tag, selected_manifest) -> None
Staff batch scoring workflow:
- Template download
- CSV upload
- Batch scoring
- Metrics and chart preview
- Download scored CSV

#### render_system_info_tab() -> None
Staff utility to look up system metadata by ID.

#### render_settings_bar(manifests) -> tuple[str, dict[str, Any], float, str]
Renders quick start controls:
- Appearance mode
- Currency
- Unit rate
Also resolves active run_tag and selected run manifest metadata.

#### render_gemini_tab(selected_run: dict[str, Any]) -> None
Customer AI Briefing flow:
- Requires latest prediction output in session.
- Offers broad preset topics and Custom request option.
- Accepts optional extra note.
- Sends structured context plus selected request to Gemini.

### Application Entrypoint

#### main() -> None
App orchestration:
1. Load environment
2. Configure page and theme
3. Validate manifest availability
4. Render hero and settings
5. Render instant estimate section
6. Render AI Briefing section
7. Optionally render staff-gated tabs

## AI Briefing Behavior Detail

### Preset Options
- Explain my estimate
- How to reduce my bill
- How to improve efficiency
- What to monitor next

### Custom Request
If user selects Custom request, they can write any free-text question.
Validation ensures the question is not empty before API call.

### Context Sent to Gemini
- run_tag, policy, strategy
- model metrics block
- summarized latest prediction payload
- selected preset or custom request
- optional user note

## Staff Access Security Model
Staff tools are hidden unless CUSTOMER_STAFF_PASSWORD exists.
When set:
- Staff Access expander appears.
- Entered code must match environment value.
- If valid, batch and system-info tabs unlock.

## Error Handling Strategy
- Missing manifests: user-friendly top-level error.
- Missing schema: feature schema warning in relevant section.
- Prediction failures: form-level error with exception string.
- Gemini unavailable: clear warning if API key is missing.
- Gemini request errors: user-friendly fallback plus details.

## Data Contracts

### Single Prediction Input Contract
The required feature columns are runtime-resolved from manifest schema.
All required columns are filled before prediction using user input + safe defaults.

### Prediction Output Columns (expected from predict_bundle)
- runtime_on_proba
- runtime_on_pred
- pred_heatpump_elec
- pred_heatpump_heat
- pred_cop
- cop_guardrail_adjusted

The UI tolerates missing optional fields but expects core prediction columns for full display.

## Operational Notes

### Running the app
Use:
1. .venv\Scripts\python.exe -m streamlit run customer_app.py

### Required artifacts
- At least one models/run_manifest_*.json
- Corresponding model and schema artifacts referenced by the manifest

### Optional features
- AI Briefing requires GEMINI_API_KEY
- Staff tools require CUSTOMER_STAFF_PASSWORD

## Maintenance Checklist
When editing customer_app.py, verify:
1. App still loads with no manifest and with manifest.
2. Predict Now path works with default inputs.
3. AI Briefing works for preset topic.
4. AI Briefing works for Custom request.
5. Staff lock and unlock behavior remains correct.
6. Theme switching remains functional for Auto, Light, Dark.

## Suggested Future Improvements
1. Add token-safe response truncation for very long Gemini outputs.
2. Persist latest AI briefing in session for history display.
3. Add lightweight analytics event hooks for user flow drop-off.
4. Add unit tests for feature row construction and type coercion.
5. Add graceful retry with backoff for Gemini network errors.
