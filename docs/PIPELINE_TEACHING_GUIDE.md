# Heat Pump ML Pipeline - Complete Teaching Guide

**For Instructors & Students**  
**Version:** 1.0  
**Last Updated:** April 18, 2026

---

## Table of Contents

0. [April 2026 Addendum](#0-april-2026-addendum)
1. [Introduction & Learning Objectives](#1-introduction--learning-objectives)
2. [System Overview](#2-system-overview)
3. [Stage 1: Data Fetching](#3-stage-1-data-fetching)
4. [Stage 2: Feature Engineering](#4-stage-2-feature-engineering)
5. [Stage 3: Data Cleaning](#5-stage-3-data-cleaning)
6. [Stage 4: Model Training](#6-stage-4-model-training)
7. [Stage 5: Inference/Prediction](#7-stage-5-inferenceprediction)
8. [Stage 6: Walk-Forward Backtesting](#8-stage-6-walk-forward-backtesting)
9. [Stage 7: Streamlit Dashboard](#9-stage-7-streamlit-dashboard)
10. [Key Concepts Summary](#10-key-concepts-summary)
11. [Exercises & Assignments](#11-exercises--assignments)
12. [Glossary](#12-glossary)

---

## 0. April 2026 Addendum

This guide remains focused on pipeline teaching and stage-by-stage internals.

Current workspace operation now also includes:

- `streamlit_app.py` admin tabs with run-context selection in Model Health and Gemini, plus System Metadata lookup.
- `customer_app.py` customer journey with simplified inputs, AI Briefing presets + custom request, and staff-gated advanced tools.

For current implementation behavior and operations details, use:

- `PIPELINE_GUIDE.md`
- `docs/CUSTOMER_APP_DETAILED_DOCUMENTATION.md`
- `docs/SOURCE_CODE_GUIDE.md`

---

## 1. Introduction & Learning Objectives

### What This Project Teaches

This Heat Pump Monitor ML pipeline is a **production-grade machine learning system** that demonstrates:

- **Time-series forecasting** for energy systems
- **Physics-informed feature engineering**
- **Multi-target regression** (predicting electricity AND heat simultaneously)
- **Walk-forward validation** (proper temporal cross-validation)
- **Production ML patterns** (imputation, guardrails, calibration)

### Learning Objectives

After studying this pipeline, students should be able to:

1. ✅ Design an end-to-end ML pipeline for time-series data
2. ✅ Create physics-informed features that encode domain knowledge
3. ✅ Implement proper temporal train/test splitting (avoiding data leakage)
4. ✅ Apply post-prediction guardrails to ensure physical validity
5. ✅ Build slice-aware calibration for multi-system deployments
6. ✅ Evaluate models using business metrics (not just R²)

### Prerequisites

- Python fundamentals (pandas, numpy)
- Basic ML concepts (regression, train/test splits)
- Understanding of what a heat pump does (converts electricity → heat)

---

## 2. System Overview

### The Business Problem

**Heat pumps** are devices that move heat from outside air into buildings. They're 3-4x more efficient than electric heaters. Utilities and homeowners need to:

1. **Forecast energy consumption** for billing and grid planning
2. **Estimate heat output** for comfort optimization
3. **Calculate COP (Coefficient of Performance)** = Heat ÷ Electricity

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        HEAT PUMP ML PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                   │
│  │  01_fetch   │ → │ 02_features │ → │  03_clean   │                   │
│  │   (API)     │   │ (43 feats)  │   │ (physics)   │                   │
│  └─────────────┘   └─────────────┘   └─────────────┘                   │
│                                             │                           │
│                    ┌────────────────────────┘                           │
│                    ▼                                                    │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                   │
│  │ 05_predict  │ ← │  04_train   │ → │ 07_backtest │                   │
│  │ (inference) │   │ (XGBoost×3) │   │ (5-fold WF) │                   │
│  └─────────────┘   └─────────────┘   └─────────────┘                   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    streamlit_app.py (UI)                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

| Stage | Input | Output | Key Concept |
|-------|-------|--------|-------------|
| **01_fetch** | API URL + dates | `raw/*.parquet` | Async I/O, rate limiting |
| **02_features** | Raw data | `processed/features.parquet` | Domain features |
| **03_clean** | Features | `processed/cleaned.parquet` | Physics constraints |
| **04_train** | Cleaned data | `models/*.joblib` | XGBoost, walk-forward |
| **05_predict** | New data + models | Predictions | Serving pipeline |
| **07_backtest** | Cleaned data | Metrics CSV/JSON | Temporal validation |

---

## 3. Stage 1: Data Fetching

### 📁 File: `src/01_fetch_data.py`

### Concept: Async API Data Collection

This stage downloads hourly telemetry from 7 heat pump systems via HTTP API.

### System Configuration

```python
# Each heat pump system has unique ID and capacity
DEFAULT_SYSTEMS = [
    {"series_id": 615, "capacity_kw": 8},  # 8 kW system
    {"series_id": 364, "capacity_kw": 8},
    {"series_id": 44,  "capacity_kw": 8},
    {"series_id": 162, "capacity_kw": 6},  # 6 kW system
    {"series_id": 228, "capacity_kw": 4},  # 4 kW system (smallest)
    {"series_id": 351, "capacity_kw": 6},
    {"series_id": 587, "capacity_kw": 6},
]
```

**💡 Teaching Point:** `capacity_kw` is crucial metadata. A 4kW system operating at 2000W is at 50% load; an 8kW system at 2000W is only at 25% load.

### Data Channels (7 Feeds)

```python
FEEDS = [
    "heatpump_elec",      # Electrical power consumption (Watts)
    "heatpump_heat",      # Heat output (Watts)
    "heatpump_returnT",   # Return water temperature (°C)
    "heatpump_flowT",     # Flow water temperature (°C)
    "heatpump_roomT",     # Indoor room temperature (°C)
    "heatpump_outsideT",  # Outdoor ambient temperature (°C)
    "heatpump_flowrate",  # Water flow rate (L/min)
]
```

### Async Fetching Pattern

```python
async def fetch_system(client: httpx.AsyncClient, system: dict):
    """
    Fetch data for one system asynchronously.
    
    Why async?
    - 7 systems × 3 years × 8760 hours/year = lots of data
    - Sequential: 7 × 60s timeout = 7 minutes worst case
    - Async parallel: ~60s total (all systems at once)
    """
    params = {
        "id": system["series_id"],
        "start": start_date,
        "end": end_date,
        "interval": 3600,  # Hourly (3600 seconds)
        "average": 1,      # Average within interval
    }
    
    # Async HTTP GET (non-blocking)
    resp = await client.get(API_URL, params=params, timeout=60.0)
    resp.raise_for_status()
    
    return system["series_id"], resp.json()
```

**💡 Teaching Point:** `await` allows the program to fetch from all 7 systems simultaneously instead of waiting for each one to finish.

### Fetch Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| `fixed_window` | Uses exact START_DATE to END_DATE | Production (known date range) |
| `max_available` | Fetches maximum data API allows | Training (want all history) |

### Calendar Alignment

```python
# Create hourly timestamp index in UTC
time_index = pd.date_range(
    start=start_dt,
    end=end_dt,
    freq="1h",        # Hourly intervals
    inclusive="left", # [start, end) - excludes end
    tz="UTC",         # Timezone-aware (prevents DST issues)
)
```

**⚠️ Common Mistake:** Not using UTC causes 23-hour and 25-hour days during daylight savings transitions.

### Output Artifacts

```
data/raw/
├── raw_system_615.parquet
├── raw_system_364.parquet
├── ...
├── fetch_log.csv          # Success/failure per system
└── fetch_quality_audit.csv # Missing data % per feed
```

---

## 4. Stage 2: Feature Engineering

### 📁 File: `src/02_feature_engineering.py`

### Concept: Transforming Raw Data into ML-Ready Features

This is where **domain knowledge** becomes **predictive signal**.

### Feature Categories Overview

| Category | Count | Examples |
|----------|-------|----------|
| Core Energy | 4 | `cop`, `heating_on`, `elec_kwh`, `heat_kwh` |
| Physics | 6 | `deltaT_house`, `deltaT_lift`, `hdh`, `temp_deficit` |
| Temporal | 7 | `hour`, `hour_sin`, `hour_cos`, `month_sin`, `day_of_week` |
| Lag (Memory) | 19 | `elec_lag1`, `elec_lag24`, `cop_lag1`, `heat_lag168` |
| Rolling | 1 | `outsideT_3h_avg` |
| State | 3 | `run_hours`, `was_defrost_lag1`, `elec_lag1_pct` |

### A. Core Energy Metrics

```python
def add_energy_metrics(df):
    """Calculate fundamental energy quantities."""
    
    # Convert W to kWh for reporting
    df["elec_kwh"] = df["heatpump_elec"] / 1000.0
    df["heat_kwh"] = df["heatpump_heat"] / 1000.0
    
    # COP = Coefficient of Performance = Heat Out / Electricity In
    # A COP of 4 means: for every 1 kW electricity, you get 4 kW heat
    df["cop"] = df["heatpump_heat"] / df["heatpump_elec"]
    
    # Binary: Is the system actively heating?
    # True if: heat > 0 AND electricity > standby threshold (50W)
    STANDBY_THRESHOLD_W = 50
    df["heating_on"] = (
        (df["heatpump_heat"] > 0) & 
        (df["heatpump_elec"] > STANDBY_THRESHOLD_W)
    ).astype(int)
    
    return df
```

**💡 Teaching Point:** COP is the key efficiency metric. Electric heaters have COP=1 (1 kW electricity → 1 kW heat). Heat pumps achieve COP=3-5 by extracting heat from outdoor air.

### B. Physics-Based Features

```python
def add_physics_features(df):
    """
    Create features based on thermodynamic principles.
    These encode domain knowledge that helps the model.
    """
    
    # 1. House-side temperature difference
    # Higher deltaT = more heat transferred to building
    df["deltaT_house"] = df["heatpump_flowT"] - df["heatpump_returnT"]
    
    # 2. Thermodynamic "lift"
    # How hard the heat pump works = flow temp - outside temp
    # Lifting heat from -5°C to 45°C is harder than 10°C to 35°C
    df["deltaT_lift"] = df["heatpump_flowT"] - df["heatpump_outsideT"]
    
    # 3. Temperature deficit (building demand proxy)
    # If room is 20°C and outside is 0°C, deficit = 20°C
    df["temp_deficit"] = df["heatpump_roomT"] - df["heatpump_outsideT"]
    
    # 4. Heating Degree Hours (HDH)
    # Industry-standard measure of heating demand
    # BASE_TEMP = 15.5°C (below this, heating needed)
    BASE_TEMP = 15.5
    df["hdh"] = (BASE_TEMP - df["heatpump_outsideT"]).clip(lower=0)
    
    # 5. Load ratio (normalized demand)
    # Accounts for different system sizes
    df["load_ratio"] = df["temp_deficit"] / df["capacity_kw"]
    
    # 6. Heating season indicator
    df["is_heating_season"] = (df["heatpump_outsideT"] < BASE_TEMP).astype(int)
    
    return df
```

**💡 Teaching Point:** `hdh` (Heating Degree Hours) is a meteorological standard. It accumulates "how cold" hours are. More HDH = more heating needed.

### C. Cyclical Time Encoding

```python
def add_temporal_features(df):
    """
    Encode time features that respect cyclical nature.
    
    Problem: Hour 23 → Hour 0 looks like a huge jump (23 units)
    Solution: Sine/cosine encoding makes it continuous
    """
    
    hour = df["timestamp"].dt.hour    # 0-23
    month = df["timestamp"].dt.month  # 1-12
    
    # Raw values (for interpretability)
    df["hour"] = hour
    df["month"] = month
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Monday
    
    # Harmonic encoding (for ML models)
    # hour_sin and hour_cos together form a circle
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    
    return df
```

**💡 Teaching Point:** With sine/cosine encoding:
- Hour 0: (sin=0, cos=1)
- Hour 6: (sin=1, cos=0)
- Hour 12: (sin=0, cos=-1)
- Hour 18: (sin=-1, cos=0)
- Hour 23: (sin≈0, cos≈1) — smooth transition to Hour 0!

### D. Lag Features (Memory)

```python
def add_lag_features(df):
    """
    Add historical values as features.
    
    Key insight: Energy consumption has patterns:
    - lag1 (1 hour ago): Immediate state persistence
    - lag24 (yesterday same hour): Daily pattern
    - lag168 (last week same hour): Weekly pattern
    """
    
    # Group by system to prevent cross-contamination
    g = df.groupby("series_id")
    
    # Immediate memory (what happened 1-6 hours ago)
    df["elec_lag1"] = g["heatpump_elec"].shift(1)
    df["elec_lag2"] = g["heatpump_elec"].shift(2)
    df["elec_lag3"] = g["heatpump_elec"].shift(3)
    df["elec_lag4"] = g["heatpump_elec"].shift(4)
    df["elec_lag6"] = g["heatpump_elec"].shift(6)
    
    # Daily pattern (same hour yesterday)
    df["elec_lag24"] = g["heatpump_elec"].shift(24)
    df["heat_lag24"] = g["heatpump_heat"].shift(24)
    
    # Weekly pattern (same hour last week)
    df["elec_lag168"] = g["heatpump_elec"].shift(168)  # 24 * 7 = 168
    df["heat_lag168"] = g["heatpump_heat"].shift(168)
    
    # Thermal state memory
    df["flowT_lag1"] = g["heatpump_flowT"].shift(1)
    df["returnT_lag1"] = g["heatpump_returnT"].shift(1)
    df["cop_lag1"] = g["cop"].shift(1)
    df["heating_on_lag1"] = g["heating_on"].shift(1)
    
    # Derived lag features
    df["deltaT_house_lag1"] = df["flowT_lag1"] - df["returnT_lag1"]
    df["deltaT_lift_lag1"] = df["flowT_lag1"] - df["heatpump_outsideT"]
    
    return df
```

**⚠️ Critical:** Always use `groupby("series_id")` before `.shift()`. Otherwise, system 615's lag would leak into system 364's features!

### E. Special State Features

```python
def add_state_features(df):
    """Track operational state indicators."""
    
    g = df.groupby("series_id")
    
    # 1. Run-hours counter
    # How many consecutive hours has heating been ON?
    # Resets to 0 when system turns OFF
    df["run_hours"] = (
        g["heating_on"]
        .transform(lambda x: x.groupby((x != x.shift()).cumsum()).cumcount())
    )
    
    # 2. Defrost detection
    # When COP < 1, system is likely in defrost cycle
    # (using electricity to melt ice, not heating house)
    df["was_defrost_lag1"] = (df["cop_lag1"] < 1.0).astype(float)
    
    # 3. Capacity utilization
    # What % of max capacity was used last hour?
    df["elec_lag1_pct"] = df["elec_lag1"] / (df["capacity_kw"] * 1000)
    
    return df
```

**💡 Teaching Point:** `was_defrost_lag1` is crucial! During defrost, COP temporarily drops below 1.0 (heat pump uses electricity but produces no useful heat). The model needs to know this state.

### Feature Engineering Summary

```
Total features created: 43+
├── Core Energy: 4
├── Physics-based: 6
├── Temporal: 7
├── Lag features: 19
├── Rolling: 1
└── State: 3+

Most important features (by model importance):
1. elec_lag1      (0.146) - What was consumption 1 hour ago?
2. elec_lag24     (0.106) - What was consumption same hour yesterday?
3. run_hours      (0.085) - How long has heating been running?
4. hdh            (0.084) - How cold is it outside?
5. heating_on_lag1 (0.162 for heat) - Was heating ON last hour?
```

---

## 5. Stage 3: Data Cleaning

### 📁 File: `src/03_clean_data.py`

### Concept: Physics-Constrained Data Quality

Raw data contains sensor errors, impossible values, and missing readings. This stage enforces **physical reality**.

### Cleaning Pipeline

```
Raw Features (115K rows)
         │
         ▼
    ┌─────────────────┐
    │ 1. Interpolation │ ← Fill 1-hour gaps in temperature
    └─────────────────┘
         │
         ▼
    ┌─────────────────┐
    │ 2. Physics Rules │ ← Remove impossible values
    └─────────────────┘
         │
         ▼
    ┌─────────────────┐
    │ 3. Drop NA      │ ← Remove rows missing targets
    └─────────────────┘
         │
         ▼
Cleaned Features (90K rows)
```

### Step 1: Smart Interpolation

```python
def smart_interpolate(df):
    """
    Fill small gaps in sensor data.
    
    Policy: Only interpolate 1-hour gaps in auxiliary sensors.
    Never interpolate targets (electricity, heat).
    """
    
    INTERP_COLS = [
        "heatpump_outsideT",
        "heatpump_roomT",
        "heatpump_flowT",
        "heatpump_returnT",
    ]
    
    # Per-system interpolation (don't mix systems!)
    df[INTERP_COLS] = (
        df.groupby("series_id")[INTERP_COLS]
        .transform(lambda x: x.interpolate(method="linear", limit=1))
    )
    
    return df
```

**💡 Teaching Point:** `limit=1` means: only fill gaps of 1 missing value. Longer gaps likely indicate real sensor outages.

### Step 2: Physics Constraint Rules

```python
# These are the physical boundaries for valid data
PHYSICS_RULES = {
    "heatpump_elec_min": 0.0,      # No negative power
    "heatpump_elec_max": None,     # No upper limit (capacity varies)
    "heatpump_heat_min": 0.0,      # No negative heat output
    "heatpump_heat_max": None,
    "cop_min": 0.0,                # COP can't be negative
    "cop_max": 8.0,                # COP > 8 is unrealistic
    "deltaT_house_min": 0.0,       # Flow must be warmer than return
    "deltaT_house_max": 20.0,      # > 20°C suggests sensor error
    "outsideT_min": -20.0,         # Realistic UK/EU winter
    "outsideT_max": 40.0,          # Realistic summer max
}

def apply_physics_rules(df):
    """Remove rows that violate physical constraints."""
    
    # Build boolean mask: True = row is valid
    valid_mask = (
        (df["heatpump_elec"] >= 0) &
        (df["heatpump_heat"] >= 0) &
        (df["cop"].isna() | ((df["cop"] >= 0) & (df["cop"] <= 8))) &
        (df["deltaT_house"] >= 0) &
        (df["deltaT_house"] <= 20) &
        (df["heatpump_outsideT"] >= -20) &
        (df["heatpump_outsideT"] <= 40)
    )
    
    invalid_count = (~valid_mask).sum()
    print(f"Removing {invalid_count} rows with physics violations")
    
    return df[valid_mask]
```

**💡 Teaching Point:** `df["cop"].isna() | ...` allows missing COP values. Why? When electricity is near zero, COP = heat/0 = undefined. That's okay - we keep those rows.

### Step 3: Target-Based Filtering

```python
def filter_missing_targets(df):
    """
    Remove rows where target values are missing.
    
    Philosophy:
    - Targets (elec, heat) MUST be present (can't train without labels)
    - Features CAN be missing (imputer fills them later)
    """
    
    REQUIRED_COLUMNS = ["heatpump_elec", "heatpump_heat"]
    
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    after = len(df)
    
    print(f"Dropped {before - after} rows missing targets")
    return df
```

### Cleaning Results Example

```
CLEANING REPORT
===============
System 162: 24,719 → 22,920 rows (92.7% kept)
System 44:  25,654 → 16,862 rows (65.7% kept)
System 228: 17,347 → 12,295 rows (70.9% kept)
System 351: 13,220 → 11,697 rows (88.5% kept)
System 364: 13,611 →  8,622 rows (63.3% kept)
System 587: 10,246 →  9,462 rows (92.3% kept)
System 615: 10,532 →  8,525 rows (80.9% kept)
───────────────────────────────────────────
TOTAL:     115,329 → 90,383 rows (78.4% kept)
```

---

## 6. Stage 4: Model Training

### 📁 File: `src/04_train_model.py`

### Concept: Multi-Target XGBoost with Walk-Forward Validation

This is the core ML stage. We train **three models**:
1. **Electricity regressor** (XGBoost)
2. **Heat regressor** (XGBoost)
3. **Runtime classifier** (XGBoost) - predicts ON/OFF state

### Training Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. TEMPORAL SPLIT                                              │
│     ┌──────────────────────────────────────┬───────────────┐   │
│     │         Training (80%)               │   Test (20%)  │   │
│     │    Jan 2023 ──────── Sep 2025       │ Sep 2025─2026 │   │
│     └──────────────────────────────────────┴───────────────┘   │
│                                                                 │
│  2. FOR EACH TARGET STRATEGY:                                   │
│     ├── "none" (raw values)                                     │
│     ├── "log1p_both" (log transform)   ← Usually wins          │
│     └── "cop_ratio_heat" (heat as ratio)                        │
│                                                                 │
│  3. WALK-FORWARD HYPERPARAMETER TUNING (3 folds)                │
│     Train on past → Validate on future → Select best params    │
│                                                                 │
│  4. TRAIN FINAL MODELS                                          │
│     ├── XGBoost Electricity (32 features)                       │
│     ├── XGBoost Heat (35 features)                              │
│     └── XGBoost Runtime Classifier (binary)                     │
│                                                                 │
│  5. BUILD SLICE CALIBRATORS                                     │
│     Per-system multiplicative corrections                       │
│                                                                 │
│  6. EVALUATE & PERSIST                                          │
│     Apply quality gates → Save artifacts                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Temporal Train/Test Split

```python
def temporal_train_test_split(df, test_ratio=0.20):
    """
    Split data by TIME, not randomly.
    
    Why? Time-series data has temporal dependencies.
    Random split would let future leak into training!
    """
    
    unique_timestamps = df["timestamp"].sort_values().unique()
    n = len(unique_timestamps)
    
    # 80% oldest for training, 20% newest for testing
    split_idx = int(n * (1 - test_ratio))
    split_timestamp = unique_timestamps[split_idx]
    
    train_df = df[df["timestamp"] < split_timestamp]
    test_df = df[df["timestamp"] >= split_timestamp]
    
    print(f"Train: {len(train_df)} rows ending at {split_timestamp}")
    print(f"Test:  {len(test_df)} rows starting at {split_timestamp}")
    
    return train_df, test_df
```

**⚠️ Critical:** NEVER use random train/test split for time-series! It causes **data leakage** (model sees future during training).

### Target Transformation Strategies

```python
def apply_target_strategy(y_train, y_test, strategy):
    """
    Transform targets to improve model performance.
    """
    
    if strategy == "none":
        # Raw values - no transformation
        return y_train, y_test, lambda x: x
    
    elif strategy == "log1p_both":
        # Log transformation: log(1 + y)
        # Why? Stabilizes variance, reduces outlier impact
        y_train_t = np.log1p(np.maximum(y_train, 0))
        y_test_t = np.log1p(np.maximum(y_test, 0))
        inverse_fn = lambda x: np.expm1(x)  # Inverse: exp(x) - 1
        return y_train_t, y_test_t, inverse_fn
    
    elif strategy == "cop_ratio_heat":
        # Predict heat as ratio: heat / electricity
        # Leverages thermodynamic coupling
        ratio_train = y_heat_train / np.maximum(y_elec_train, 1e-6)
        ratio_test = y_heat_test / np.maximum(y_elec_test, 1e-6)
        return ratio_train, ratio_test, None  # Special handling needed
```

**💡 Teaching Point:** `log1p(x)` = `log(1 + x)`. The "+1" handles zero values (log(0) is undefined). `expm1(x)` = `exp(x) - 1` is the inverse.

### XGBoost Hyperparameters

```python
ELEC_HYPERPARAMS = {
    "n_estimators": 2200,       # Number of trees
    "learning_rate": 0.02,      # Step size shrinkage (smaller = slower but better)
    "max_depth": 7,             # Maximum tree depth
    "subsample": 0.8,           # Row sampling ratio (80% per tree)
    "colsample_bytree": 0.75,   # Feature sampling ratio (75% per tree)
    "min_child_weight": 8,      # Minimum leaf node weight
    "reg_alpha": 0.2,           # L1 regularization (sparsity)
    "reg_lambda": 2.0,          # L2 regularization (smoothness)
    "gamma": 0.1,               # Minimum gain for split
    "early_stopping_rounds": 120,
    "eval_metric": "mae",       # Optimize for Mean Absolute Error
}

HEAT_HYPERPARAMS = {
    "n_estimators": 2600,       # More trees (heat is noisier)
    "learning_rate": 0.015,     # Slower learning
    "max_depth": 8,             # Slightly deeper
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "min_child_weight": 6,
    "reg_alpha": 0.1,
    "reg_lambda": 1.8,
    "gamma": 0.05,
    "early_stopping_rounds": 150,
    "eval_metric": "mae",
}
```

**💡 Teaching Point:** Why different hyperparameters?
- **Electricity** is more predictable (cleaner signal) → fewer trees, stronger regularization
- **Heat** depends on many factors (occupant behavior, setpoints) → more trees, weaker regularization

### Walk-Forward Hyperparameter Tuning

```python
def walk_forward_tune(X_train, y_train, candidates):
    """
    Tune hyperparameters using forward-chaining cross-validation.
    
    Unlike K-Fold, walk-forward respects temporal order:
    - Fold 1: Train [Jan-Apr], Validate [May-Jun]
    - Fold 2: Train [Jan-Jun], Validate [Jul-Aug]
    - Fold 3: Train [Jan-Aug], Validate [Sep-Oct]
    """
    
    from sklearn.model_selection import TimeSeriesSplit
    
    splitter = TimeSeriesSplit(n_splits=3)
    
    best_params = None
    best_avg_mae = float("inf")
    
    for params in candidates:
        fold_maes = []
        
        for train_idx, val_idx in splitter.split(X_train):
            model = XGBRegressor(**params)
            model.fit(
                X_train[train_idx], 
                y_train[train_idx],
                eval_set=[(X_train[val_idx], y_train[val_idx])],
                verbose=False
            )
            
            pred = model.predict(X_train[val_idx])
            mae = mean_absolute_error(y_train[val_idx], pred)
            fold_maes.append(mae)
        
        avg_mae = np.mean(fold_maes)
        if avg_mae < best_avg_mae:
            best_avg_mae = avg_mae
            best_params = params
    
    return best_params
```

### Runtime ON/OFF Classifier

```python
def train_runtime_classifier(X_train, X_test, y_train_on, y_test_on):
    """
    Binary classifier: Is heating ON or OFF?
    
    Purpose: Gate mechanism for predictions.
    If predicted OFF → force elec=50W (standby), heat=0W
    """
    
    clf = XGBClassifier(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=6,
        eval_metric="logloss",
    )
    
    clf.fit(
        X_train, 
        y_train_on,  # 1 = ON, 0 = OFF
        eval_set=[(X_test, y_test_on)]
    )
    
    # Predict probability of ON state
    on_proba = clf.predict_proba(X_test)[:, 1]
    on_pred = (on_proba >= 0.5).astype(int)
    
    return clf, on_pred, on_proba
```

**💡 Teaching Point:** Runtime classifier achieved **F1 = 1.0** (perfect). This is critical for safety - we never want to predict heating when system is OFF, or miss heating when ON.

### Slice Calibration

```python
def build_slice_calibrators(df, pred_elec, pred_heat):
    """
    Learn per-system bias corrections.
    
    Problem: Model might systematically under-predict system 615 by 5%
    Solution: Learn multiplier (1.05) and apply at inference time
    """
    
    calibrators = {}
    
    for series_id, grp in df.groupby("series_id"):
        pred_e_sum = grp["pred_elec"].sum()
        true_e_sum = grp["heatpump_elec"].sum()
        
        pred_h_sum = grp["pred_heat"].sum()
        true_h_sum = grp["heatpump_heat"].sum()
        
        # Multiplicative correction (clipped to prevent wild values)
        elec_mult = np.clip(true_e_sum / pred_e_sum, 0.7, 1.3)
        heat_mult = np.clip(true_h_sum / pred_h_sum, 0.7, 1.3)
        
        calibrators[series_id] = {
            "elec_mult": elec_mult,
            "heat_mult": heat_mult,
        }
    
    return calibrators
```

### COP Guardrail (Physics Post-Processing)

```python
def apply_cop_guardrail(pred_elec_w, pred_heat_w, cop_min=0.0, cop_max=8.0):
    """
    Ensure predictions obey physics: 0 ≤ COP ≤ 8
    
    COP = heat / electricity
    
    If predicted COP > 8, clip heat down to max allowed.
    If predicted COP < 0, clip heat up to 0.
    """
    
    # Ensure non-negative
    elec = np.maximum(pred_elec_w, 0)
    heat = np.maximum(pred_heat_w, 0)
    
    # Calculate COP bounds
    valid_elec = elec > 1e-6
    min_heat = np.zeros_like(heat)
    max_heat = np.zeros_like(heat)
    
    max_heat[valid_elec] = elec[valid_elec] * cop_max  # heat ≤ elec × 8
    min_heat[valid_elec] = elec[valid_elec] * cop_min  # heat ≥ elec × 0
    
    # Clip heat to physical bounds
    heat_clipped = np.clip(heat, min_heat, max_heat)
    
    adjusted_count = np.count_nonzero(np.abs(heat_clipped - heat) > 1e-6)
    
    return elec, heat_clipped, adjusted_count
```

**Example:**
- Predicted: elec=1000W, heat=10000W → COP=10 (impossible!)
- After guardrail: elec=1000W, heat=8000W → COP=8 (maximum realistic)

### Quality Gates

```python
SCIENCE_GATES = {
    "r2_elec_pass": r2_elec > 0.80,     # 80% variance explained
    "r2_heat_pass": r2_heat > 0.80,
    "mae_elec_pass": mae_elec < 90.0,   # Within 90W
    "mae_heat_pass": mae_heat < 120.0,  # Within 120W
}

BUSINESS_GATES = {
    "energy_err_pass": energy_err_pct < 5.0,   # ±5% total energy
    "cop_err_pass": cop_err_pct < 6.0,         # ±6% COP
    "runtime_fn_pass": false_negatives == 0,   # Never miss ON state
    "leakage_pass": r2 < 0.97,                 # R² > 0.97 suggests leakage
}

production_ready = all(BUSINESS_GATES.values())
research_ready = all(SCIENCE_GATES.values()) and production_ready
```

---

## 7. Stage 5: Inference/Prediction

### 📁 File: `src/05_predict_model.py`

### Concept: Serving Pipeline with Guardrails

This stage applies trained models to new data with proper feature alignment and physics enforcement.

### Inference Pipeline Flow

```
Input DataFrame
      │
      ▼
┌─────────────────────┐
│ 1. Load Artifacts   │ ← Models, imputers, schema, calibrators
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ 2. Validate Schema  │ ← Check all required columns present
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ 3. Impute Missing   │ ← Median imputation (from training)
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ 4. Predict Runtime  │ ← ON or OFF?
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ 5. Predict Targets  │ ← Electricity & Heat
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ 6. Apply Calibrators│ ← Per-system bias correction
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ 7. Apply Runtime    │ ← OFF → elec=50W, heat=0W
│    Gating           │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ 8. Apply COP        │ ← Ensure 0 ≤ COP ≤ 8
│    Guardrail        │
└─────────────────────┘
      │
      ▼
Output DataFrame + Predictions
```

### Full Prediction Code

```python
def predict_bundle(df: pd.DataFrame, run_tag: str) -> pd.DataFrame:
    """
    Score a batch of rows with the trained model bundle.
    """
    
    # 1. Load all artifacts
    bundle = load_model_bundle(run_tag)
    
    # 2. Validate input schema
    validate_columns(df, bundle["feature_schema"])
    
    # 3. Extract feature matrices
    X_elec = df[bundle["elec_features"]].values
    X_heat = df[bundle["heat_features"]].values
    
    # 4. Impute missing values (median from training)
    X_elec_imp = bundle["elec_imputer"].transform(X_elec)
    X_heat_imp = bundle["heat_imputer"].transform(X_heat)
    
    # 5. Predict runtime ON/OFF
    on_proba = bundle["runtime_clf"].predict_proba(X_elec_imp)[:, 1]
    on_pred = (on_proba >= 0.5).astype(int)
    
    # 6. Predict targets (with inverse transform if needed)
    if bundle["target_strategy"] == "log1p_both":
        pred_elec_w = np.expm1(bundle["elec_model"].predict(X_elec_imp))
        pred_heat_w = np.expm1(bundle["heat_model"].predict(X_heat_imp))
    else:
        pred_elec_w = bundle["elec_model"].predict(X_elec_imp)
        pred_heat_w = bundle["heat_model"].predict(X_heat_imp)
    
    # 7. Apply slice calibrators
    for i, row in df.iterrows():
        sid = row["series_id"]
        if sid in bundle["calibrators"]:
            pred_elec_w[i] *= bundle["calibrators"][sid]["elec_mult"]
            pred_heat_w[i] *= bundle["calibrators"][sid]["heat_mult"]
    
    # 8. Apply runtime gating
    off_mask = (on_pred == 0)
    pred_elec_w[off_mask] = 50.0   # Standby power
    pred_heat_w[off_mask] = 0.0    # No heat when OFF
    
    # 9. Apply COP guardrail
    pred_elec_w, pred_heat_w, _ = apply_cop_guardrail(
        pred_elec_w, pred_heat_w, cop_min=0.0, cop_max=8.0
    )
    
    # 10. Calculate COP
    pred_cop = np.divide(
        pred_heat_w, 
        np.maximum(pred_elec_w, 1e-6),
        out=np.full_like(pred_heat_w, np.nan),
        where=pred_elec_w > 1e-6
    )
    
    # 11. Attach predictions
    df["pred_heatpump_elec"] = pred_elec_w
    df["pred_heatpump_heat"] = pred_heat_w
    df["pred_cop"] = pred_cop
    
    return df
```

---

## 8. Stage 6: Walk-Forward Backtesting

### 📁 File: `src/backtest_model_07.py`

### Concept: Simulating Real Deployment Over Time

Walk-forward validation answers: "How would this model have performed if deployed at different points in time?"

### Walk-Forward Fold Construction

```
Data Timeline: ────────────────────────────────────────────►
               Jan 2023                                Mar 2026

Fold 1: Train [====50%====]  Test [10%]
        Jan─Jun 2024        Jul─Aug 2024

Fold 2: Train [======60%======]  Test [10%]
        Jan─Aug 2024            Sep─Oct 2024

Fold 3: Train [========70%========]  Test [10%]
        Jan─Oct 2024                Nov─Dec 2024

Fold 4: Train [==========80%==========]  Test [10%]
        Jan─Dec 2024                      Jan─Feb 2025

Fold 5: Train [============90%============]  Test [10%]
        Jan 2023─Feb 2025                    Mar─May 2025
```

**Key Properties:**
- Training window **grows** each fold (more historical data)
- Test window is always **future** relative to training
- No overlap between train and test (no leakage)

### Fold Generation Code

```python
def make_walk_forward_folds(df, min_train_ratio=0.50, test_ratio=0.10, step_ratio=0.10):
    """
    Generate walk-forward validation folds.
    
    Args:
        df: Full dataset with timestamp column
        min_train_ratio: Initial training window size (50% = 6 months for 1 year data)
        test_ratio: Test window size (10% = ~6 weeks)
        step_ratio: How much to advance between folds (10%)
    
    Returns:
        List of (fold_id, train_df, test_df) tuples
    """
    
    unique_ts = df["timestamp"].sort_values().unique()
    n = len(unique_ts)
    
    train_size = int(n * min_train_ratio)
    test_size = int(n * test_ratio)
    step_size = int(n * step_ratio)
    
    folds = []
    train_end_idx = train_size
    
    while train_end_idx + test_size <= n:
        train_end_ts = unique_ts[train_end_idx]
        test_end_ts = unique_ts[min(train_end_idx + test_size, n) - 1]
        
        train_df = df[df["timestamp"] < train_end_ts].copy()
        test_df = df[
            (df["timestamp"] >= train_end_ts) & 
            (df["timestamp"] <= test_end_ts)
        ].copy()
        
        if len(train_df) > 0 and len(test_df) > 0:
            folds.append((len(folds) + 1, train_df, test_df))
        
        train_end_idx += step_size
    
    return folds
```

### Per-Fold Training

```python
for fold_id, train_df, test_df in folds:
    print(f"\n=== FOLD {fold_id} ===")
    print(f"Train: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"Test:  {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    
    # 1. Prepare features
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_elec_train = train_df["heatpump_elec"].values
    y_heat_train = train_df["heatpump_heat"].values
    
    # 2. Fit imputer on training data ONLY
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)  # Use training medians!
    
    # 3. Train models (ON-state rows only for regression)
    on_mask_train = train_df["heating_on"] == 1
    
    elec_model = XGBRegressor(**ELEC_PARAMS)
    elec_model.fit(X_train_imp[on_mask_train], y_elec_train[on_mask_train])
    
    heat_model = XGBRegressor(**HEAT_PARAMS)
    heat_model.fit(X_train_imp[on_mask_train], y_heat_train[on_mask_train])
    
    # 4. Evaluate on test set
    pred_elec = elec_model.predict(X_test_imp)
    pred_heat = heat_model.predict(X_test_imp)
    
    # 5. Apply guardrails and compute metrics
    metrics = evaluate_fold(test_df, pred_elec, pred_heat)
    fold_results.append(metrics)
```

### Slice-Level Analysis

```python
def compute_slice_metrics(test_df, pred_elec, pred_heat):
    """
    Compute metrics per system and per capacity.
    
    Reveals which systems/configurations perform poorly.
    """
    
    results = []
    
    # Per-system metrics
    for series_id, grp in test_df.groupby("series_id"):
        idx = grp.index
        results.append({
            "slice_type": "series_id",
            "slice_value": series_id,
            "rows": len(grp),
            "r2_elec": r2_score(grp["heatpump_elec"], pred_elec[idx]),
            "r2_heat": r2_score(grp["heatpump_heat"], pred_heat[idx]),
            "mae_elec_w": mean_absolute_error(grp["heatpump_elec"], pred_elec[idx]),
            "mae_heat_w": mean_absolute_error(grp["heatpump_heat"], pred_heat[idx]),
        })
    
    # Per-capacity metrics
    for capacity, grp in test_df.groupby("capacity_kw"):
        # ... similar
    
    return pd.DataFrame(results)
```

### Backtest Output Files

```
models/
├── walk_forward_folds.csv      # Fold-level metrics
│   └── fold_id, train_start, test_end, r2_elec, r2_heat, mae_elec, ...
│
├── walk_forward_slices.csv     # Per-system/capacity breakdown
│   └── fold_id, slice_type, slice_value, r2_elec, mae_heat, ...
│
└── walk_forward_summary.json   # Aggregated averages
    └── avg_r2_elec, avg_r2_heat, avg_mae_elec, avg_mae_heat, ...
```

---

## 9. Stage 7: Streamlit Dashboard

### 📁 File: `streamlit_app.py`

### Concept: Interactive ML Operations Interface

The Streamlit app provides a web-based UI for:
1. Running pipeline stages
2. Making predictions
3. Viewing backtest results
4. Configuring parameters

### Tab Structure

```
┌─────────────────────────────────────────────────────────────────┐
│  Heat Pump Monitor Dashboard                                    │
├─────────┬─────────────┬──────────────────┬─────────────────────┤
│ Pipeline│ Predictions │ Backtest Results │ Settings            │
├─────────┴─────────────┴──────────────────┴─────────────────────┤
│                                                                 │
│  [Tab content displayed here]                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline Tab

```python
# Discover available stages
stages = [
    {"id": "01_fetch", "label": "01 Fetch Data", "path": "src/01_fetch_data.py"},
    {"id": "02_features", "label": "02 Feature Engineering", "path": "src/02_feature_engineering.py"},
    {"id": "03_clean", "label": "03 Clean Data", "path": "src/03_clean_data.py"},
    {"id": "04_train", "label": "04 Train Model", "path": "src/04_train_model.py"},
    # ...
]

# Run stage button
for stage in stages:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(stage["label"])
    with col2:
        if st.button("Run", key=stage["id"]):
            with st.spinner(f"Running {stage['label']}..."):
                result = subprocess.run(
                    [sys.executable, stage["path"]],
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                if result.returncode == 0:
                    st.success("✅ Completed")
                else:
                    st.error(f"❌ Failed: {result.stderr}")
```

### Predictions Tab

```python
st.header("Make Predictions")

# File upload option
uploaded = st.file_uploader("Upload CSV with features", type="csv")

if uploaded:
    df = pd.read_csv(uploaded, parse_dates=["timestamp"])
    st.write(f"Loaded {len(df)} rows")
    
    if st.button("Generate Predictions"):
        predictions = predict_bundle(df, run_tag=latest_model_tag)
        
        st.dataframe(predictions[["timestamp", "pred_heatpump_elec", "pred_heatpump_heat", "pred_cop"]])
        
        st.download_button(
            "Download Predictions CSV",
            predictions.to_csv(index=False),
            "predictions.csv"
        )

# Single row prediction
st.subheader("Single Point Prediction")
with st.form("single_pred"):
    timestamp = st.date_input("Date")
    hour = st.slider("Hour", 0, 23, 12)
    outside_temp = st.number_input("Outside Temp (°C)", -20.0, 40.0, 10.0)
    room_temp = st.number_input("Room Temp (°C)", 10.0, 30.0, 20.0)
    capacity = st.selectbox("System Capacity (kW)", [4, 6, 8])
    
    if st.form_submit_button("Predict"):
        # Build single row DataFrame
        single_df = build_single_row_features(timestamp, hour, outside_temp, room_temp, capacity)
        result = predict_bundle(single_df, run_tag=latest_model_tag)
        
        st.metric("Predicted Electricity", f"{result['pred_heatpump_elec'].iloc[0]:.0f} W")
        st.metric("Predicted Heat", f"{result['pred_heatpump_heat'].iloc[0]:.0f} W")
        st.metric("Predicted COP", f"{result['pred_cop'].iloc[0]:.2f}")
```

### Backtest Results Tab

```python
st.header("Backtest Results")

# Load summary
summary = json.load(open("models/walk_forward_summary.json"))

# Display key metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Avg R² Elec", f"{summary['avg_r2_elec']:.3f}")
with col2:
    st.metric("Avg R² Heat", f"{summary['avg_r2_heat']:.3f}")
with col3:
    st.metric("Avg MAE Elec", f"{summary['avg_mae_elec_w']:.1f} W")
with col4:
    st.metric("Avg MAE Heat", f"{summary['avg_mae_heat_w']:.1f} W")

# Fold-level table
fold_df = pd.read_csv("models/walk_forward_folds.csv")
st.subheader("Per-Fold Metrics")
st.dataframe(fold_df)

# Slice analysis chart
slice_df = pd.read_csv("models/walk_forward_slices.csv")
st.subheader("Per-System Performance")
fig = px.bar(
    slice_df[slice_df["slice_type"] == "series_id"],
    x="slice_value",
    y="r2_heat",
    title="Heat R² by System"
)
st.plotly_chart(fig)
```

---

## 10. Key Concepts Summary

### 🎯 Core ML Concepts Demonstrated

| Concept | Where Applied | Why Important |
|---------|---------------|---------------|
| **Temporal Train/Test Split** | Stage 4 | Prevents future data leakage |
| **Walk-Forward Validation** | Stage 7 | Simulates real deployment over time |
| **Feature Engineering** | Stage 2 | Domain knowledge → predictive power |
| **Imputation** | Stage 4, 5 | Handles missing sensor data |
| **Multi-Target Regression** | Stage 4 | Predicts correlated outputs together |
| **Post-Prediction Guardrails** | Stage 4, 5 | Ensures physical validity |
| **Slice Calibration** | Stage 4, 5 | Per-system bias correction |

### 🔑 Critical Design Decisions

1. **Why XGBoost over Neural Networks?**
   - Tabular data with engineered features
   - Interpretable feature importance
   - Handles missing values gracefully
   - Fast training and inference

2. **Why log1p transformation?**
   - Energy consumption is right-skewed
   - Stabilizes variance
   - Reduces outlier impact

3. **Why separate runtime classifier?**
   - OFF state is qualitatively different
   - Gating mechanism prevents false heating predictions
   - Achieves perfect F1 = 1.0

4. **Why COP guardrails?**
   - Model can predict impossible values
   - COP > 8 violates thermodynamics
   - Ensures all outputs are physically valid

### 📊 Production Metrics

```
Current Model Performance (v20260331):
├── R² Electricity:  0.74 (target: 0.80)
├── R² Heat:         0.78 (target: 0.80)
├── Energy Error:    3.85% ✅ (target: <5%)
├── COP Error:       2.71% ✅ (target: <6%)
├── Runtime F1:      1.00 ✅ (perfect)
└── COP Validity:    100% ✅ (all predictions physical)

Verdict: PRODUCTION READY (business metrics pass)
```

---

## 11. Exercises & Assignments

### Exercise 1: Feature Importance Analysis (Beginner)

**Task:** Run the training pipeline and analyze feature importance.

1. Train the model using default settings
2. Load the feature importance from the trained model
3. Answer: Which features are most important for electricity vs heat prediction?
4. Why do you think `heating_on_lag1` is more important for heat than electricity?

### Exercise 2: Hyperparameter Experiment (Intermediate)

**Task:** Modify hyperparameters and observe impact.

1. Change `max_depth` from 7 to 4. What happens to R²?
2. Change `learning_rate` from 0.02 to 0.1. What happens?
3. Remove `early_stopping_rounds`. Does the model overfit?

### Exercise 3: New Feature Engineering (Intermediate)

**Task:** Add a new physics-informed feature.

1. Create `efficiency_ratio = cop / capacity_kw` (normalized efficiency)
2. Add it to the feature set
3. Retrain and compare metrics
4. Did it improve? Why or why not?

### Exercise 4: System Exclusion Analysis (Advanced)

**Task:** Investigate poor-performing systems.

1. System 228 has R² = -0.388 for heat. Why?
2. Remove System 228 from training data
3. Retrain and compare overall metrics
4. Does the model improve on other systems?

### Exercise 5: Alternative Model Architecture (Advanced)

**Task:** Replace XGBoost with LightGBM.

1. Install LightGBM: `pip install lightgbm`
2. Modify `04_train_model.py` to use `LGBMRegressor`
3. Tune hyperparameters for LightGBM
4. Compare metrics with XGBoost

### Exercise 6: Prediction Interval Implementation (Expert)

**Task:** Add uncertainty quantification.

1. Research quantile regression
2. Train models for 5th and 95th percentiles
3. Add prediction intervals to the output
4. Evaluate coverage: Do 90% of true values fall within intervals?

---

## 12. Glossary

| Term | Definition |
|------|------------|
| **COP** | Coefficient of Performance = Heat Output / Electricity Input. Higher is better. |
| **HDH** | Heating Degree Hours. Cumulative measure of how cold it is below a base temperature. |
| **Walk-Forward** | Validation method that trains on past, tests on future, then advances in time. |
| **Temporal Leakage** | Bug where future information is used to predict past (causes overly optimistic metrics). |
| **Slice Calibration** | Learning per-system bias corrections to improve predictions. |
| **Guardrail** | Post-prediction rule that ensures outputs are physically valid. |
| **Lag Feature** | Value from a previous time step (e.g., `elec_lag1` = electricity 1 hour ago). |
| **Cyclical Encoding** | Using sine/cosine to represent periodic features (hour, month) smoothly. |
| **Imputation** | Filling missing values with estimates (e.g., median, interpolation). |
| **XGBoost** | eXtreme Gradient Boosting - an ensemble of decision trees trained sequentially. |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | March 31, 2026 | Initial comprehensive teaching guide |

---

*This documentation is designed for classroom use. Students should have access to the full codebase alongside this guide.*

**Repository:** `D:\Client Projects\Heat Pump Monitor\HeatPump`  
**Contact:** [Instructor Contact Information]
