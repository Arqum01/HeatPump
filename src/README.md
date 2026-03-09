# Daikin Heat Pump ML Pipeline
## Run Order

```
python 01_fetch_data.py          # Download from HPM API
python 02_feature_engineering.py # Build all features
python 03_clean_data.py          # Apply physical cleaning rules
python 04_train_model.py         # Train + evaluate XGBoost
```

## File Structure
```
data/
  raw/
    system_615_raw.csv           # One file per system (easy debugging)
    system_364_raw.csv
    ...
    fetch_log.csv                # Which systems succeeded/failed
  processed/
    daikin_features.csv          # All features before cleaning
    daikin_clean.csv             # ML-ready dataset
    cleaning_report.csv          # Row retention per system
models/
  xgb_heatpump.json              # Trained model
  training_report.txt            # R², MAE, COP error
```

## Key Design Decisions

| Decision | Why |
|---|---|
| END = "01-02-2026" not "30-01-2026" | inclusive="left" makes end exclusive |
| UTC timezone | Prevents DST ambiguity in spring/autumn |
| Calendar-first padding | Never trim time axis to fit bad data |
| flowT/returnT REMOVED from features | Data leakage — only exist after pump runs |
| _lag1 features ADDED | Last hour's values — legitimate, no leakage |
| groupby before shift | Prevents System 615 bleeding into System 364 |
| dropna on core cols only | Saves System 228 (always missing roomT) |
| Time-based split, never shuffle | Train on past, test on future |
| Single model + math conversion | kWh = W/1000, no inconsistency risk |

## Expected R² Range
- With leaky features (flowT etc): ~0.97 ← FAKE
- Honest baseline (1 month data):  ~0.68
- + lag features:                  ~0.82
- + 12 months of data:             ~0.86–0.88 ← REAL, production-grade
