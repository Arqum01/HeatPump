import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

# XGBoost
from xgboost import XGBRegressor

DATA_PATH = "data/processed/daikin_clean_01-03-2025_to_01-03-2026_hourly.csv"

# -----------------------------
# 1) Load and sort (time split)
# -----------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# -----------------------------
# 2) Define features and targets
# -----------------------------
FEATURES = [
    "heatpump_outsideT",
    "heatpump_flowT",
    "heatpump_returnT",
    "deltaT",
    "heatpump_roomT",
    "heatpump_flowrate",
    "capacity_kw",
]

# Targets: Watts and kWh
TARGET_W = "heatpump_elec"
TARGET_KWH = "elec_kwh"

# Columns used only for evaluation (NOT as features)
HEAT_KWH = "heat_kwh"

# Drop rows with any missing values in required columns
needed = FEATURES + [TARGET_W, TARGET_KWH, HEAT_KWH]
df = df.dropna(subset=needed).copy()

X = df[FEATURES]
y_w = df[TARGET_W].astype(float)
y_kwh = df[TARGET_KWH].astype(float)

# -----------------------------
# 3) Time-based split (80/20)
# -----------------------------
split_idx = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_w_train, y_w_test = y_w.iloc[:split_idx], y_w.iloc[split_idx:]
y_kwh_train, y_kwh_test = y_kwh.iloc[:split_idx], y_kwh.iloc[split_idx:]

heat_kwh_test = df[HEAT_KWH].iloc[split_idx:].astype(float)

print("Rows total:", len(df))
print("Train rows:", len(X_train), "Test rows:", len(X_test))
print("Feature columns:", FEATURES)

# -----------------------------
# 4) Helper metrics
# -----------------------------
def regression_report(name: str, y_true, y_pred, unit: str):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name} ({unit})")
    print(f"  MAE : {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R^2 : {r2:.4f}")
    return mae, rmse, r2

def energy_and_cop_report(label: str, heat_kwh_true: np.ndarray, elec_kwh_true: np.ndarray, elec_kwh_pred: np.ndarray):
    true_e = elec_kwh_true.sum()
    pred_e = elec_kwh_pred.sum()

    energy_err_pct = abs(pred_e - true_e) / (true_e + 1e-9) * 100.0

    # COP over the test period
    true_cop = heat_kwh_true.sum() / (true_e + 1e-9)
    pred_cop = heat_kwh_true.sum() / (pred_e + 1e-9)
    cop_err_pct = abs(pred_cop - true_cop) / (true_cop + 1e-9) * 100.0

    print(f"\n{label} — Test-period energy & COP")
    print(f"  True total elec (kWh): {true_e:.3f}")
    print(f"  Pred total elec (kWh): {pred_e:.3f}")
    print(f"  Energy error (%):      {energy_err_pct:.2f}%")
    print(f"  True COP:              {true_cop:.3f}")
    print(f"  Pred COP:              {pred_cop:.3f}")
    print(f"  COP error (%):         {cop_err_pct:.2f}%")

# -----------------------------
# 5) Train models for W target
# -----------------------------
gbr_w = GradientBoostingRegressor(random_state=42)
gbr_w.fit(X_train, y_w_train)
pred_gbr_w = gbr_w.predict(X_test)

xgb_w = XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=4,
)
xgb_w.fit(X_train, y_w_train)
pred_xgb_w = xgb_w.predict(X_test)

# Convert W predictions to kWh for energy/COP reporting
pred_gbr_kwh_from_w = pred_gbr_w / 1000.0
pred_xgb_kwh_from_w = pred_xgb_w / 1000.0

# -----------------------------
# 6) Train models for kWh target
# -----------------------------
gbr_kwh = GradientBoostingRegressor(random_state=42)
gbr_kwh.fit(X_train, y_kwh_train)
pred_gbr_kwh = gbr_kwh.predict(X_test)

xgb_kwh = XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=4,
)
xgb_kwh.fit(X_train, y_kwh_train)
pred_xgb_kwh = xgb_kwh.predict(X_test)

# -----------------------------
# 7) Reports (hourly + energy + COP)
# -----------------------------
print("\n==================== HOURLY METRICS ====================")
regression_report("GradientBoosting -> Electricity", y_w_test, pred_gbr_w, "W")
regression_report("XGBoost -> Electricity",         y_w_test, pred_xgb_w, "W")

regression_report("GradientBoosting -> Electricity", y_kwh_test, pred_gbr_kwh, "kWh")
regression_report("XGBoost -> Electricity",          y_kwh_test, pred_xgb_kwh, "kWh")

print("\n==================== ENERGY & COP (TEST PERIOD) ====================")
# Using kWh predictions (direct)
energy_and_cop_report("GradientBoosting (kWh target)", heat_kwh_test.values, y_kwh_test.values, pred_gbr_kwh)
energy_and_cop_report("XGBoost (kWh target)",          heat_kwh_test.values, y_kwh_test.values, pred_xgb_kwh)

# Using W predictions converted to kWh
energy_and_cop_report("GradientBoosting (W->kWh)",     heat_kwh_test.values, y_kwh_test.values, pred_gbr_kwh_from_w)
energy_and_cop_report("XGBoost (W->kWh)",              heat_kwh_test.values, y_kwh_test.values, pred_xgb_kwh_from_w)
