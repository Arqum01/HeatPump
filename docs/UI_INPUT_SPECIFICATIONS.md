# Heat Pump Monitor - Prediction Input Specifications

## User Guide for Heat Pump Predictions

This document specifies the full **35-feature serving schema** used for model predictions. It also clarifies how simplified customer inputs are expanded into this schema.

### Input Modes

- **Technical mode (admin app):** direct feature-level entry for diagnostics and power users.
- **Customer mode (customer app):** simplified inputs are collected, and the app derives missing schema fields automatically.

Customer-mode defaults include:

- Heat pump size default of **8.0 kW**.
- Currency default of **GBP** and unit rate default of **0.28**.
- Internal model diagnostics hidden from customer flow.

---

## Complete Feature Input Specifications

| # | Feature Name | Data Type | Range/Units | Default | Description |
|---|---|---|---|---|---|
| **CURRENT CONDITIONS** |
| 1 | `heatpump_outsideT` | float | -20.0 to 40.0 °C | 6.0 | Outside ambient temperature |
| 2 | `outsideT_3h_avg` | float | -20.0 to 40.0 °C | 6.0 | 3-hour rolling average of outside temp |
| 3 | `heatpump_roomT` | float | 10.0 to 30.0 °C | 20.0 | Indoor room temperature |
| 4 | `capacity_kw` | float | 1.0 to 30.0 kW | 6.0 (technical), 8.0 (customer form) | System nominal heat pump capacity |
| **DERIVED CONDITIONS** |
| 5 | `hdh` | float | 0.0 to ~40.0 | 0.0 | Heating Degree Hours (BASE_TEMP - outsideT, clipped ≥0) |
| 6 | `temp_deficit` | float | -30.0 to 50.0 °C | ~14.0 | Temperature difference (roomT - outsideT) |
| 7 | `load_ratio` | float | 0.0 to ~8.0 | ~2.3 | Normalized heating burden (temp_deficit / capacity_kw) |
| 8 | `is_heating_season` | int | 0 or 1 | 0 | Binary flag: 1 if outsideT < 15.5°C, else 0 |
| **TIME ENCODING** |
| 9 | `hour_sin` | float | -1.0 to 1.0 | 0.707 | Cyclical hour encoding: sin(2π×hour/24) |
| 10 | `hour_cos` | float | -1.0 to 1.0 | 0.707 | Cyclical hour encoding: cos(2π×hour/24) |
| 11 | `month_sin` | float | -1.0 to 1.0 | 0.866 | Cyclical month encoding: sin(2π×month/12) |
| 12 | `month_cos` | float | -1.0 to 1.0 | -0.5 | Cyclical month encoding: cos(2π×month/12) |
| 13 | `day_of_week` | int | 0-6 | 1 | Day of week (0=Monday, 6=Sunday) |
| **ELECTRICITY HISTORY** |
| 14 | `elec_lag1` | float | 0.0 to ~3000.0 W | 0.0 | Electricity consumption 1 hour ago |
| 15 | `elec_lag2` | float | 0.0 to ~3000.0 W | 0.0 | Electricity consumption 2 hours ago |
| 16 | `elec_lag3` | float | 0.0 to ~3000.0 W | 0.0 | Electricity consumption 3 hours ago |
| 17 | `elec_lag4` | float | 0.0 to ~3000.0 W | 0.0 | Electricity consumption 4 hours ago |
| 18 | `elec_lag6` | float | 0.0 to ~3000.0 W | 0.0 | Electricity consumption 6 hours ago |
| 19 | `elec_lag24` | float | 0.0 to ~3000.0 W | 0.0 | Electricity consumption 24 hours ago |
| 20 | `elec_lag168` | float | 0.0 to ~3000.0 W | 0.0 | Electricity consumption 168 hours ago (last week) |
| 21 | `elec_lag1_pct` | float | 0.0 to ~0.375 | 0.0 | Normalized lag: elec_lag1 / (capacity_kw × 1000) |
| **TEMPERATURE HISTORY** |
| 22 | `flowT_lag1` | float | 10.0 to 60.0 °C | 35.0 | Supply/flow temperature 1 hour ago |
| 23 | `returnT_lag1` | float | 10.0 to 50.0 °C | 30.0 | Return temperature 1 hour ago |
| 24 | `deltaT_house_lag1` | float | -10.0 to 40.0 °C | ~5.0 | Thermal gradient: flowT_lag1 - returnT_lag1 |
| 25 | `deltaT_lift_lag1` | float | -10.0 to 50.0 °C | ~29.0 | Lift pressure: flowT_lag1 - outsideT |
| 26 | `lift_per_kw_lag1` | float | -10.0 to ~8.0 °C/kW | ~4.8 | Normalized lift: deltaT_lift_lag1 / capacity_kw |
| **OPERATIONAL HISTORY** |
| 27 | `flowrate_lag1` | float | 0.0 to ~60.0 L/min | 0.0 | Fluid flow rate 1 hour ago |
| 28 | `heating_on_lag1` | float | 0.0 or 1.0 | 0.0 | Was pump actively heating 1 hour ago? |
| 29 | `heating_on_lag24` | float | 0.0 or 1.0 | 0.0 | Was pump actively heating 24 hours ago? |
| 30 | `run_hours` | int | 0 to ~24+ hours | 0 | Contiguous runtime (current heating streak duration) |
| **PERFORMANCE HISTORY** |
| 31 | `cop_lag1` | float | 0.0 to ~8.0 | 3.0 | Coefficient of Performance 1 hour ago |
| 32 | `was_defrost_lag1` | float | 0.0 or 1.0 | 0.0 | Binary: 1 if cop_lag1 < 1.0 (defrost cycle) |
| 33 | `heat_lag1` | float | 0.0 to ~5000.0 W | 0.0 | Heat output 1 hour ago |
| 34 | `heat_lag24` | float | 0.0 to ~5000.0 W | 0.0 | Heat output 24 hours ago |
| 35 | `heat_lag168` | float | 0.0 to ~5000.0 W | 0.0 | Heat output 168 hours ago (last week) |

---

## Quick Input Guide

### **Essential Current Conditions** (Required)
1. **Outside Temperature** (`heatpump_outsideT`): Current outdoor temperature in °C
2. **Room Temperature** (`heatpump_roomT`): Current indoor temperature in °C  
3. **System Capacity** (`capacity_kw`): Your heat pump's rated capacity in kW
4. **Time Information**: Hour of day (0-23), month (1-12), day of week (0-6)

### **Historical Context** (For Better Accuracy)
- **Recent Electricity Use**: What was the power consumption 1, 2, 3+ hours ago?
- **Recent Temperatures**: Flow and return temperatures from previous hour
- **Recent Operation**: Was the system heating in the past 1-24 hours?
- **Performance History**: COP and heat output from previous hours/days

### **If You Don't Know Historical Values**
- **Default to 0.0** for all lag values (electricity, heat history)
- **Use typical values**: flowT_lag1=35°C, returnT_lag1=30°C, cop_lag1=3.0
- **System will still predict**, but accuracy improves with real historical data

---

## Data Type Categories

| Category | Count | Data Type | Examples |
|----------|-------|-----------|----------|
| **Float Features** | 26 | float | Temperatures, energy values, ratios |
| **Integer Features** | 2 | int | `day_of_week`, `run_hours` |
| **Boolean Features** | 7 | int (0/1) | `is_heating_season`, `heating_on_lag1`, `was_defrost_lag1` |

---

## Input Validation Rules

### **Physics Constraints**
- **COP Range**: 0.0-8.0 (automatically enforced after prediction)
- **Base Temperature**: 15.5°C (for heating degree hours calculation)
- **Standby Threshold**: 50W (below this = system effectively off)

### **Derived Feature Calculations**
- `hdh` = max(0, 15.5 - heatpump_outsideT)
- `temp_deficit` = heatpump_roomT - heatpump_outsideT
- `load_ratio` = temp_deficit / capacity_kw
- `deltaT_house_lag1` = flowT_lag1 - returnT_lag1
- `deltaT_lift_lag1` = flowT_lag1 - heatpump_outsideT
- `lift_per_kw_lag1` = deltaT_lift_lag1 / capacity_kw
- `elec_lag1_pct` = elec_lag1 / (capacity_kw × 1000)

### **Time Encoding**
- `hour_sin` = sin(2π × hour / 24)
- `hour_cos` = cos(2π × hour / 24)
- `month_sin` = sin(2π × month / 12)
- `month_cos` = cos(2π × month / 12)

---

## Prediction Outputs

After entering all features, you'll receive:

1. **Predicted Electricity Consumption** (W)
2. **Predicted Heat Output** (W)  
3. **Predicted COP** (Coefficient of Performance)

All predictions are validated and constrained to physically realistic ranges.