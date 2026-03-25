### 🔧 Copilot Instruction — Heat Pump Data Pipeline (Daikin Systems)

You are building a data pipeline using the Heat Pump Monitor (HPM) API.

---

## 📡 API Reference

Base API helper:
https://heatpumpmonitor.org/api-helper

### Example system:

* System ID: 615 (8 kW Daikin)
  https://heatpumpmonitor.org/system/view?id=615

### Available timeseries:

https://heatpumpmonitor.org/timeseries/available?id=615

---

## 📊 Data Collection

Fetch hourly time-series data using:

https://heatpumpmonitor.org/timeseries/data

### Required query parameters:

* `id` → system ID
* `feeds` → comma-separated variables
* `start` → DD-MM-YYYY
* `end` → DD-MM-YYYY
* `interval=3600` → hourly data
* `average=1`
* `timeformat=notime`

---

## 🔢 Variables

### Independent Variables (Features):

* `heatpump_outsideT` → outside temperature (°C)
* `heatpump_roomT` → room temperature (°C)
* `heatpump_flowT` → water flow temperature (°C)
* `heatpump_returnT` → return temperature (°C)
* `heatpump_flowrate` → water flow rate (l/min)

### Dependent Variables (Targets):

* `heatpump_elec` → electricity consumption (W)
* `heatpump_heat` → heat output (W)

---

## 🧠 Derived Metrics

* **COP (Coefficient of Performance)**
  `COP = heatpump_heat / heatpump_elec`

* **Energy Consumption (kWh)**
  Convert W → kWh using:
  `value / 1000`

---

## 🏭 System Scope (Daikin Only)

Use ONLY the following system IDs:

### 8 kW systems:

* 615, 364, 44

### 6 kW systems:

* 162, 351, 587

### 4 kW system:

* 228 (NOTE: missing room temperature data)

---

## ⚠️ Data Notes

* Data is sampled hourly (3600 seconds)
* Some systems may have missing values
* System 228 has missing `heatpump_roomT`
* Data range can be adjusted (last month, year, etc.)

---

## 🧱 Implementation Guidelines

1. Fetch data asynchronously for multiple systems
2. Align all data to a consistent hourly time index
3. Handle missing values carefully:

   * Do NOT blindly interpolate target variables
4. Ensure all features have equal length (pad or trim if needed)
5. Add derived features like:

   * COP
   * temperature differences
6. Keep system-level separation using `system_id`
7. Save output in CSV format

---

## 🎯 Goal

Build a clean dataset suitable for:

* machine learning modeling
* performance analysis
* energy efficiency evaluation

---

## 🚫 Constraints

* Do NOT include non-Daikin systems
* Do NOT assume perfect data (handle missing values)
* Do NOT hardcode time ranges unless specified
* Avoid data leakage when creating features

---
