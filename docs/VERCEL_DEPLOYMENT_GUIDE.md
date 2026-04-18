# Vercel Deployment Guide

## Purpose

This guide explains how to deploy the production prediction API from this repository on Vercel.

The deployment target is the serverless API entrypoint in `api/index.py`.

## What Changed For Production

- Added Vercel entrypoint: `api/index.py`
- Added Vercel config: `vercel.json`
- Added deployment dependency list: `requirements.txt`
- Added deployment bundle script: `scripts/prepare_production_bundle.py`
- Added deployment ignore rules: `.vercelignore`

## Why This Approach

Vercel runs Python as serverless functions. Streamlit apps are long-running interactive servers and are not a good direct fit for Vercel serverless runtime.

This repository now supports:

- Local Streamlit apps for admin/customer UI
- Production API deployment on Vercel for inference

## Pre-Deployment Checklist

1. Activate your virtual environment.
2. Ensure training artifacts exist in `models/`.
3. Package a minimal deployment bundle.

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/prepare_production_bundle.py
```

The script creates `models/production` with only required files for the latest run.

## Vercel Project Setup

Set environment variables in your Vercel project:

- `MODEL_DIR=models/production`
- `MODEL_RUN_TAG=<optional run tag>`
- `CORS_ORIGINS=https://your-frontend-domain.com`

`MODEL_RUN_TAG` is optional. If omitted, the API uses the newest manifest in the active model directory.

## Deploy Command

```powershell
vercel --prod
```

## API Endpoints

### Health

- Method: `GET`
- Path: `/api/health`

Response contains status and active model metadata.

### List Runs

- Method: `GET`
- Path: `/api/runs`

Returns available run tags in the active model directory.

### Technical Prediction

- Method: `POST`
- Path: `/api/predict/technical`

Example payload:

```json
{
  "run_tag": null,
  "rows": [
    {
      "heatpump_outsideT": 5.5,
      "outsideT_3h_avg": 5.2,
      "hdh": 10.0,
      "heatpump_roomT": 20.0,
      "temp_deficit": 14.5,
      "load_ratio": 1.81,
      "capacity_kw": 8.0,
      "hour_sin": 0.5,
      "hour_cos": 0.866,
      "month_sin": 0.5,
      "month_cos": 0.866,
      "day_of_week": 2,
      "is_heating_season": 1
    }
  ]
}
```

### Customer Prediction

- Method: `POST`
- Path: `/api/predict/customer`

Example payload:

```json
{
  "outside_t": 6.0,
  "room_t": 20.0,
  "capacity_kw": 8.0,
  "overrides": {
    "day_of_week": 2
  }
}
```

## Local API Smoke Test

Run locally:

```powershell
python -m uvicorn api.index:app --host 127.0.0.1 --port 8000
```

Then test:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
```

## Troubleshooting

### No manifests found

- Confirm `models/production/run_manifest_*.json` exists.
- Verify `MODEL_DIR` value in Vercel env settings.

### Artifact file not found

- Re-run `python scripts/prepare_production_bundle.py`.
- Commit and push generated production model bundle files.

### 400 prediction failed

- Verify payload includes required model features.
- For customer endpoint, ensure numeric fields are valid numbers.
