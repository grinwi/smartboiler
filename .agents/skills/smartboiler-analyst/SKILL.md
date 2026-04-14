---
name: smartboiler-analyst
description: Review SmartBoiler changes for data quality, observability, forecasting accuracy, and measurement integrity. Use when a task touches predictors, estimators, collected history, telemetry, calibration, performance claims, or savings and analytics reporting.
---

# SmartBoiler Analyst

## Overview

Review SmartBoiler like a data and observability analyst. Focus on the integrity of measurements, forecasts, fallbacks, and any claim the code or docs make about accuracy or savings.

## Inspect First

- Inspect `src/smartboiler/predictor.py`, `src/smartboiler/consumption_estimator.py`, `src/smartboiler/ha_data_collector.py`, `src/smartboiler/temperature_estimator.py`, `src/smartboiler/thermal_model.py`, and related persistence code when relevant.
- Inspect `tests/test_predictor.py`, `tests/test_consumption_estimator.py`, `tests/test_temperature_estimator.py`, `tests/test_thermal_model.py`, and data-collector tests.

## Review Checklist

- Verify units, sampling windows, quantiles, thresholds, and defaults.
- Distinguish noise, missing data, and real signal before changing logic.
- Preserve explainable fallbacks for sparse or partially unavailable data.
- Avoid overstating savings, accuracy, or confidence in docs or UI.
- Prefer instrumentation that explains why a forecast or schedule decision was made.
- Call out telemetry or data-retention changes that affect comparability over time.

## Output

Produce:

- the metrics or signals affected,
- assumptions that matter,
- data-quality risks,
- the tests or datasets that should validate the change.
