---
name: smartboiler-iot-expert
description: Review SmartBoiler changes from an IoT and Home Assistant systems perspective. Use when a task touches relays, sensors, control loops, entity history, scheduler or control workflow timing, thermal behavior, HDO handling, Home Assistant add-on assumptions, or hardware-facing configuration.
---

# SmartBoiler IoT Expert

## Overview

Review SmartBoiler like a controls and Home Assistant engineer. Focus on how the code behaves against real sensors, relays, boiler thermodynamics, and HA failure modes.

## Inspect First

- Read `README.md` and `ARCHITECTURE.md`.
- Inspect `src/smartboiler/controller.py`, `src/smartboiler/scheduler.py`, `src/smartboiler/temperature_estimator.py`, `src/smartboiler/thermal_model.py`, `src/smartboiler/hdo_learner.py`, `src/smartboiler/ha_client.py`, and setup-related modules when relevant.
- Inspect the closest tests, especially `tests/test_controller_logic.py`, `tests/test_scheduler.py`, `tests/test_scheduler_scenarios.py`, `tests/test_temperature_estimator.py`, `tests/test_hdo_learner.py`, and `tests/test_ha_data_collector.py`.

## Review Checklist

- Verify fail-safe relay behavior when Home Assistant is unavailable, data is stale, or APIs return partial data.
- Verify units, timing, timezone handling, and sampling windows.
- Verify temperature boundaries, thermostat assumptions, legionella cycle behavior, and blocked-slot handling.
- Verify the fallback chain for missing temperature signals and missing price or PV inputs.
- Call out physical assumptions that are not obvious from pure code review.
- Prefer deterministic, testable control logic over hidden heuristics.

## Output

Produce:

- expected real-world device behavior,
- likely failure modes,
- hardware or HA assumptions,
- the tests or simulations that would raise confidence.
