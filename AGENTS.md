# SmartBoiler Repo Guide

## Core Goals

- Keep boiler control safe, deterministic, and explainable.
- Prefer stable, readable Python over clever or opaque abstractions.
- Preserve Home Assistant add-on compatibility, persisted setup, and safe fallback behavior unless an explicit migration is part of the task.

## Use The Repo Agents

- Use `$smartboiler-orchestrator` for non-trivial work, cross-module changes, or anything that should be reviewed from multiple perspectives.
- Bring in only the specialist lenses that matter:
  - `$smartboiler-iot-expert` for relays, sensors, control loops, Home Assistant entities, HDO, thermostat, timing, and physical behavior.
  - `$smartboiler-product-manager` for user value, setup UX, defaults, migrations, and release framing.
  - `$smartboiler-qa` for regressions, edge cases, and validation strategy.
  - `$smartboiler-architect` for module boundaries, shared state, refactors, and complexity control.
  - `$smartboiler-security-engineer` for secrets, ingress, unsafe defaults, and failure-mode safety.
  - `$smartboiler-analyst` for telemetry, predictors, data quality, and measurement integrity.
  - `$smartboiler-researcher` for repo evidence gathering when requirements or existing behavior are unclear.
  - `$smartboiler-release-devops` for Docker, startup, dependencies, packaging, upgrade safety, and release readiness.

## Standard Review Order

- For ambiguous or exploratory tasks:
  - Start with `$smartboiler-researcher`.
  - Add the most relevant domain role next.
  - Add `$smartboiler-architect` if the change may cross module boundaries.
  - Finish with `$smartboiler-qa`.
- For implementation tasks:
  - Start with `$smartboiler-iot-expert` when physical control or HA behavior is involved.
  - Add `$smartboiler-security-engineer`, `$smartboiler-architect`, `$smartboiler-analyst`, `$smartboiler-product-manager`, and `$smartboiler-release-devops` only when their concerns are directly in scope.
  - Finish with `$smartboiler-qa`.

## Build Context First

- Read `README.md` for product intent and operating model.
- Read `ARCHITECTURE.md` before changing cross-module behavior or workflow boundaries.
- Read `MANUAL.md` when hardware or installation assumptions matter.
- Read the nearest tests in `tests/` before changing behavior.

## High-Risk Areas

- `src/smartboiler/controller.py`
- `src/smartboiler/scheduler.py`
- `src/smartboiler/temperature_estimator.py`
- `src/smartboiler/thermal_model.py`
- `src/smartboiler/legionella_protector.py`
- `src/smartboiler/ha_client.py`
- `src/smartboiler/web_routes.py`
- `src/smartboiler/setup_config.py`
- `src/smartboiler/state_store.py`
- `Dockerfile`
- `config.yml`

## Validation

- Run targeted `pytest` files for the touched behavior first.
- Expand to neighboring tests when shared control logic, persistence, or setup flows change.
- Update docs and tests when externally visible behavior changes.
- Call out residual risk when live Home Assistant, hardware, or network behavior cannot be exercised locally.
