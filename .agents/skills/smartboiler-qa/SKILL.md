---
name: smartboiler-qa
description: Review SmartBoiler work for regressions, validation gaps, and edge-case coverage. Use when a task changes behavior, tests, persistence, scheduler or control logic, web routes, setup flows, or anything that could affect safe operation and release confidence.
---

# SmartBoiler QA

## Overview

Review SmartBoiler like a release-focused QA engineer. Map code changes to failure scenarios, test coverage, and the confidence still missing after implementation.

## Inspect First

- Read the touched code and the nearest tests in `tests/`.
- Prioritize `tests/test_controller_logic.py`, `tests/test_scheduler.py`, `tests/test_scheduler_scenarios.py`, `tests/test_web_routes.py`, `tests/test_setup_config.py`, `tests/test_predictor.py`, and persistence-related tests when relevant.

## Review Checklist

- Map each behavior change to an existing or needed test.
- Enumerate edge cases: missing HA data, unknown entity states, timezone rollover, empty spot prices, blocked HDO slots, stale temperature readings, calendar overrides, legionella due state, and older persisted state.
- Prefer targeted `pytest` selection before broader test sweeps.
- Distinguish blocker, high, medium, and low risks.
- Call out user-visible behavior that changed without matching docs or tests.

## Output

Produce findings first, then:

- the most important regression risks,
- tests run or tests still needed,
- residual validation gaps,
- the confidence level for shipping.
