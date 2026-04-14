---
name: smartboiler-orchestrator
description: Coordinate multi-role SmartBoiler development work across control logic, scheduling, Home Assistant integration, web setup flows, persistence, testing, security, analytics, and release readiness. Use when a task is non-trivial, spans multiple modules, needs balanced tradeoffs, or the user asks for stable, production-quality changes with cross-functional review.
---

# SmartBoiler Orchestrator

## Overview

Coordinate SmartBoiler work by selecting the right specialist skills, sequencing their reviews, and merging the outcome into one safe and shippable implementation.

## Start Here

- Read `README.md` and `ARCHITECTURE.md` before making repo-wide decisions.
- Read the touched modules and the closest tests in `tests/`.
- Treat boiler safety, Home Assistant compatibility, readability, and regression resistance as co-equal priorities.

## Specialist Selection

- Use `$smartboiler-researcher` when requirements, existing behavior, or module ownership are unclear.
- Use `$smartboiler-iot-expert` for sensors, relays, HDO logic, control loops, setup entities, scheduler timing, and physical-system assumptions.
- Use `$smartboiler-architect` for refactors, shared state, module boundaries, persistence design, and complexity control.
- Use `$smartboiler-security-engineer` for secrets, ingress, web routes, outbound requests, unsafe defaults, and stuck-on failure modes.
- Use `$smartboiler-analyst` for predictors, telemetry, history collection, calibration, and measurement quality.
- Use `$smartboiler-product-manager` for user-facing behavior, setup flow, scope cuts, migrations, and release framing.
- Use `$smartboiler-release-devops` for Docker, entrypoints, runtime startup, dependencies, packaging, upgrades, and release readiness.
- Use `$smartboiler-qa` for regression mapping, edge cases, and validation strategy.
- Use only the roles that materially change the result.

## Standard Review Orders

### Research-Heavy Or Ambiguous Tasks

1. Start with `$smartboiler-researcher`.
2. Bring in the domain specialist with the highest uncertainty next: usually `$smartboiler-iot-expert`, `$smartboiler-analyst`, or `$smartboiler-security-engineer`.
3. Use `$smartboiler-architect` if the options span module boundaries or introduce new structure.
4. Use `$smartboiler-product-manager` when the decision changes setup UX, defaults, or release scope.
5. End with `$smartboiler-qa` to turn the findings into a validation plan.

### Implementation Changes

1. Build context from docs, touched files, and relevant tests.
2. Pull in `$smartboiler-iot-expert` first when control behavior, entities, timing, or physical assumptions are involved.
3. Pull in `$smartboiler-security-engineer` when auth, routes, config, network calls, or unsafe failure modes are involved.
4. Pull in `$smartboiler-architect` when the change spans workflows, persistence, or shared state.
5. Pull in `$smartboiler-analyst` when forecasts, telemetry, calibration, or savings logic are affected.
6. Pull in `$smartboiler-product-manager` when the user-facing setup, defaults, docs, or migration story changes.
7. Pull in `$smartboiler-release-devops` when startup, packaging, dependencies, Docker, or upgrade behavior changes.
8. Finish with `$smartboiler-qa`.

### Minimal Delivery Path

1. Collect concrete findings from only the roles that matter.
2. Separate blockers from nice-to-haves.
3. Implement the smallest safe change that fully resolves the user request.
4. Run targeted tests first, then broaden validation only where risk remains.
5. Summarize the outcome, validations, and residual risks.

## Repo Priorities

- Keep control decisions deterministic and explainable.
- Preserve safe fallback behavior when Home Assistant, sensors, or price feeds are missing.
- Keep persisted setup and state compatible unless migration is explicit.
- Keep startup and upgrade behavior predictable for existing installations.
- Match the existing style and update tests or docs when behavior changes.

## Deliverable

Report:

- which specialist skills were used,
- what issues they surfaced,
- what changed,
- what was validated,
- what residual risk remains.
