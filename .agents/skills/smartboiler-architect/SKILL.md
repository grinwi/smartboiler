---
name: smartboiler-architect
description: Protect SmartBoiler architecture, readability, and maintainability. Use when a task involves refactoring, cross-module changes, shared state, workflow boundaries, persistence design, new integrations, or any change that risks accidental complexity or muddled responsibilities.
---

# SmartBoiler Architect

## Overview

Review SmartBoiler like a system architect. Focus on boundaries, coupling, state flow, and how to keep the codebase legible as features grow.

## Inspect First

- Read `ARCHITECTURE.md`.
- Inspect the touched modules and nearby dependencies.
- Pay extra attention to `src/smartboiler/controller.py`, `src/smartboiler/state_store.py`, `src/smartboiler/scheduler.py`, `src/smartboiler/predictor.py`, `src/smartboiler/web_server.py`, and `src/smartboiler/web_routes.py` when the change spans workflows.

## Review Checklist

- Keep forecast workflow, control workflow, setup flow, and web concerns clearly separated.
- Avoid turning `controller.py` into a grab-bag; extract helpers only when they create a real boundary.
- Maintain explicit data contracts and units across scheduler, estimator, predictor, and storage code.
- Preserve thread-safe shared state and simple state handoff between loops.
- Prefer deletion, simplification, and naming clarity over introducing new layers.
- Update architecture-facing docs when responsibilities move.

## Output

Produce:

- coupling or layering concerns,
- the simplest sustainable structure,
- refactor opportunities worth doing now,
- refactors that should wait.
