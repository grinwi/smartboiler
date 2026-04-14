---
name: smartboiler-release-devops
description: Review SmartBoiler changes for packaging, runtime operations, deployment behavior, and release readiness. Use when a task touches Docker, entrypoints, versioning, addon-facing startup, dependency changes, config files, operational logging, or anything that could break installation, upgrades, or runtime stability.
---

# SmartBoiler Release DevOps

## Overview

Review SmartBoiler like a release and operations engineer. Focus on whether the repo still builds, starts, upgrades, and behaves predictably in Home Assistant add-on style environments.

## Inspect First

- Read `README.md`, `Dockerfile`, `config.yml`, `deploy_docker.mk`, `setup.py`, and `requirements.txt`.
- Inspect startup and runtime modules such as `src/smartboiler/main.py`, `src/smartboiler/controller.py`, `src/smartboiler/web_server.py`, and configuration-loading code when relevant.
- Inspect tests that cover startup, setup, web routes, or persisted state when a release-risking behavior changes.

## Review Checklist

- Check whether dependency, packaging, or entrypoint changes are minimal and reversible.
- Check whether runtime defaults remain compatible with Home Assistant add-on expectations.
- Check whether logging is useful for diagnosis without becoming noisy or leaking secrets.
- Check whether configuration and persisted state changes need migration notes or version bumps.
- Call out operational hazards: startup ordering, missing env vars, partial setup state, and recovery after restart.
- Prefer release-safe changes that preserve existing installs over broad infrastructure churn.

## Output

Produce:

- packaging or deployment risks,
- startup and upgrade concerns,
- runtime observability gaps,
- release-note or migration actions,
- the smallest checks needed before shipping.
