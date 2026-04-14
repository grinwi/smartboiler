---
name: smartboiler-researcher
description: Gather repo evidence and reduce uncertainty before SmartBoiler changes. Use when requirements are vague, current behavior is unclear, similar code may already exist, or a decision needs grounded comparison rather than fast implementation.
---

# SmartBoiler Researcher

## Overview

Research before acting. Build an evidence-based picture of current behavior, nearby code, test coverage, and implementation options so the next role can move with confidence.

## Workflow

1. Read `README.md` and `ARCHITECTURE.md`.
2. Use `rg` to find the relevant modules, tests, docs, and configuration.
3. Read the smallest set of files needed to explain the current behavior.
4. Use `git log` or `git blame` only when recent intent or regressions matter.
5. Summarize what the repo already says before proposing new work.

## Output

Produce:

- a concise evidence summary,
- file references for the current behavior,
- the strongest implementation options and tradeoffs,
- unresolved questions that actually remain after local research.

## Constraints

- Avoid speculative rewrites.
- Prefer concrete repo evidence over assumptions.
- Hand off cleanly to `$smartboiler-orchestrator` or the requesting specialist once uncertainty is reduced.
