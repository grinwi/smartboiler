# Concept: from SmartBoiler to a Home Energy Manager

Status: **in progress** — see "Done so far" under the roadmap. Audience: the maintainer and whoever implements it.
Scope: this library (all the logic, UI and tests) and the thin `grinwi/smartboiler-add-on` repo that only packages it for Home Assistant.

Goal: extend the boiler controller to a household with any mix of
**PV (FVE), battery, heat pump (tepelné čerpadlo), air conditioning and a hot-water boiler**,
while keeping setup short and the control explainable.

---

## 1. Where we are today

| Area | Today | Limit for the new goal |
|------|-------|------------------------|
| Devices | One boiler behind one relay | `controller.py` (1 400 lines) is boiler-specific end to end |
| Planner | Greedy fill of cheapest hours until `min_tmp` holds, 24 × 1 h slots | Good core idea, reusable. Hourly slots, while EU day-ahead market is 15-min |
| Price | Raw spot EUR/MWh; PV hour = cost `-1`, boost = `-2` | No distribution fee / VAT / low-high tariff, no export price. "PV is free" is wrong when export is paid, "sell_first" is a workaround for that |
| PV forecast | Forecast entity, else **current PV power repeated for 24 h** | Unusable for planning; no house base load subtracted, so "surplus" is not really surplus |
| Battery | Simulated inside the boiler scheduler | Belongs to a shared energy picture, not to one device |
| Reaction time | Plan per hour, executed every 60 s | No reaction to real export at the meter (cloud passes, kettle on) |
| Setup | Web wizard, ~35 fields, entity IDs typed/picked one by one | Would grow to 100+ fields with more devices |
| HA feedback | Only own ingress dashboard | No entities in HA → no automations, no Lovelace cards |
| Learning | Consumption histogram, HDO learner, cooling model | Solid and tested, reusable for house load and house thermal model |

Housekeeping found on the way (cheap, do first):

This repo:

- Unrelated or dead files in the tree: `test.txt` (study notes), `prediction_form.pdf`, the
  LSTM and analysis notebooks, empty `cmd`, `control/`, `scripts/edesky_*.py` with their four
  `edesky-*` workflows (a different project). *(done)*
- Dead code shipped in the package: `legacy/` (LSTM-era boiler, forecast, data handler), `fotovoltaics.py`
  (Solax cloud), `switch.py`, `main.py`. Only `tests/test_boiler_calculations.py` imports `legacy.boiler`; delete both together.
- `MANUAL.md` still describes the InfluxDB / NMAP / Google-token setup removed in 2.0.
- `src/smartboiler/tests/` holds one stray test file; everything else is in `tests/`.
- Known items from `ARCHITECTURE.md` §13: non-atomic pickle writes, no HA retry, no schema version.

Add-on repo (`smartboiler-add-on`):

- `model_*.weights.h5`, `scaler_*.pkl` — LSTM leftovers, unused since 2.0.
- `MANUAL.md` and the top of `README.md` describe the old setup; install badge points to `home-assistant/addons-example`.
- `config.yaml` says `1.5.10`, `CHANGELOG.md` says `2.0.0`.

---

## 2. The one idea that makes it simple

**Every controllable device is a storage with a comfort band.**

| Device | What is stored | Band (user comfort) | Efficiency | Demand forecast |
|--------|----------------|---------------------|------------|-----------------|
| Boiler | Heat in tank water | `min_tmp` … `set_tmp` | ~0.95 | Hot-water histogram (exists) |
| Heat pump – DHW | Heat in tank water | same | COP(T_out) ≈ 2–3.5 | same histogram |
| Heat pump – heating | Heat in the building | e.g. 20.5 … 22.5 °C | COP(T_out) ≈ 2.5–5 | Heat loss ∝ (T_in − T_out) |
| Air conditioning | "Cold" in the building | e.g. 23 … 26 °C | EER(T_out) | Heat gain ∝ (T_out − T_in) + sun |
| Battery | Electricity | min SoC … 100 % | ~0.9 round trip | — |
| EV (later) | Electricity | target SoC by departure | ~0.9 | Departure time |

So one planner answers one question for all of them:
*"When is energy cheapest to put into this storage so it never leaves its band?"*

That is exactly what `HeatingScheduler.plan()` already does for the boiler.
Two generalisations are enough:

1. **One price signal instead of special cases.** For each slot compute the *marginal cost of 1 kWh of electricity*:

   ```
   surplus expected in slot  →  cost = export price   (what we would have earned; may be 0 or negative)
   otherwise                 →  cost = import price   (spot or fixed + distribution + taxes, by low/high tariff)
   ```

   This replaces the `-1` PV hack and `battery_priority`: with a paid feed-in the boiler waits for cheap
   hours on its own; with zero/negative export price it soaks up surplus on its own.

2. **Divide by efficiency.** Cost of 1 kWh *of heat* = `price[h] / COP[h]`.
   The heat pump then prefers the warm afternoon (high COP) without any extra rule.

---

## 3. Target architecture

```
                 ┌───────────────── Energy context (shared, device-agnostic) ───────────────┐
 HA entities ──► │ Timeline 15 min × 36 h                                                    │
 Open-Meteo  ──► │  PV forecast · house base-load forecast · outdoor temp · import/export    │
 Spot API    ──► │  price · tariff windows (HDO)   →  surplus[t], cost[t]                    │
                 └──────────────┬──────────────────────────────────────┬─────────────────────┘
                                │ every 15 min                         │ every 10–30 s
                     ┌──────────▼──────────┐                 ┌─────────▼──────────┐
                     │ Planner             │    plan[t]      │ Surplus dispatcher │
                     │ devices in priority │ ──────────────► │ real grid export → │
                     │ order, each takes   │                 │ start/stop next    │
                     │ from the remainder  │                 │ device in priority │
                     └──────────┬──────────┘                 └─────────┬──────────┘
                                └───────────────┬──────────────────────┘
                                      ┌─────────▼─────────┐
                                      │ Safety guard      │  hard limits, always last word
                                      └─────────┬─────────┘
                        ┌───────────┬───────────┼────────────┬───────────┐
                     Boiler     HeatPump       AC         Battery      (EV)
                     relay      SG-Ready /   climate.*    read-only
                                climate.*                 at first
```

### 3.1 Three layers, three jobs

| Layer | Period | Job | Why separate |
|-------|--------|-----|--------------|
| **Planner** | 15 min, 36 h horizon | Cheapest way to stay in every comfort band | Needs forecasts, may be slow |
| **Surplus dispatcher** | 10–30 s | Forecast is always wrong in detail: use *measured* export now | Simple rule engine, no forecast |
| **Safety guard** | every command | Legionella, min/max temps, compressor min on/off time, max starts/h, main-breaker limit | Small, exhaustively tested, never bypassed |

Fail-safe rule: if the add-on or HA dies, every device falls back to **its own native control**
(boiler thermostat with relay ON, heat pump in "normal", AC to its last user setpoint).
The add-on only ever *nudges*; it never becomes the only thing keeping the house warm.

### 3.2 Device contract

One small interface; everything device-specific lives behind it.

```python
class FlexDevice(Protocol):
    id: str
    priority: int                       # order for planner and dispatcher

    def demand(self, tl: Timeline) -> list[float]: ...          # kWh useful energy needed per slot
    def plan(self, tl: Timeline, cost: list[float],
             surplus_kw: list[float]) -> list[float]: ...       # kW electrical per slot
    def can_absorb_now(self) -> tuple[float, float]: ...        # (min_kw, max_kw) for dispatcher
    def apply(self, power_kw: float, reason: str) -> None: ...  # actuate (through safety guard)
    def release(self) -> None: ...                              # hand back to native control
    def status(self) -> dict: ...                               # dashboard + "why" text
```

`reason` is carried to the log and the dashboard ("heating: cheapest slot before 07:00 shower",
"pre-cooling: 2.1 kW export"). Explainability is a feature, not a debug aid.

### 3.3 Planner: sequential greedy, not a solver

Devices are planned **one after another in priority order**; each sees the surplus left by the
previous ones. Each device runs the existing greedy-fill loop on its own storage model.

- Pro: ~100 lines per device, deterministic, every decision has a one-line reason, same test style as today.
- Con: not globally optimal (a joint LP as in EMHASS would squeeze a few more %).
- The planner sits behind one function (`plan_all(context, devices) -> plans`), so an LP can replace it later
  without touching devices. Not recommended before real data shows the gap matters.

### 3.4 How each device is actually controlled

| Device | Preferred actuator | Fallback | Never |
|--------|--------------------|----------|-------|
| Boiler | Relay on/off (today) | — | — |
| Heat pump | **SG-Ready** (2 relay contacts: block / normal / boost) | `climate` setpoint offset ±1–2 K; `water_heater` DHW setpoint | Cutting compressor power with a relay |
| AC | `climate.set_temperature` / `set_hvac_mode` within the band | on/off switch | Fighting a user who just changed it by hand (back off for N hours) |
| Battery | Read SoC and power only | later: "charge from grid now" switch if the inverter exposes one | Assuming control exists |

SG-Ready is the key for heat pumps: nearly every model has it, it is two cheap relays,
and the heat pump's own controller keeps all protections.

---

## 4. Setup: as short as possible, still complete

Principle: **ask what the user has, discover the rest, learn the numbers.**

1. **"What is in your house?"** — tiles: Boiler · Heat pump · AC · PV · Battery. Nothing else on the page.
2. **Discover.** Read HA's own Energy dashboard config (`energy/get_prefs` over the Supervisor websocket):
   grid import/export, PV production and battery sensors are already defined there by most PV owners.
   For the rest, pre-filter entity pickers by `device_class`, unit and domain
   (`climate.*`, `water_heater.*`, power sensors in W) and pre-select the best match.
3. **Per device: one entity + 2–3 numbers** with defaults (volume, rated power, comfort band).
   Everything else (standby loss, COP curve, house heat loss, time constant) starts from a default and is **learned**.
4. **Tariff** — pick one: `Spot` / `Fixed single` / `Fixed low+high (HDO)`; enter the fee per kWh and the export price.
   HDO windows are learned (exists) or typed.
5. **One slider "Savings ↔ Comfort"** — maps to prediction quantile and band width. Advanced fields are hidden behind "Expert".
6. **Finish in "Monitor" mode.** Every device has `Off / Monitor / Auto`. In Monitor the add-on plans and
   shows what it *would* do and what it *would* save, but touches nothing. The user flips each device to Auto when convinced.
   This also gives the learners their data and makes onboarding risk-free.

The dashboard then nags usefully: *"Add a tank temperature sensor → plan accuracy ±2 °C instead of ±8 °C"*.
Works with the minimum, improves with every sensor (the existing L1–L4 temperature fallback is the template).

Config file gets a `schema_version` and a `devices: [...]` list; the current flat boiler config migrates to
`devices[0]` automatically, so existing installations keep running.

### Back into Home Assistant

- **Sensors** (plan, next run, mode, today's saving, reason) via REST `POST /api/states` — zero dependencies.
- **Controls** (mode select, "Boost now" button, comfort slider) via **MQTT discovery** when the Mosquitto
  service is present; otherwise only in the ingress UI.

---

## 4a. User interface

Visual mock-up (three screens, sample data): <https://claude.ai/artifact/JeJAMGujpUksfVSv4e4aLu>
(private link — the owner must share it for others to open it).

The UI answers three questions, in this order: **what is happening now, why, and was it worth it.**

| Screen | Content |
|--------|---------|
| **Now** | One plain-language sentence at the top ("The sun is running the whole house…"). Three numbers: saved today, own solar used, price now. **Animated energy flow**: sources (solar, grid, battery) → house → devices; dashes move in the flow direction, faster = more power, grey = idle. One card per device: main value, comfort band bar, state, a one-line **"Why"**, `Off / Watch / Auto` switch and `Boost`. 24-hour plan strip: price, solar forecast, one row per device coloured by energy source, paused hours outlined. Two lists: "What it did and why", "Coming up". |
| **Statistics** | Day / Week / Month / Year. Saved vs. "no control" baseline, paid, self-consumption, self-sufficiency, comfort kept. Daily cost bars (baseline vs. paid). Per-device energy mix (solar / battery / grid). Learned hot-water habit heat map. "How well it knows your home": every learned parameter with its value and a confidence bar. |
| **Phone** | Same "Now" content in one column with a bottom tab bar; this is what HA's mobile app shows through ingress. |

Rules for the implementer:

- **Every state shown has a reason next to it.** The `reason` string from `FlexDevice.apply()` (§3.2) is the single source
  for the device card, the decision log and the HA sensor attribute.
- **Sentence first, numbers second.** The headline is generated from the same state (who supplies, who consumes,
  what changes next); it is the feature non-technical household members will read.
- **Animation only carries information**: flow direction and magnitude, "live" pulse, active-heating shimmer,
  bars growing on load. Everything is CSS-only and switched off under `prefers-reduced-motion`.
- **Colour = energy source**, consistently on every screen: amber solar, blue grid, teal battery; coral outline = paused/blocked.
  Never colour alone: each state also has a text label.
- **Tech**: keep Flask + server-rendered template + one small vanilla JS file polling `/api/state` every 5–10 s
  (or Server-Sent Events). No SPA framework, no build step. Charts are plain `div`/SVG bars as in the mock-up,
  so the existing Chart.js CDN dependency can go — the add-on then works without internet access.
- **Light and dark**: the mock-up is dark; define all colours as CSS variables and follow the HA theme
  (`prefers-color-scheme`) so a light variant is a second variable block.
- **Diagnostics move out of the way.** The current thermal-model, calibration and HDO tabs are valuable for debugging;
  keep them under `Settings → Diagnostics`, not on the main screen.
- **Language**: all UI strings through one dictionary (`en`, `cs` first).

---

## 5. Roadmap

Each phase is shippable alone and useful alone.

| Phase | Content | User-visible win |
|-------|---------|------------------|
| **0 Cleanup** | Remove stale files and docs, sync version, atomic writes, HA retry, `schema_version` | Trustworthy base |
| **1 Energy context** | 15-min timeline; import/export price model; PV forecast (Open-Meteo irradiance × kWp, self-calibrated against real production per hour-of-day → learns shading); base-load forecast (reuse `RollingHistogramPredictor` on house consumption); Energy-prefs discovery | Boiler plan finally correct for PV owners |
| **2 Surplus dispatcher** | Fast loop on measured grid export, hysteresis, min run time | Boiler follows real sun; biggest saving for FVE without spot tariff |
| **3 Device framework** | Extract `FlexDevice`, wrap boiler, split `controller.py`, `devices[]` config, Monitor/Auto, tiles wizard, HA sensors | Same behaviour, new foundation (refactor guarded by existing tests) |
| **4 Air conditioning** | `climate` device: pre-cool / pre-heat inside band on surplus or cheap slots; manual-override back-off | Second device type; easiest to prove the framework |
| **5 Heat pump** | SG-Ready + DHW; block in expensive slots with max block time by outdoor temp; then learned 1R1C house model and COP curve | Largest consumer in the house under control |
| **6 Reporting** | Cost vs "dumb" baseline, self-consumption %, forecast accuracy, notifications | Proof of value |
| **7 Later** | Battery grid-charging, EV charger, main-breaker / 15-min peak guard, joint LP, proportional PV diverter | — |

### Done so far

- **Phase 0**: dead code and unrelated files removed, atomic state writes, HA GET retries, multi-window
  HDO schedules accepted by validation. Open: `MANUAL.md` rewrite, `schema_version`.
- **Phase 1, part**: tariff model (`tariff.py`: spot / fixed / low+high, fees, export price) used by the scheduler;
  Open-Meteo PV forecast (`pv_forecast.py`) with a constant house base load. Open: 15-min slots,
  self-calibration of the PV forecast, learned base load, Energy-dashboard discovery.
- **Phase 2**: `surplus_dispatcher.py` runs the boiler on measured export with hysteresis (2 readings,
  start ≥ 80 % covered, stop < 50 %). Uses the existing 60 s control loop; a faster loop is open.

Phases 1–2 already pay off for boiler-only users. Do the refactor (3) **after** them only if
they can be added cleanly; otherwise 0 → 3 → 1 → 2. Implementer's call after reading `controller.py`.

---

## 6. Testing strategy

- **Pure core.** Planner, price model, forecasts and device models take data in and return data out — no HA, no clock.
  Scenario tests in the style of `tests/test_scheduler_scenarios.py`.
- **Day simulator.** Replay a recorded or synthetic day (PV, prices, draws, weather) through planner + device models,
  compare cost with a thermostat-only baseline. One tool, three uses: regression tests,
  the Monitor-mode "would have saved" number, and the reporting baseline.
- **Safety invariants** as property tests: for any input, no command violates band, min on/off time or legionella rule.
- **Fake HA client** for controller-level tests (exists in spirit in `tests/test_ha_client.py`).
- **Migration tests**: every old config shape loads into the new schema.
- Add-on repo: keep CI lint/build, add a smoke test that the image starts and serves `/`.

---

## 7. Risks

| Risk | Mitigation |
|------|------------|
| Heat-pump short cycling, warranty | SG-Ready only; min on/off and max starts/h in the safety guard |
| Comfort loss → user turns it off | Monitor mode first, narrow default bands, one-tap Boost, back off after manual change |
| Bad PV / load forecast | Dispatcher corrects in real time; planner re-plans every 15 min |
| Negative spot prices | Falls out of the price model (export cost negative → consume) |
| HP tariffs with HDO blocks (e.g. 20 h low tariff) | Reuse `HDOLearner` per device |
| Cloud-only AC with slow API | Per-device minimum command interval |
| Scope creep into "another EMHASS" | Stay thermal-first, zero-YAML, self-learning; no generic optimisation DSL |

---

## 8. Decisions needed from the maintainer

1. **Name and identity.** Keep the `smartboiler` name and add-on slug (seamless upgrade, misleading name) or publish a second add-on
   under a new name from the add-on repository and keep `smartboiler` frozen? Recommendation: second add-on, same library, shared code.
2. **Add-on vs. HACS integration.** Recommendation: stay an add-on (free choice of Python deps, own UI, existing code)
   and publish entities to HA as in §4.
3. **First real installation.** Which heat pump, inverter and AC brands are available for testing? That decides
   which actuators phase 4–5 implement first.
4. **Battery.** Observe only (recommended start) or control grid charging too?
5. **Order of phases 1–3** (see note under the roadmap).
