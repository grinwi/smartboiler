# SmartBoiler — Architecture & Design

This document describes the internal architecture, data flows, algorithms, and design decisions of SmartBoiler v2.0.
It also includes worked examples and a savings simulation so you can understand what the system does in practice.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Module Dependency Map](#2-module-dependency-map)
3. [Dual-Workflow Architecture](#3-dual-workflow-architecture)
4. [Prediction Pipeline](#4-prediction-pipeline)
5. [Scheduling Algorithm](#5-scheduling-algorithm)
6. [Temperature Estimation](#6-temperature-estimation)
7. [HDO Ripple-Control Learning](#7-hdo-ripple-control-learning)
8. [Legionella Protection](#8-legionella-protection)
9. [Storage Model](#9-storage-model)
10. [Dashboard & Web API](#10-dashboard--web-api)
11. [Worked Example — 24-Hour Schedule](#11-worked-example--24-hour-schedule)
12. [Savings Simulation](#12-savings-simulation)
13. [Known Limitations & Improvement Areas](#13-known-limitations--improvement-areas)

---

## 1. System Overview

SmartBoiler sits inside a Home Assistant add-on container.
It reads entity states and history from HA via the Supervisor REST API, and writes back only through `switch.turn_on` / `switch.turn_off` service calls.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Home Assistant                                    │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Shelly   │  │ ESPHome NTC  │  │ Flow + temp  │  │  PV inverter │  │
│  │  relay    │  │  case/water  │  │   sensors    │  │  integration │  │
│  └─────┬─────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│        │               │                 │                  │          │
│        └───────────────┴─────────────────┴──────────────────┘          │
│                             HA entity registry                          │
│                             HA history recorder                         │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │  Supervisor REST API
                                  │  (SUPERVISOR_TOKEN auth)
                   ┌──────────────▼───────────────────┐
                   │      SmartBoiler add-on           │
                   │                                   │
                   │   ┌─────────────────────────┐    │
                   │   │    HAClient wrapper      │    │
                   │   └────────────┬────────────┘    │
                   │                │                  │
                   │   ┌────────────▼────────────┐    │
                   │   │  ForecastWorkflow (1h)  │    │
                   │   │  ControlWorkflow (60s)  │    │
                   │   └────────────┬────────────┘    │
                   │                │                  │
                   │   ┌────────────▼────────────┐    │
                   │   │  Flask dashboard :8099  │    │
                   │   │  (HA ingress)           │    │
                   │   └─────────────────────────┘    │
                   │                                   │
                   │   /data/ (persistent volume)      │
                   │   ├── state.json                  │
                   │   ├── consumption_history.pkl     │
                   │   ├── predictor.pkl               │
                   │   ├── hdo_learner.pkl             │
                   │   └── smartboiler_setup.json      │
                   └───────────────────────────────────┘
```

**External data source:** `energy-charts.info` free API for day-ahead spot electricity prices (no account required, covers CZ/SK/AT/DE/PL/HU and more).

---

## 2. Module Dependency Map

```
controller.py (SmartBoilerController)
    ├── ha_client.py          (HAClient)
    ├── state_store.py        (StateStore)
    ├── ha_data_collector.py  (HADataCollector)
    │       └── ha_client.py
    ├── predictor.py          (RollingHistogramPredictor)
    ├── hdo_learner.py        (HDOLearner)
    ├── scheduler.py          (HeatingScheduler, BoilerParams, HourSlot)
    ├── spot_price.py         (SpotPriceFetcher)
    ├── temperature_estimator.py  (TemperatureEstimator)
    │       ├── ha_client.py
    │       └── thermal_model.py  (ThermalModel)
    ├── legionella_protector.py   (LegionellaProtector)
    │       └── state_store.py
    └── web_server.py         (Flask app)
            └── web_routes.py
                    └── web_setup.py  (setup wizard field definitions)
                    └── setup_config.py  (config persistence)
```

All modules are **pure-Python** with no ML framework dependencies. Only standard scientific stack: `numpy`, `pandas`, `flask`, `requests`, `pytz`.

---

## 3. Dual-Workflow Architecture

The controller runs two loops in separate threads:

```
Thread 1: ForecastWorkflow                    Thread 2: ControlWorkflow
─────────────────────────────────────────────────────────────────────────
t=0h   ┌──────────────────────────┐
       │ collect_and_update(6h)   │           every 60 s:
       │ predictor.update()       │           ┌──────────────────────────┐
       │ spot_price.fetch()       │           │ get_boiler_tmp()         │
       │ hdo_learner.next_24h()   │           │ check legionella_due?    │
       │ scheduler.plan()         │           │ execute heating_plan[h]  │
       │ state_store.save()       │           │ hdo_learner.observe()    │
       └──────────────────────────┘           │ thermal_model.observe()  │
                                              └──────────────────────────┘

t=1h   ┌──────────────────────────┐
       │ (repeats)                │
       └──────────────────────────┘
```

**Shared state:** `_shared_plan` (List[bool] of 24 hours) and `_shared_slots` are written by ForecastWorkflow and read by ControlWorkflow. Writes happen at most once per hour; Python's GIL makes list-swap effectively atomic for this use case.

**What happens if HA goes offline?**
- `HAClient` raises an exception; the workflow logs the error and retries next cycle.
- The last heating plan from `state_store` is used as fallback.
- The controller never leaves the relay stuck ON: ControlWorkflow turns OFF at the end of each planned hour.

**What happens if spot prices are unavailable?**
- `SpotPriceFetcher` returns an empty dict; slots get `spot_price=None`.
- `HourSlot.__post_init__` assigns `DEFAULT_MEDIUM_PRICE` (75 EUR/MWh), so the scheduler still works — it just can't optimise across price differences.
- Tomorrow's prices are typically published at ~14:30 CET; the ForecastWorkflow fetches them on the next hourly cycle after they appear.

---

## 4. Prediction Pipeline

```
HA entity history (last 6 h)
         │
         ▼
HADataCollector._fetch_hourly()
  ├── relay on-time  →  fraction of hour relay was ON
  ├── power sensor   →  average W → kWh
  └── flow + outlet temp  →  Q = ṁ × cₚ × ΔT  (most accurate)
         │
         ▼  DataFrame: index=datetime, columns=[consumed_kwh, relay_on, power_w]
         │
StateStore.save_consumption_history()   ──► /data/consumption_history.pkl
         │                                  (rolling 90-day window)
         │
         ▼
RollingHistogramPredictor.update(df)
  ├── cutoff = now − 12 weeks
  ├── filter: consumed_kwh > 0.005  (remove sensor noise)
  └── build histogram:
      _hist[(weekday, hour)] = [list of kWh values from matching slots]
         │
         ▼
RollingHistogramPredictor.predict_next_24h()
  for each of next 24 hours:
      key = (dt.weekday(), dt.hour)
      if len(_hist[key]) >= 4:
          prediction = np.quantile(_hist[key], self.quantile)  # p50/p75/p90
      else:
          prediction = global_fallback_quantile   # fallback until enough data
```

**Why histogram instead of LSTM?**

| Criterion | Rolling Histogram | LSTM |
|-----------|-------------------|------|
| Min data to start | 1-2 weeks (global fallback) | 3-6 months |
| Training time | < 1 ms | minutes (GPU) or hours (CPU) |
| Dependencies | numpy | TensorFlow / PyTorch |
| Accuracy vs routine | Excellent | Excellent |
| Accuracy vs non-routine | Falls back to base rate | Same |
| Memory footprint | < 1 MB | 50-500 MB |

For a typical single-family household, the LSTM offered no measurable accuracy improvement over p75 histogram when less than 6 months of data was available — and most installations never accumulate more than a few months before routines change anyway.

---

## 5. Scheduling Algorithm

### Inputs

| Input | Source |
|-------|--------|
| `current_tmp` | TemperatureEstimator |
| `consumption_forecast[24]` | RollingHistogramPredictor |
| `pv_forecast[24]` | PV entity / forecast entity (kWh) |
| `spot_prices{0..23}` | SpotPriceFetcher |
| `hdo_blocked[24]` | HDOLearner |
| `calendar_events` | CalendarManager |

### Slot Cost Assignment

Each hour slot is assigned an **effective cost** at construction time:

```
calendar_mode == vacation_off  →  cost = ∞           (never heat)
calendar_mode == boost_*       →  cost = -2.0         (always heat)
hdo_blocked                    →  cost = ∞           (cannot heat)
pv_surplus > 0.1 kWh           →  cost = -1.0         (free solar)
spot_price known               →  cost = spot_price   (market rate)
spot_price unknown             →  cost = 75 EUR/MWh   (medium guess)
```

### Greedy Fill Loop

```
heating_plan = [False] * 24

# Step 1: always enable forced/free slots
for slot in slots:
    if slot.effective_cost < 0:
        heating_plan[slot.index] = True

# Step 2: greedy fill until temperature constraint satisfied
repeat up to 24 times:
    trajectory = simulate(current_tmp, slots, heating_plan)
    violations = [i where trajectory[i] < min_tmp]
    if no violations: DONE

    first_viol = min(violations)
    candidates = [slots where index <= first_viol and not already heating and cost < ∞]

    cheapest = argmin over candidates of:
        (effective_cost / 1000 * wattage_kw) + standby_penalty * (first_viol - slot.index)

    heating_plan[cheapest.index] = True
```

The **standby-loss penalty** ensures that heating 5 hours early at 40 EUR/MWh is only preferred over heating 1 hour early at 70 EUR/MWh when the price saving (30 EUR/MWh × 2 kW × 1h = 0.06 EUR) actually exceeds the extra standby energy cost (e.g. 50 W × 4 h × 40/1000 = 0.008 EUR). In this case yes — but if early heating would last 20+ hours the standby penalty would flip the decision.

### Temperature Simulation

```python
tmp = current_tmp
for each slot:
    tmp -= standby_loss_kwh(tmp)  ÷ (m_kg × c_p / 3.6e6)   # cooling
    tmp -= consumption_kwh(slot)  ÷ (m_kg × c_p / 3.6e6)   # hot draw
    if heating_plan[slot]:
        tmp += (wattage_w × efficiency / 1000)               # heating
        tmp = min(tmp, set_tmp)                              # thermostat cap
    tmp = max(tmp, area_tmp)                                 # physical floor
```

### Calendar Event Modes

| `calendar_mode` | Effect |
|-----------------|--------|
| `vacation_off` | Slot cost = ∞; `min_tmp` lowered to vacation floor (e.g. 20 °C) |
| `vacation_min` | Normal cost; `min_tmp` lowered to vacation floor |
| `boost_max` | Slot cost = −2 (forced heat to `set_tmp`) |
| `boost_temp` | Slot cost = −2 (forced heat to `calendar_target_tmp`) |

Calendar events are sourced from a HA Calendar entity (via `calendar.*` entity or CalendarManager). Add events with keywords:
- `#off` → `vacation_off`
- `#min` → `vacation_min`
- `heat water at 65` → `boost_temp` (target=65 °C)
- `#boost` → `boost_max`

---

## 6. Temperature Estimation

The controller must know current water temperature to execute the heating plan correctly.
Because most boilers don't have an accessible internal NTC probe, four fallback levels are provided:

```
┌─────────────────────────────────────────────────────────────────┐
│ L1  boiler_direct_tmp_entity_id  (NTC inside boiler)           │
│     → use directly if available and numeric                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │ (entity unavailable or not configured)
┌──────────────────────────▼──────────────────────────────────────┐
│ L2  Power feedback                                              │
│     relay state = ON  AND  power_w < 50 W                      │
│     → internal boiler thermostat cut the heating element       │
│     → water must be at T_set (target temperature reached)      │
│     NOTE: relay is still energised — the internal bimetal/NTC  │
│     thermostat inside the boiler cuts the element, not the     │
│     smart relay. Power reading drops to ~0-5 W (relay idle).   │
└──────────────────────────┬──────────────────────────────────────┘
                           │ power > 50 W → actively heating
                           │  L3 is a cooling model — invalid during heating
                           │  → skip L3, fall directly to L4
                           │
                           │ relay OFF → no power inference possible
                           │  → continue to L3
┌──────────────────────────▼──────────────────────────────────────┐
│ L3  ThermalModel (Newton's law cooling)                         │
│     Valid only when relay is OFF (passive cool-down)            │
│     T(t) = T_amb + (T_0 - T_amb) × exp(−k × Δt)               │
│     k is fitted from calibration events (L2 transitions)       │
│     Calibration: relay ON + power ≈ 0  →  T_0 = T_set         │
│     Then observe T_case during passive cooling                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ (fewer than 8 calibration samples)
┌──────────────────────────▼──────────────────────────────────────┐
│ L4  Last known value                                            │
│     (logged as WARNING — use with caution)                      │
└─────────────────────────────────────────────────────────────────┘
```

**ThermalModel fitting details:**
- Requires ≥ 8 passive-cooling observations within a 7-day window
- Uses weighted log-linear regression: `ln[(T_case − T_amb) / (T_0 − T_amb)] = −k × Δt`
- Weights: `1/Δt` (prioritises early samples where the signal-to-noise is highest)
- Refits every 7 days; `k` is bounded to `[0.005, 3.0] h⁻¹` for physical plausibility

---

## 7. HDO Ripple-Control Learning

HDO (High-tariff Disconnect Order) is a system used by Czech/Slovak distributors to temporarily disconnect high-consumption loads (like boilers) during peak demand periods. The boiler relay becomes physically inoperable.

**Detection:** when the Shelly switch entity's state in HA is `unavailable` or `unknown`, the HDO relay has cut the circuit.

**Critical distinction:**
```
relay state = unavailable          →  HDO blocking (cannot heat)
relay state = on + power ≈ 0 W    →  Thermostat tripped (target temp reached; NOT HDO)
```

**Learning algorithm:**

```
For each 5-minute time slot (288 per day × 7 days = 2016 slots per week):

  observe(dt, relay_unavailable=True/False):
    slot_key = (weekday, hour * 12 + minute // 5)
    append (timestamp, relay_unavailable) to _observations[slot_key]
    prune observations older than 3 weeks

  is_blocked(weekday, hour):
    for each 5-min sub-slot of the hour:
      collect observations for that slot
      compute weeks_seen = count distinct ISO week numbers
      if weeks_seen < 2: not confident enough → not blocked
      compute weighted confidence:
        weight_i = exp(-(now - t_i) / half_life)   [half-life = 2 weeks]
        confidence = sum(weight_i if blocked_i) / sum(weight_i)
      if confidence >= 0.70: this 5-min slot is blocked
    return True if any 5-min sub-slot of the hour is blocked
```

**Explicit schedule override:** set `hdo_explicit_schedule = "22:00-06:00,12:05-13:35"` in the wizard. The explicit schedule takes priority over the learned one, and is the recommended approach when you know your distributor's schedule.

### Weekly HDO Schedule Visualisation (example)

```
         Mon  Tue  Wed  Thu  Fri  Sat  Sun
00:00  ██████████████████████████████████  ← blocked all nights 22-06
01:00  ██████████████████████████████████
02:00  ██████████████████████████████████
03:00  ██████████████████████████████████
04:00  ██████████████████████████████████
05:00  ██████████████████████████████████
06:00  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
...
12:05  ██████████████████████████████████  ← blocked 12:05-13:35 (midday)
13:35  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
...
22:00  ██████████████████████████████████

(█ = HDO blocked,  ░ = available)
```

> Note: being HDO-blocked is not necessarily bad — the low-tariff electricity rate during HDO-available hours can be significantly cheaper (distributor-dependent). SmartBoiler schedules heating *around* HDO blocks by filling cheap available hours just before predicted consumption.

---

## 8. Legionella Protection

*Legionella pneumophila* proliferates at water temperatures between 20–50 °C. Regular heating to ≥ 60 °C kills it.

SmartBoiler enforces a **65 °C full heat cycle every 21 days** regardless of the normal heating plan:

```
LegionellaProtector.check_and_act(boiler_tmp):

  if days_since_last_legionella_cycle >= 21:
    set_legionella_mode(True)
    heat until boiler_tmp >= 65 °C
    record timestamp in state_store
    set_legionella_mode(False)
```

- The cycle runs to completion before the normal schedule resumes.
- The last protection timestamp is persisted in `/data/state.json` and survives restarts.
- The cycle is logged and visible on the dashboard as "Legionella protection active".

---

## 9. Storage Model

All state is persisted to `/data/` — the HA add-on persistent volume that survives container restarts and updates.

```
/data/
├── smartboiler_setup.json    Configuration (from setup wizard)
├── state.json                Controller runtime state
│     ├── last_legionella_heating: "2026-02-28T03:15:00"
│     ├── heating_plan: [false, true, true, ...]
│     ├── heating_plan_dt: "2026-03-20T14:00:00"
│     └── spot_cache: { "2026-03-20": {0: 45.2, 1: 42.1, ...} }
├── consumption_history.pkl   DataFrame (90-day rolling window)
│     Columns: consumed_kwh, relay_on, power_w
│     Index:   DatetimeIndex (hourly)
├── predictor.pkl             RollingHistogramPredictor object
│     _hist: Dict[(weekday, hour) → List[float]]
└── hdo_learner.pkl           HDOLearner object
      _observations: Dict[slot_key → List[(timestamp, bool)]]
```

**Persistence guarantees:**
- JSON files: written atomically via Python `json.dump` (single write)
- Pickle files: written in-place; a crash during write can corrupt the file. The loader catches `Exception` and starts fresh (losing accumulated history but not crashing the controller)

---

## 10. Dashboard & Web API

The Flask dashboard runs on port 8099, proxied through HA ingress (authentication handled by HA).

**Routes:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Main dashboard (redirects to `/setup` if not configured) |
| GET | `/setup` | Setup wizard (multi-step entity picker) |
| GET | `/settings` | Advanced settings |
| GET | `/api/state` | JSON: current controller state (temp, plan, predictor stats) |
| GET | `/api/ha/entities` | JSON: all HA entity IDs (used by wizard pickers) |
| GET | `/api/plan` | JSON: today's 24-hour heating plan with slot details |
| GET | `/api/histogram` | JSON: per-(weekday,hour) consumption histogram summary |
| POST | `/api/setup/save` | Save wizard configuration |

**State provider contract:**

The controller injects a callable via `set_state_provider(fn)`. The dashboard calls `fn()` to get:

```json
{
  "boiler_tmp": 52.3,
  "relay_on": false,
  "legionella_active": false,
  "plan_generated_at": "2026-03-20T14:00:00",
  "heating_hours_today": ["03:00", "04:00", "12:00"],
  "predictor_ready": true,
  "hdo_schedule": { "Mon": [0, 1, 2, 3, 4, 5, 22, 23], ... }
}
```

**Rate limiting:** 120 requests per IP per 60-second window (in-memory token bucket). Resets on container restart — not a security control, just DoS mitigation.

---

## 11. Worked Example — 24-Hour Schedule

**Setup:** 120 L boiler, 2 kW heater, Czech Republic, Tuesday in January.
**Tariff:** day-ahead spot, standard single-tariff grid connection (no HDO).
**PV:** 5 kWp system, winter day — minimal surplus (mostly zero).
**Predicted consumption:** 0.28 kWh around 07:00, 0.35 kWh around 19:00.
**Current temperature at 00:00:** 38 °C (below set point of 60 °C).

### Spot prices (CZ, example winter day)

| Hour | EUR/MWh | Category |
|------|---------|----------|
| 00:00 | 45 | cheap |
| 01:00 | 42 | **cheapest** |
| 02:00 | 38 | **cheapest** |
| 03:00 | 40 | cheap |
| 04:00 | 48 | cheap |
| 05:00 | 55 | cheap |
| 06:00 | 95 | medium |
| 07:00 | 148 | **expensive** |
| 08:00 | 165 | **expensive** |
| 09:00 | 155 | **expensive** |
| 10:00 | 110 | medium |
| 11:00 | 88 | medium |
| 12:00 | 75 | medium |
| 13:00 | 68 | medium |
| 14:00 | 72 | medium |
| 15:00 | 80 | medium |
| 16:00 | 105 | medium |
| 17:00 | 145 | **expensive** |
| 18:00 | 168 | **expensive** |
| 19:00 | 152 | **expensive** |
| 20:00 | 118 | medium |
| 21:00 | 88 | medium |
| 22:00 | 65 | cheap |
| 23:00 | 50 | cheap |

### Scheduler output

The boiler needs ~2 kWh to heat from 38 °C to 60 °C:

```
Q = m × c_p × ΔT = 120 kg × 4.186 kJ/kg·K × 22 K = 11,051 kJ = 3.07 kWh
With 90 % efficiency → 3.07 / 0.9 ≈ 3.4 kWh input = ~1.7 hours of heating
```

Scheduler selects the two cheapest hours before first predicted consumption (07:00):

| Hour | Heating | Temp after | Cost/h | Reason |
|------|---------|-----------|--------|--------|
| 00:00 | ✗ | 36.8 °C | — | 45 EUR/MWh, standby penalty makes later cheaper |
| 01:00 | ✗ | 35.7 °C | — | |
| 02:00 | ✅ | 51.4 °C | 0.076 € | Cheapest hour (38 EUR/MWh) |
| 03:00 | ✅ | 60.0 °C | 0.080 € | 2nd cheapest (40 EUR/MWh); reaches set_tmp |
| 04:00 | ✗ | 59.4 °C | — | Already at set_tmp; further heating wasteful |
| 05:00 | ✗ | 58.8 °C | — | |
| 06:00 | ✗ | 58.2 °C | — | |
| 07:00 | ✗ | 54.8 °C | — | 0.28 kWh drawn (morning shower) |
| 08:00–11:00 | ✗ | ~52 °C | — | Slow cooling |
| 12:00–18:00 | ✗ | ~48 °C | — | Still above min_tmp |
| 19:00 | ✗ | 43.3 °C | — | 0.35 kWh drawn (evening shower/dishes) |
| 20:00 | ✗ | 41.8 °C | — | |
| 21:00 | ✗ | 40.3 °C | — | |
| 22:00 | ✅ | 55.3 °C | 0.130 € | Evening reheat at cheap rate |
| 23:00 | ✅ | 60.0 °C | 0.100 € | Back to set_tmp |

**Total heating cost for this day:** 0.076 + 0.080 + 0.130 + 0.100 = **0.386 €**

### Without optimisation (thermostat-only)

A conventional thermostat would run whenever temperature drops below the cut-in temperature (~55 °C). It would trigger roughly:
- 07:00 (after consumption drop) → heats at **148 EUR/MWh** → 0.296 €
- 19:00 (after evening draw) → heats at **152 EUR/MWh** → 0.304 €

**Total heating cost (thermostat-only):** ~**0.60 €**

**Single-day saving: ~0.21 € (35 %)**

Temperature trajectory comparison:

```
°C
60 ┤                                  ●●●                         ●●●
58 ┤        ●●●●●●●●●●●●●            ·  ·                       ·  ·
56 ┤       ·             ·           ·    ·                     ·
54 ┤      ·               ·    ●●   ·     ·                    ·
52 ┤     ·                 ·  ·  ●●·       ·                  ·
50 ┤    ·                   ●●                ·              ·
48 ┤   ·                                       ·            ·
46 ┤  ·                                         ·          ·
44 ┤ ·                                           ·        ·
42 ┤·                                             ·      ·
40 ┤                                               ·    ·
38 ┤──────────────────────────────────────────────────●─────────  ← SmartBoiler
   │
37 ┤── min_tmp constraint ─────────────────────────────────────
   │
   0  2  4  6  8 10 12 14 16 18 20 22 24   (hour)

   ● = SmartBoiler,   · = thermostat-only (schematic)
```

The key insight: SmartBoiler **pre-heats before expensive peak hours** so the boiler coasts through 07:00–19:00 on stored heat with no relay activation.

---

## 12. Savings Simulation

### Scenario A — Spot-price optimisation only (no PV)

Assumptions:
- 120 L boiler, 2 kW, 90 % efficiency
- Czech household, 2 adults
- ~2.5 kWh/day input energy for hot water
- Average spot price: 80 EUR/MWh; peak hours: 130 EUR/MWh; valley hours: 45 EUR/MWh
- Thermostat fires in peak hours 40 % of the time (morning/evening routine)

| Mode | Avg effective price | Annual energy | Annual cost |
|------|---------------------|---------------|-------------|
| Thermostat only | ~100 EUR/MWh | 912 kWh | **91 €** |
| SmartBoiler (spot) | ~52 EUR/MWh | 912 kWh | **47 €** |
| **Saving** | | | **~44 € / year** |

### Scenario B — With 5 kWp PV system

PV surplus is modelled as 0.5–1.5 kWh/day free energy for the boiler (seasonal average).
This displaces ~250 kWh/year of paid heating:

| Mode | Annual paid kWh | Annual cost |
|------|-----------------|-------------|
| Thermostat only | 912 kWh | **91 €** |
| SmartBoiler (spot) | 912 kWh | **47 €** |
| SmartBoiler (spot + PV) | ~662 kWh | **34 €** |
| **Total saving vs thermostat** | | **~57 € / year** |

### Scenario C — HDO low-tariff tariff (ČEZ D45d or similar)

HDO tariffs in Czechia offer a significantly discounted rate during HDO-available windows (off-peak hours). SmartBoiler schedules heating *only* during available windows and *around* blocks:

- HDO-available rate: ~2.50 CZK/kWh (~0.10 €/kWh)
- Single-tariff peak rate: ~5.80 CZK/kWh (~0.23 €/kWh)

| Mode | Annual cost (CZK) |
|------|-------------------|
| Thermostat (single tariff) | ~2,100 CZK |
| SmartBoiler + HDO D45d | ~575 CZK |
| **Saving** | **~1,525 CZK / year (~60 €)** |

> These are illustrative calculations. Actual savings depend on household size, insulation quality of the boiler, grid tariff structure, and PV system size. The dashboard shows real-time estimated cost and actual heating hours for monitoring.

---

## 13. Known Limitations & Improvement Areas

### Algorithm

| Area | Current | Improvement |
|------|---------|-------------|
| Prediction seasonality | 12-week rolling window treats all seasons equally | Split into winter/summer histogram or use 52-week window with seasonal weighting |
| Scheduler optimality | Greedy (good but not global optimal) | ILP / dynamic programming for true global minimum cost |
| Scheduler hysteresis | Plan can flip hour-by-hour on volatile prices | Add minimum heating-block length constraint |
| Thermal model ambient | Uses current T_amb for all historical calibration samples | Store T_amb at calibration time |

### Reliability

| Area | Current | Improvement |
|------|---------|-------------|
| HA API retries | No retry on 429/504 | Exponential backoff with jitter |
| Pickle corruption | Silent — starts fresh | Atomic write via temp file + rename |
| Schema versioning | None | Add version field to `state.json` + migration code |
| Sensor validation | No range checks | Reject readings outside physical bounds (e.g., T > 100 °C) |

### Observability

| Area | Current | Improvement |
|------|---------|-------------|
| Prediction accuracy | Not tracked | Record predicted vs actual consumption per slot |
| Cost tracking | Not persisted | Daily cost log (planned vs unoptimised baseline) |
| Thermal model diagnostics | Available but not exposed | Add `/api/thermal_model` dashboard endpoint |
| Metrics | None | Prometheus `/metrics` endpoint |
