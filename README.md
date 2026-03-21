# SmartBoiler

**Intelligent water boiler controller for Home Assistant.**
Predicts household hot water consumption patterns and schedules heating at the lowest-cost hours — using day-ahead spot electricity prices, photovoltaic surplus, and HDO ripple-control awareness.

> Originally developed as a master's thesis at FIT VUT Brno (2024).
> v2.0 is a complete rewrite: no LSTM, no InfluxDB, no external APIs required beyond HA itself.

[![GitHub](https://img.shields.io/badge/add--on-smartboiler--add--on-blue)](https://github.com/grinwi/smartboiler-add-on)
![Supports aarch64](https://img.shields.io/badge/aarch64-yes-green.svg)
![Supports amd64](https://img.shields.io/badge/amd64-yes-green.svg)

---

## How it works

SmartBoiler runs two parallel workflows inside a Home Assistant add-on:

```
┌──────────────────────────────────────────────────────┐
│  ForecastWorkflow  (runs every hour)                 │
│                                                      │
│  HA entity history ──► Rolling histogram predictor   │
│  Spot price API    ──► Greedy scheduler              │
│  HDO learner       ──► 24-hour heating plan          │
│  PV surplus        ──►  (persisted to /data/)        │
└──────────────────────────────────────────────────────┘
         │  heating_plan[24]
         ▼
┌──────────────────────────────────────────────────────┐
│  ControlWorkflow   (runs every 60 s)                 │
│                                                      │
│  Read boiler temperature  ◄── 4-level estimator      │
│  Check legionella due?    ◄── every 21 days          │
│  Execute plan hour        ──► turn relay ON / OFF    │
│  Observe HDO patterns     ──► update HDO learner     │
└──────────────────────────────────────────────────────┘
```

### Prediction — Rolling Histogram

Instead of an LSTM network, SmartBoiler uses a **per-(weekday, hour) quantile histogram** over the last 12 weeks of actual consumption data pulled from HA entity history. This approach:

- Works from the very first weeks of data (global quantile fallback until enough per-slot samples accumulate)
- Requires zero GPU, zero model training time
- Adapts naturally to changing household routines (old data ages out of the 12-week window)
- Uses p50/p75/p90 quantile — configurable via `prediction_conservatism` setting

### Scheduling — Greedy Cost Minimisation

The scheduler assigns 24 hour-slots an effective cost and fills from cheapest until the minimum water temperature constraint is always satisfied:

| Priority | Condition | Effective cost |
|----------|-----------|---------------|
| 1 (highest) | Calendar boost event | −2.0 (forced) |
| 2 | PV surplus > 0.1 kWh | −1.0 (free solar) |
| 3 | Known day-ahead spot price | spot EUR/MWh |
| 4 | No price data | 75 EUR/MWh (medium) |
| ✗ blocked | HDO relay cut / vacation_off | ∞ (never) |

A **standby-loss penalty** adjusts the cost of heating early: a slot cheaper by 5 EUR/MWh is only preferred over a later one if that price advantage exceeds the extra energy lost to standing-by while waiting.

### Temperature Estimation — 4-Level Fallback

```
L1  Direct NTC probe inside boiler  (most accurate)
 │  (if entity unavailable)
L2  Power feedback: relay ON + power < 50 W → T = T_set
 │  (relay energised but internal boiler thermostat cut the element — target reached)
 │  (if power > 50 W: actively heating; no inference possible from this level)
L3  Thermal model: Newton's law cooling from case sensor
 │  (if no calibration data yet)
L4  Last known value  (stale — logged as warning)
```

### HDO Ripple-Control Learning

Czech/Slovak low-tariff electricity uses an HDO relay that physically disconnects the boiler switch.
SmartBoiler detects this automatically: when the switch entity enters the `unavailable` state in HA it records a blocking observation for that 5-minute slot. After data from ≥ 2 distinct ISO calendar weeks is accumulated, the slot is considered reliably blocked and excluded from scheduling.

An explicit schedule can also be set in configuration (e.g. `"22:00-06:00,12:05-13:35"`).

---

## Requirements

### Minimum (no sensors)

| Component | Purpose |
|-----------|---------|
| Home Assistant | Data source + service calls |
| Shelly Plus 1PM (or any HA switchable relay) | Boiler power control |

Without sensors the system can still control heating based on time-of-use tariffs and HDO schedule — just without consumption-aware optimisation.

### Recommended

| Component | Purpose |
|-----------|---------|
| NTC thermistor inside boiler casing | Direct water temperature reading (L1 estimator) |
| Power monitoring relay (e.g. Shelly 1PM) | Temperature feedback + HDO detection |
| ESPHome flow meter + outlet thermistor | Accurate consumption measurement |
| PV inverter with HA integration | Solar surplus scheduling |

See [DOCS.md](../smartboiler-add-on/smartboiler/DOCS.md) for hardware wiring and ESPHome configuration.

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for:

- Full module dependency diagram
- Dual-workflow timing
- Prediction pipeline detail
- Scheduling algorithm walkthrough
- Savings simulation with real example

---

## Installation

Install via the Home Assistant add-on store:

1. Add repository `https://github.com/grinwi/smartboiler-add-on` to HA Add-on Store
2. Install **Smart Boiler**
3. Start the add-on; open the web UI via **Open Web UI**
4. Complete the setup wizard (entity pickers, boiler parameters, optional features)
5. The controller starts automatically once setup is saved

No manual YAML editing required. All configuration is persisted in `/data/smartboiler_setup.json` inside the add-on container.

---

## Repository structure

```
smartboiler/
├── src/smartboiler/
│   ├── controller.py          # Main orchestrator (ForecastWorkflow + ControlWorkflow)
│   ├── ha_client.py           # Home Assistant REST API wrapper
│   ├── state_store.py         # Persistent JSON + pickle storage (/data/)
│   ├── predictor.py           # Rolling histogram predictor
│   ├── hdo_learner.py         # HDO ripple-control pattern learner
│   ├── scheduler.py           # Greedy day-ahead heating scheduler
│   ├── spot_price.py          # energy-charts.info day-ahead price fetcher
│   ├── ha_data_collector.py   # Hourly consumption data from HA history
│   ├── temperature_estimator.py  # 4-level water temp estimation
│   ├── thermal_model.py       # Newton's law cooling model (L3 estimator)
│   ├── legionella_protector.py   # Periodic 65 °C safety cycle
│   ├── web_server.py          # Flask dashboard (HA ingress, port 8099)
│   ├── web_routes.py          # Route handlers
│   ├── web_setup.py           # Setup wizard field metadata
│   └── setup_config.py        # Configuration persistence
├── tests/                     # Unit + integration tests
├── README.md                  # This file
├── ARCHITECTURE.md            # Deep-dive with diagrams
└── setup.py
```

---

## Development

```bash
pip install -e smartboiler/
python -m pytest smartboiler/tests/
```

Entry point (matches the add-on run script):

```bash
python -m smartboiler.controller
```
