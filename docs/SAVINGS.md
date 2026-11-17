# Savings analysis — what would smart heating have saved?

`smartboiler.savings` replays your real history (hot-water use, PV production, prices)
through three heating strategies with the same boiler physics, and reports what each
would have cost:

| Strategy | Meaning |
|---|---|
| Thermostat (today) | Heats whenever the water drops 5 °C below the set temperature. |
| Smart, perfect forecast | The scheduler re-plans every hour knowing the real future use. Upper bound. |
| Smart, learned forecast | The scheduler re-plans every hour with the predictor trained only on the past. Realistic. |

It runs anywhere with Python and the package installed; no Home Assistant connection is needed.

## 1. Export your data from Home Assistant

1. Open **History**, pick the hot-water sensor (for example `sensor.total_water_usage`) and the
   time range — ideally a whole year, at least a few weeks.
2. Click the **download** icon (top right) to get a CSV.
3. Do the same for the PV energy sensor (for example `sensor.pv_energy_total`).

Both exports can also be one file with several entities; pick one with `--water-entity` / `--pv-entity`.
A plain CSV with two columns `time,value` works as well.

Copy `/data/smartboiler_setup.json` from the add-on (boiler size, tariff, base load, HDO
schedule) — or write a small JSON with just the keys you want to change; missing keys use defaults.

### …or fetch straight from Home Assistant

Create a long-lived access token (your HA profile → **Security** → *Long-lived access tokens*)
and keep it in an environment variable, never on the command line:

```bash
export HA_TOKEN=...   # paste the token here, in your own terminal
python -m smartboiler.savings --ha-url https://ha.example.com --list-sensors
python -m smartboiler.savings --ha-url https://ha.example.com --days 30 \
  --water-entity sensor.total_water_usage --pv-entity sensor.pv_energy_total \
  --config smartboiler_setup.json --currency Kč
```

Units are read from the sensors (L, m³, kWh, Wh, W, kW, L/min) and converted automatically.
The REST API only returns what the recorder still keeps — **10 days by default**
(`recorder: purge_keep_days`). For longer periods export CSVs or raise the retention.

## 2. Run it

```bash
python -m smartboiler.savings \
  --config smartboiler_setup.json \
  --water water.csv --water-kind counter --water-unit litres --hot-temp 50 \
  --pv pv.csv --pv-kind counter \
  --currency Kč --out report.md
```

What the input options mean:

| Option | Values |
|---|---|
| `--water-kind`, `--pv-kind` | `counter` cumulative total (L, kWh) · `power_w` power in W · `per_minute` rate like L/min · `per_hour` amounts per hour |
| `--water-unit` | `litres` (converted to heat with `--hot-temp` and the cold-water temperature) or `kwh` if you already measure heat |
| `--spot` | optional CSV `time,value` with hourly spot prices in EUR/MWh for spot tariffs; without it unknown hours use 75 EUR/MWh |
| `--timezone` | default `Europe/Prague`; decides which hours are low tariff, weekdays for the predictor |

A year of hourly data takes about half a minute.

## 3. Read the report

- **Electricity / from PV / from grid** — what the heater drew, split by source.
- **Standby loss** — heat lost through the tank jacket; lower when the water is kept cooler.
- **Cost** — grid kWh × import price + PV kWh × export price (feed-in income given up).
- **Hours below min** — comfort check; should be 0 or close to it.
- **Cost per month** — shows where the saving comes from (winter: prices and losses; summer: solar).

## Assumptions and limits

- Hot water is counted as heat at `--hot-temp`, so every strategy delivers the same showers.
- PV surplus = production − `house_base_load_w` (constant). If your house uses more at midday,
  the real surplus is smaller.
- Smart strategies see the real PV of the next 24 h (a good forecast plus the real-time
  surplus dispatcher come close).
- Hourly resolution: within-hour effects are averaged.
- A small tank limits solar use: 120 L between 40 and 60 °C stores only about 2.8 kWh.
  Try a higher `boiler_set_tmp` in the config to see what storing more heat would gain.
