"""Template metadata for the setup/settings wizard."""

from smartboiler.setup_config import DEFAULTS

# ── PV / Battery wizard sections ──────────────────────────────────────────────
# Describes the fields shown in the PV and Battery configuration cards.
# Consumed by the setup.html template to render field labels, hints, and
# the entity-selector dropdowns that populate from /api/ha/entities.

PV_SECTION = {
    "title": "Solar / PV (FVE)",
    "description": (
        "Configure your solar installation so the scheduler can heat the boiler "
        "for free when PV surplus is available."
    ),
    "toggle_key": "has_pv",
    "fields": [
        {
            "cfg_key": "pv_installed_power_kw",
            "label": "Installed peak power (kWp)",
            "type": "number",
            "hint": "Total installed PV capacity. Used to sanity-check forecast values.",
        },
        {
            "cfg_key": "pv_power_entity_id",
            "label": "Current PV power entity (W)",
            "type": "entity",
            "hint": (
                "Sensor reporting the current PV output in Watts. "
                "Used as a flat 24-hour proxy when no forecast entity is available."
            ),
        },
        {
            "cfg_key": "pv_forecast_entity_id",
            "label": "PV hourly forecast entity (kWh)",
            "type": "entity",
            "hint": (
                "Optional: a sensor that provides the expected PV production for the "
                "next hour in kWh (e.g. Solcast, Forecast.Solar). When set, this is "
                "preferred over the current-power sensor for scheduling."
            ),
        },
    ],
}

BATTERY_SECTION = {
    "title": "Battery storage",
    "description": (
        "If you have a home battery, configure it here so the scheduler knows how "
        "much PV surplus is actually available for the boiler."
    ),
    "toggle_key": "has_battery",
    "fields": [
        {
            "cfg_key": "battery_capacity_kwh",
            "label": "Usable battery capacity (kWh)",
            "type": "number",
            "hint": "Total usable energy storage in kWh (not peak / nameplate).",
        },
        {
            "cfg_key": "battery_soc_entity_id",
            "label": "State-of-charge (SoC) entity",
            "type": "entity",
            "hint": "Sensor reporting the current battery charge level.",
        },
        {
            "cfg_key": "battery_soc_unit",
            "label": "SoC unit",
            "type": "select",
            "options": [
                {"value": "percent", "label": "Percent (0–100 %)"},
                {"value": "kwh",     "label": "Kilowatt-hours (kWh)"},
            ],
            "hint": "Unit reported by the SoC sensor — either percent or kWh.",
        },
        {
            "cfg_key": "battery_max_charge_kw",
            "label": "Max charge rate (kW)",
            "type": "number",
            "hint": "Maximum charge power in kW. Leave 0 if unknown or unlimited.",
        },
        {
            "cfg_key": "battery_priority",
            "label": "PV allocation priority",
            "type": "select",
            "options": [
                {"value": "battery_first", "label": "Battery first → then boiler → then sell"},
                {"value": "boiler_first",  "label": "Boiler first → then battery → then sell"},
                {"value": "sell_first",    "label": "Sell everything (no free energy for boiler)"},
            ],
            "hint": (
                "Determines how available PV surplus is allocated. "
                "'Battery first' is the default: the battery is topped up before the "
                "boiler benefits from free solar. "
                "'Boiler first' is useful when the battery is already full most of the time."
            ),
        },
    ],
}

INFLUX_BOOTSTRAP_SECTIONS = [
    {
        "mode": "standard",
        "title": "Standard prediction bootstrap",
        "description": (
            "Flow-based import for the standard predictor. Use the relay and "
            "power entities plus flow, outlet-pipe temperature, and boiler-case "
            "temperature when available."
        ),
        "fields": [
            {
                "cfg_key": "influxdb_standard_relay_entity_id",
                "label": "Boiler relay entity",
                "hint": "State history of the boiler relay in InfluxDB.",
            },
            {
                "cfg_key": "influxdb_standard_power_entity_id",
                "label": "Power entity",
                "hint": "Used for better ON-time import and thermostat-trip detection.",
            },
            {
                "cfg_key": "influxdb_standard_flow_entity_id",
                "label": "Flow sensor entity",
                "hint": "Water flow in L/min for direct consumption import.",
            },
            {
                "cfg_key": "influxdb_standard_water_temp_entity_id",
                "label": "Hot-water outlet temperature entity",
                "hint": "Temperature in the hot-water output pipe.",
            },
            {
                "cfg_key": "influxdb_standard_case_tmp_entity_id",
                "label": "Boiler case temperature entity",
                "hint": "Seeds the thermal model from historical case cooling.",
            },
        ],
    },
    {
        "mode": "simple",
        "title": "Simple prediction bootstrap",
        "description": (
            "Flowless import for the simple predictor. Use the relay and power "
            "entities together with inlet-pipe, outlet-pipe, and boiler-case "
            "temperatures."
        ),
        "fields": [
            {
                "cfg_key": "influxdb_simple_relay_entity_id",
                "label": "Boiler relay entity",
                "hint": "State history of the boiler relay in InfluxDB.",
            },
            {
                "cfg_key": "influxdb_simple_power_entity_id",
                "label": "Power entity",
                "hint": "Required for meaningful simple-mode bootstrap estimates.",
            },
            {
                "cfg_key": "influxdb_simple_inlet_tmp_entity_id",
                "label": "Cold-water inlet temperature entity",
                "hint": "Temperature of the inlet pipe during historical import.",
            },
            {
                "cfg_key": "influxdb_simple_outlet_tmp_entity_id",
                "label": "Hot-water outlet temperature entity",
                "hint": "Used when available as the preferred hot-water signal.",
            },
            {
                "cfg_key": "influxdb_simple_case_tmp_entity_id",
                "label": "Boiler case temperature entity",
                "hint": "Fallback hot-water estimate and thermal-model seeding.",
            },
        ],
    },
]

INFLUX_BOOTSTRAP_FIELD_KEYS = [
    field["cfg_key"]
    for section in INFLUX_BOOTSTRAP_SECTIONS
    for field in section["fields"]
]


def get_setup_template_context(settings_mode: bool) -> dict:
    return {
        "settings_mode": settings_mode,
        "config_defaults": dict(DEFAULTS),
        "influx_bootstrap_sections": INFLUX_BOOTSTRAP_SECTIONS,
        "influx_bootstrap_field_keys": INFLUX_BOOTSTRAP_FIELD_KEYS,
        "pv_section": PV_SECTION,
        "battery_section": BATTERY_SECTION,
    }
