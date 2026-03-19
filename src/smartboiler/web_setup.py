"""Template metadata for the setup/settings wizard."""

from smartboiler.setup_config import DEFAULTS

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
    }
