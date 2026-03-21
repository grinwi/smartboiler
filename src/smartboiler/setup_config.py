"""
Persistent setup configuration for SmartBoiler.

Stored in /data/smartboiler_setup.json — written by the web UI wizard,
read by the controller. Replaces the brittle HA config.yaml schema for
all entity/integration options.
"""

import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

_DATA_DIR = os.environ.get("DATA_PATH", "/data/")
_SETUP_PATH = os.path.join(_DATA_DIR, "smartboiler_setup.json")

DEFAULTS: dict = {
    # ── Mode ──────────────────────────────────────────────────────────────
    "operation_mode": "standard",

    # ── Required entities ─────────────────────────────────────────────────
    "boiler_switch_entity_id": "",

    # ── Optional entities ─────────────────────────────────────────────────
    "boiler_power_entity_id": "",
    "boiler_direct_tmp_entity_id": "",
    "boiler_case_tmp_entity_id": "",
    "boiler_inlet_tmp_entity_id": "",
    "boiler_outlet_tmp_entity_id": "",
    "boiler_area_tmp_entity_id": "",
    "boiler_water_flow_entity_id": "",
    "boiler_water_temp_entity_id": "",
    "pv_surplus_entity_id": "",
    "person_entity_ids": "",
    "calendar_entity_id": "",

    # ── Boiler physics ────────────────────────────────────────────────────
    "boiler_volume": 120,
    "boiler_set_tmp": 60,
    "boiler_min_operation_tmp": 37,
    "boiler_watt_power": 2000,
    "average_boiler_surroundings_temp": 20,

    # ── Simple mode physics ───────────────────────────────────────────────
    "cold_water_temp": 10,
    "thermal_coupling_ratio": 0.45,
    "boiler_standby_watts": 50,
    "draw_detection_threshold_c": 2.0,
    "thermal_model_window_days": 7,
    "thermal_mass_ratio": 0.3,

    # ── PV / Solar (FVE) ──────────────────────────────────────────────────
    "has_pv": False,
    "pv_installed_power_kw": 0.0,       # peak installed capacity (kWp)
    "pv_power_entity_id": "",           # current PV production sensor (W or kW)
    "pv_forecast_entity_id": "",        # hourly forecast entity (kWh — e.g. Solcast)

    # ── Battery ───────────────────────────────────────────────────────────
    "has_battery": False,
    "battery_capacity_kwh": 0.0,        # usable capacity in kWh
    "battery_soc_entity_id": "",        # state-of-charge sensor
    "battery_soc_unit": "percent",      # "percent" | "kwh"
    "battery_max_charge_kw": 0.0,       # max charge rate (kW); 0 = unlimited
    # Allocation priority when PV is available:
    #   battery_first — fill battery, then boiler, then sell
    #   boiler_first  — heat boiler, then battery, then sell
    #   sell_first    — sell everything (no free energy for boiler or battery)
    "battery_priority": "battery_first",

    # ── Spot price ────────────────────────────────────────────────────────
    "has_spot_price": False,
    "spot_price_region": "CZ",

    # ── HDO ───────────────────────────────────────────────────────────────
    "hdo_explicit_schedule": "",
    "hdo_history_weeks": 3,
    "hdo_decay_weeks": 2,
    "hdo_retrain_weeks": 3,

    # ── Prediction ────────────────────────────────────────────────────────
    "prediction_conservatism": "medium",
    "min_training_days": 30,
    "predictor_retrain_weeks": 4,

    # ── InfluxDB bootstrap (optional) ─────────────────────────────────────
    "influxdb_host": "",
    "influxdb_port": 8086,
    "influxdb_db": "homeassistant",
    "influxdb_username": "",
    "influxdb_password": "",
    "influxdb_relay_entity_id": "",
    "influxdb_flow_entity_id": "",
    "influxdb_water_temp_entity_id": "",
    "influxdb_case_tmp_entity_id": "",
    "influxdb_inlet_tmp_entity_id": "",
    "influxdb_power_entity_id": "",
    "influxdb_standard_relay_entity_id": "",
    "influxdb_standard_flow_entity_id": "",
    "influxdb_standard_water_temp_entity_id": "",
    "influxdb_standard_case_tmp_entity_id": "",
    "influxdb_standard_power_entity_id": "",
    "influxdb_simple_relay_entity_id": "",
    "influxdb_simple_case_tmp_entity_id": "",
    "influxdb_simple_inlet_tmp_entity_id": "",
    "influxdb_simple_outlet_tmp_entity_id": "",
    "influxdb_simple_power_entity_id": "",
    "influxdb_measurement_temp": "°C",
    "influxdb_measurement_flow": "L/min",
    "influxdb_measurement_power": "W",
    "influxdb_measurement_state": "state",
    "influxdb_history_years": 2,
    "influxdb_start_date": "",

    # ── Calendar ──────────────────────────────────────────────────────────
    "vacation_mode": "min_temp",
    "vacation_min_temp": 30,

    # ── System ────────────────────────────────────────────────────────────
    "logging_level": "INFO",
}

_HDO_PATTERN = re.compile(r"^\d{1,2}:\d{2}-\d{1,2}:\d{2}$")

_MODE_SPECIFIC_INFLUX_FALLBACKS = {
    "influxdb_standard_relay_entity_id": ("influxdb_relay_entity_id",),
    "influxdb_standard_flow_entity_id": ("influxdb_flow_entity_id",),
    "influxdb_standard_water_temp_entity_id": ("influxdb_water_temp_entity_id",),
    "influxdb_standard_case_tmp_entity_id": ("influxdb_case_tmp_entity_id",),
    "influxdb_standard_power_entity_id": ("influxdb_power_entity_id",),
    "influxdb_simple_relay_entity_id": ("influxdb_relay_entity_id",),
    "influxdb_simple_case_tmp_entity_id": ("influxdb_case_tmp_entity_id",),
    "influxdb_simple_inlet_tmp_entity_id": ("influxdb_inlet_tmp_entity_id",),
    "influxdb_simple_outlet_tmp_entity_id": ("influxdb_water_temp_entity_id",),
    "influxdb_simple_power_entity_id": ("influxdb_power_entity_id",),
}

_ACTIVE_MODE_INFLUX_MIRRORS = {
    "standard": {
        "influxdb_relay_entity_id": "influxdb_standard_relay_entity_id",
        "influxdb_flow_entity_id": "influxdb_standard_flow_entity_id",
        "influxdb_water_temp_entity_id": "influxdb_standard_water_temp_entity_id",
        "influxdb_case_tmp_entity_id": "influxdb_standard_case_tmp_entity_id",
        "influxdb_inlet_tmp_entity_id": None,
        "influxdb_power_entity_id": "influxdb_standard_power_entity_id",
    },
    "simple": {
        "influxdb_relay_entity_id": "influxdb_simple_relay_entity_id",
        "influxdb_flow_entity_id": None,
        "influxdb_water_temp_entity_id": "influxdb_simple_outlet_tmp_entity_id",
        "influxdb_case_tmp_entity_id": "influxdb_simple_case_tmp_entity_id",
        "influxdb_inlet_tmp_entity_id": "influxdb_simple_inlet_tmp_entity_id",
        "influxdb_power_entity_id": "influxdb_simple_power_entity_id",
    },
}


def _populate_mode_specific_influx_fields(config: dict) -> None:
    for new_key, legacy_keys in _MODE_SPECIFIC_INFLUX_FALLBACKS.items():
        if str(config.get(new_key, "")).strip():
            continue
        for legacy_key in legacy_keys:
            legacy_val = str(config.get(legacy_key, "")).strip()
            if legacy_val:
                config[new_key] = legacy_val
                break


def _sync_legacy_influx_fields(config: dict) -> None:
    mirrors = _ACTIVE_MODE_INFLUX_MIRRORS.get(
        config.get("operation_mode", "standard"),
        _ACTIVE_MODE_INFLUX_MIRRORS["standard"],
    )
    for legacy_key, source_key in mirrors.items():
        if not source_key:
            config[legacy_key] = ""
            continue
        config[legacy_key] = str(config.get(source_key, "")).strip()


def _normalize_influx_fields(config: dict) -> None:
    _populate_mode_specific_influx_fields(config)
    _sync_legacy_influx_fields(config)


def validate_config(config: dict) -> list[str]:
    """Return a list of human-readable validation error strings.

    An empty list means the config is valid.
    """
    errors: list[str] = []

    # ── Required ────────────────────────────────────────────────────────
    if not str(config.get("boiler_switch_entity_id", "")).strip():
        errors.append("boiler_switch_entity_id is required")

    # ── Enums ───────────────────────────────────────────────────────────
    if config.get("operation_mode") not in ("simple", "standard"):
        errors.append("operation_mode must be 'simple' or 'standard'")

    if config.get("spot_price_region") not in ("CZ", "SK", "AT", "DE", "PL", "HU", "FR", "IT", "ES"):
        errors.append("spot_price_region must be one of: CZ, SK, AT, DE, PL, HU, FR, IT, ES")

    if config.get("prediction_conservatism") not in ("low", "medium", "high"):
        errors.append("prediction_conservatism must be 'low', 'medium', or 'high'")

    if config.get("vacation_mode") not in ("min_temp", "off"):
        errors.append("vacation_mode must be 'min_temp' or 'off'")

    # ── PV / Battery enums (only when enabled) ───────────────────────────
    if config.get("has_battery"):
        if config.get("battery_soc_unit") not in ("percent", "kwh"):
            errors.append("battery_soc_unit must be 'percent' or 'kwh'")
        if config.get("battery_priority") not in ("battery_first", "boiler_first", "sell_first"):
            errors.append("battery_priority must be 'battery_first', 'boiler_first', or 'sell_first'")

    if config.get("logging_level") not in ("DEBUG", "INFO", "WARNING", "ERROR"):
        errors.append("logging_level must be one of: DEBUG, INFO, WARNING, ERROR")

    # ── Numeric ranges ──────────────────────────────────────────────────
    _check_range(errors, config, "boiler_volume", 10, 1000, "litres")
    _check_range(errors, config, "boiler_set_tmp", 30, 95, "°C")
    _check_range(errors, config, "boiler_min_operation_tmp", 20, 90, "°C")
    _check_range(errors, config, "boiler_watt_power", 100, 20000, "W")
    _check_range(errors, config, "average_boiler_surroundings_temp", -10, 50, "°C")
    _check_range(errors, config, "cold_water_temp", 0, 30, "°C")
    _check_range(errors, config, "thermal_coupling_ratio", 0.0, 1.0)
    _check_range(errors, config, "boiler_standby_watts", 0, 500, "W")
    _check_range(errors, config, "draw_detection_threshold_c", 0.0, 20.0, "°C")
    _check_range(errors, config, "thermal_model_window_days", 1, 365, "days")
    _check_range(errors, config, "thermal_mass_ratio", 0.0, 1.0)
    _check_range(errors, config, "influxdb_port", 1, 65535)
    _check_range(errors, config, "influxdb_history_years", 1, 20, "years")
    _check_range(errors, config, "min_training_days", 1, 365, "days")
    _check_range(errors, config, "predictor_retrain_weeks", 1, 52, "weeks")
    _check_range(errors, config, "hdo_history_weeks", 2, 12, "weeks")
    _check_range(errors, config, "hdo_decay_weeks", 1, 12, "weeks")
    _check_range(errors, config, "hdo_retrain_weeks", 1, 12, "weeks")
    _check_range(errors, config, "vacation_min_temp", 10, 60, "°C")

    # ── PV / Battery ranges ──────────────────────────────────────────────
    if config.get("has_pv"):
        _check_range(errors, config, "pv_installed_power_kw", 0.1, 200.0, "kWp")
    if config.get("has_battery"):
        _check_range(errors, config, "battery_capacity_kwh", 0.1, 500.0, "kWh")
        _check_range(errors, config, "battery_max_charge_kw", 0.0, 200.0, "kW")

    # ── Cross-field ─────────────────────────────────────────────────────
    try:
        if int(config.get("boiler_min_operation_tmp", 0)) >= int(config.get("boiler_set_tmp", 0)):
            errors.append("boiler_min_operation_tmp must be less than boiler_set_tmp")
    except (ValueError, TypeError):
        pass  # already caught by range checks above

    # ── HDO format ──────────────────────────────────────────────────────
    hdo = str(config.get("hdo_explicit_schedule", "")).strip()
    if hdo and not _HDO_PATTERN.match(hdo):
        errors.append("hdo_explicit_schedule must be empty or in format HH:MM-HH:MM (e.g. 22:00-06:00)")

    return errors


def _check_range(
    errors: list[str],
    config: dict,
    key: str,
    lo: float,
    hi: float,
    unit: str = "",
) -> None:
    try:
        val = float(config[key])
        if not (lo <= val <= hi):
            unit_str = f" {unit}" if unit else ""
            errors.append(f"{key} must be {lo}–{hi}{unit_str} (got {val})")
    except (KeyError, TypeError, ValueError):
        unit_str = f" {unit}" if unit else ""
        errors.append(f"{key} must be a number between {lo} and {hi}{unit_str}")


def load_setup_config() -> dict:
    """Load /data/smartboiler_setup.json merged with defaults.

    Also accepts a legacy /data/options.json written by HA from config.yaml,
    so existing installs keep working without re-running the wizard.
    """
    config = dict(DEFAULTS)

    # 1. Legacy HA options.json (lower priority)
    legacy = os.path.join(_DATA_DIR, "options.json")
    if os.path.exists(legacy):
        try:
            with open(legacy) as f:
                config.update(json.load(f))
        except Exception as e:
            logger.warning("Could not read legacy options.json: %s", e)

    # 2. Web UI setup config (higher priority — overrides legacy)
    if os.path.exists(_SETUP_PATH):
        try:
            with open(_SETUP_PATH) as f:
                config.update(json.load(f))
        except Exception as e:
            logger.warning("Could not read smartboiler_setup.json: %s", e)

    _normalize_influx_fields(config)
    return config


def save_setup_config(data: dict) -> None:
    """Validate and persist config to /data/smartboiler_setup.json.

    Raises ValueError with a newline-joined list of errors if validation fails.
    """
    merged = dict(DEFAULTS)
    merged.update({k: v for k, v in data.items() if k in DEFAULTS})

    # Type coercions — applied before validation so range checks get numbers
    for int_key in ("boiler_volume", "boiler_set_tmp", "boiler_min_operation_tmp",
                    "boiler_watt_power", "average_boiler_surroundings_temp",
                    "cold_water_temp", "boiler_standby_watts",
                    "min_training_days", "predictor_retrain_weeks",
                    "hdo_history_weeks", "hdo_decay_weeks", "hdo_retrain_weeks",
                    "thermal_model_window_days", "influxdb_port",
                    "influxdb_history_years", "vacation_min_temp"):
        try:
            merged[int_key] = int(merged[int_key])
        except (ValueError, TypeError):
            merged[int_key] = DEFAULTS[int_key]

    for float_key in ("thermal_coupling_ratio", "draw_detection_threshold_c",
                      "thermal_mass_ratio", "pv_installed_power_kw",
                      "battery_capacity_kwh", "battery_max_charge_kw"):
        try:
            merged[float_key] = float(merged[float_key])
        except (ValueError, TypeError):
            merged[float_key] = DEFAULTS[float_key]

    merged["has_spot_price"] = bool(merged.get("has_spot_price", False))
    merged["has_pv"] = bool(merged.get("has_pv", False))
    merged["has_battery"] = bool(merged.get("has_battery", False))
    _normalize_influx_fields(merged)

    errors = validate_config(merged)
    if errors:
        raise ValueError("\n".join(errors))

    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_SETUP_PATH, "w") as f:
        json.dump(merged, f, indent=2)
    logger.info("Setup config saved to %s", _SETUP_PATH)


def is_setup_complete(config: Optional[dict] = None) -> bool:
    """Return True if the minimum required field (boiler switch) is configured."""
    if config is None:
        config = load_setup_config()
    return bool(config.get("boiler_switch_entity_id", "").strip())
