"""Tests for smartboiler.setup_config — validation, save, load, is_setup_complete."""

import json
import os
import tempfile

import pytest

# Point DATA_PATH at a temp dir so tests never touch /data/
_TMP = tempfile.mkdtemp()
os.environ["DATA_PATH"] = _TMP

# Import after env var is set so module-level paths are correct
from smartboiler.setup_config import (  # noqa: E402
    DEFAULTS,
    is_setup_complete,
    load_setup_config,
    save_setup_config,
    validate_config,
)


# ── helpers ─────────────────────────────────────────────────────────────────

def _valid() -> dict:
    """Return a minimal valid config dict (all required fields present)."""
    cfg = dict(DEFAULTS)
    cfg["boiler_switch_entity_id"] = "switch.boiler"
    return cfg


# ── validate_config ──────────────────────────────────────────────────────────

class TestValidateConfig:
    def test_valid_defaults_with_switch(self):
        assert validate_config(_valid()) == []

    def test_missing_switch_entity(self):
        cfg = _valid()
        cfg["boiler_switch_entity_id"] = ""
        errs = validate_config(cfg)
        assert any("boiler_switch_entity_id" in e for e in errs)

    def test_whitespace_switch_entity(self):
        cfg = _valid()
        cfg["boiler_switch_entity_id"] = "   "
        errs = validate_config(cfg)
        assert any("boiler_switch_entity_id" in e for e in errs)

    # ── Enum checks ─────────────────────────────────────────────────────────

    def test_invalid_operation_mode(self):
        cfg = _valid()
        cfg["operation_mode"] = "turbo"
        errs = validate_config(cfg)
        assert any("operation_mode" in e for e in errs)

    @pytest.mark.parametrize("mode", ["simple", "standard"])
    def test_valid_operation_modes(self, mode):
        cfg = _valid()
        cfg["operation_mode"] = mode
        assert validate_config(cfg) == []

    def test_invalid_spot_price_region(self):
        cfg = _valid()
        cfg["spot_price_region"] = "US"
        errs = validate_config(cfg)
        assert any("spot_price_region" in e for e in errs)

    @pytest.mark.parametrize("region", ["CZ", "SK", "AT", "DE", "PL", "HU", "FR", "IT", "ES"])
    def test_valid_spot_regions(self, region):
        cfg = _valid()
        cfg["spot_price_region"] = region
        assert validate_config(cfg) == []

    def test_invalid_prediction_conservatism(self):
        cfg = _valid()
        cfg["prediction_conservatism"] = "extreme"
        errs = validate_config(cfg)
        assert any("prediction_conservatism" in e for e in errs)

    @pytest.mark.parametrize("val", ["low", "medium", "high"])
    def test_valid_conservatism(self, val):
        cfg = _valid()
        cfg["prediction_conservatism"] = val
        assert validate_config(cfg) == []

    def test_invalid_vacation_mode(self):
        cfg = _valid()
        cfg["vacation_mode"] = "away"
        errs = validate_config(cfg)
        assert any("vacation_mode" in e for e in errs)

    def test_invalid_logging_level(self):
        cfg = _valid()
        cfg["logging_level"] = "VERBOSE"
        errs = validate_config(cfg)
        assert any("logging_level" in e for e in errs)

    # ── Range checks ────────────────────────────────────────────────────────

    @pytest.mark.parametrize("key,bad_value", [
        ("boiler_volume", 5),
        ("boiler_volume", 1001),
        ("boiler_set_tmp", 29),
        ("boiler_set_tmp", 96),
        ("boiler_min_operation_tmp", 19),
        ("boiler_min_operation_tmp", 91),
        ("boiler_watt_power", 99),
        ("boiler_watt_power", 20001),
        ("average_boiler_surroundings_temp", -11),
        ("average_boiler_surroundings_temp", 51),
        ("cold_water_temp", -1),
        ("cold_water_temp", 31),
        ("thermal_coupling_ratio", -0.1),
        ("thermal_coupling_ratio", 1.1),
        ("boiler_standby_watts", -1),
        ("boiler_standby_watts", 501),
        ("draw_detection_threshold_c", -0.1),
        ("draw_detection_threshold_c", 20.1),
        ("thermal_model_window_days", 0),
        ("thermal_model_window_days", 366),
        ("thermal_mass_ratio", -0.1),
        ("thermal_mass_ratio", 1.1),
        ("influxdb_port", 0),
        ("influxdb_port", 65536),
        ("influxdb_history_years", 0),
        ("influxdb_history_years", 21),
        ("min_training_days", 0),
        ("min_training_days", 366),
        ("predictor_retrain_weeks", 0),
        ("predictor_retrain_weeks", 53),
        ("hdo_history_weeks", 1),
        ("hdo_history_weeks", 13),
        ("hdo_decay_weeks", 0),
        ("hdo_decay_weeks", 13),
        ("hdo_retrain_weeks", 0),
        ("hdo_retrain_weeks", 13),
        ("vacation_min_temp", 9),
        ("vacation_min_temp", 61),
    ])
    def test_out_of_range_values(self, key, bad_value):
        cfg = _valid()
        cfg[key] = bad_value
        errs = validate_config(cfg)
        assert any(key in e for e in errs), f"Expected error for {key}={bad_value}"

    def test_boundary_values_accepted(self):
        cfg = _valid()
        cfg["boiler_volume"] = 10
        cfg["boiler_set_tmp"] = 95
        cfg["boiler_min_operation_tmp"] = 20
        assert validate_config(cfg) == []

    # ── Cross-field: min < set ───────────────────────────────────────────────

    def test_min_operation_tmp_must_be_less_than_set_tmp(self):
        cfg = _valid()
        cfg["boiler_set_tmp"] = 55
        cfg["boiler_min_operation_tmp"] = 55
        errs = validate_config(cfg)
        assert any("boiler_min_operation_tmp" in e and "boiler_set_tmp" in e for e in errs)

    def test_min_operation_tmp_equal_to_set_tmp_fails(self):
        cfg = _valid()
        cfg["boiler_set_tmp"] = 60
        cfg["boiler_min_operation_tmp"] = 60
        errs = validate_config(cfg)
        assert any("boiler_min_operation_tmp" in e for e in errs)

    # ── HDO schedule format ──────────────────────────────────────────────────

    def test_empty_hdo_is_valid(self):
        cfg = _valid()
        cfg["hdo_explicit_schedule"] = ""
        assert validate_config(cfg) == []

    @pytest.mark.parametrize("hdo", ["22:00-06:00", "0:00-8:00", "23:59-00:00"])
    def test_valid_hdo_formats(self, hdo):
        cfg = _valid()
        cfg["hdo_explicit_schedule"] = hdo
        assert validate_config(cfg) == []

    @pytest.mark.parametrize("bad_hdo", ["2200-0600", "22:00", "22:00/06:00", "aa:bb-cc:dd"])
    def test_invalid_hdo_formats(self, bad_hdo):
        cfg = _valid()
        cfg["hdo_explicit_schedule"] = bad_hdo
        errs = validate_config(cfg)
        assert any("hdo_explicit_schedule" in e for e in errs)

    def test_multiple_errors_reported(self):
        cfg = _valid()
        cfg["boiler_switch_entity_id"] = ""
        cfg["operation_mode"] = "turbo"
        cfg["boiler_volume"] = 9999
        errs = validate_config(cfg)
        assert len(errs) >= 3


# ── save_setup_config / load_setup_config ────────────────────────────────────

class TestSaveLoadConfig:
    def setup_method(self):
        # Clean up any file left by a previous test
        setup_path = os.path.join(_TMP, "smartboiler_setup.json")
        if os.path.exists(setup_path):
            os.remove(setup_path)

    def test_save_then_load_roundtrip(self):
        data = _valid()
        data["boiler_volume"] = 80
        save_setup_config(data)
        loaded = load_setup_config()
        assert loaded["boiler_volume"] == 80
        assert loaded["boiler_switch_entity_id"] == "switch.boiler"

    def test_save_raises_on_invalid_config(self):
        data = _valid()
        data["boiler_switch_entity_id"] = ""
        with pytest.raises(ValueError):
            save_setup_config(data)

    def test_save_raises_includes_all_errors(self):
        data = _valid()
        data["boiler_switch_entity_id"] = ""
        data["operation_mode"] = "bad"
        try:
            save_setup_config(data)
            pytest.fail("Expected ValueError")
        except ValueError as exc:
            msg = str(exc)
            assert "boiler_switch_entity_id" in msg
            assert "operation_mode" in msg

    def test_save_coerces_string_int(self):
        data = _valid()
        data["boiler_volume"] = "150"  # string from HTML form
        save_setup_config(data)
        loaded = load_setup_config()
        assert loaded["boiler_volume"] == 150

    def test_save_coerces_string_float(self):
        data = _valid()
        data["thermal_coupling_ratio"] = "0.5"
        save_setup_config(data)
        loaded = load_setup_config()
        assert loaded["thermal_coupling_ratio"] == 0.5

    def test_save_ignores_unknown_keys(self):
        data = _valid()
        data["totally_unknown_field"] = "should be ignored"
        save_setup_config(data)
        loaded = load_setup_config()
        assert "totally_unknown_field" not in loaded

    def test_load_merges_defaults(self):
        # Save only the switch, everything else should come from DEFAULTS
        save_setup_config(_valid())
        loaded = load_setup_config()
        for key, default_val in DEFAULTS.items():
            assert key in loaded

    def test_load_populates_mode_specific_influx_fields_from_flat_config(self, tmp_path):
        import smartboiler.setup_config as sc
        original_data_dir = sc._DATA_DIR
        original_setup_path = sc._SETUP_PATH
        sc._DATA_DIR = str(tmp_path)
        sc._SETUP_PATH = str(tmp_path / "smartboiler_setup.json")
        try:
            (tmp_path / "smartboiler_setup.json").write_text(json.dumps({
                "boiler_switch_entity_id": "switch.boiler",
                "operation_mode": "simple",
                "influxdb_host": "influx.local",
                "influxdb_db": "homeassistant",
                "influxdb_relay_entity_id": "switch.boiler",
                "influxdb_power_entity_id": "sensor.boiler_power",
                "influxdb_case_tmp_entity_id": "sensor.boiler_case",
                "influxdb_inlet_tmp_entity_id": "sensor.inlet",
                "influxdb_water_temp_entity_id": "sensor.outlet",
            }))
            loaded = sc.load_setup_config()
            assert loaded["influxdb_simple_relay_entity_id"] == "switch.boiler"
            assert loaded["influxdb_simple_power_entity_id"] == "sensor.boiler_power"
            assert loaded["influxdb_simple_case_tmp_entity_id"] == "sensor.boiler_case"
            assert loaded["influxdb_simple_inlet_tmp_entity_id"] == "sensor.inlet"
            assert loaded["influxdb_simple_outlet_tmp_entity_id"] == "sensor.outlet"
        finally:
            sc._DATA_DIR = original_data_dir
            sc._SETUP_PATH = original_setup_path

    def test_save_syncs_active_mode_specific_influx_fields_back_to_flat_keys(self):
        data = _valid()
        data["operation_mode"] = "standard"
        data["influxdb_host"] = "influx.local"
        data["influxdb_db"] = "homeassistant"
        data["influxdb_standard_relay_entity_id"] = "switch.boiler"
        data["influxdb_standard_power_entity_id"] = "sensor.boiler_power"
        data["influxdb_standard_flow_entity_id"] = "sensor.flow"
        data["influxdb_standard_water_temp_entity_id"] = "sensor.outlet"
        data["influxdb_standard_case_tmp_entity_id"] = "sensor.case"
        save_setup_config(data)
        loaded = load_setup_config()
        assert loaded["influxdb_relay_entity_id"] == "switch.boiler"
        assert loaded["influxdb_power_entity_id"] == "sensor.boiler_power"
        assert loaded["influxdb_flow_entity_id"] == "sensor.flow"
        assert loaded["influxdb_water_temp_entity_id"] == "sensor.outlet"
        assert loaded["influxdb_case_tmp_entity_id"] == "sensor.case"

    def test_load_without_file_returns_defaults(self):
        # No file exists; should still return DEFAULTS
        loaded = load_setup_config()
        for key, val in DEFAULTS.items():
            assert loaded[key] == val

    def test_load_merges_legacy_options_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATA_PATH", str(tmp_path))
        # Reload module-level paths by patching directly
        import smartboiler.setup_config as sc
        original_data_dir = sc._DATA_DIR
        original_setup_path = sc._SETUP_PATH
        sc._DATA_DIR = str(tmp_path)
        sc._SETUP_PATH = str(tmp_path / "smartboiler_setup.json")
        try:
            legacy = tmp_path / "options.json"
            legacy.write_text(json.dumps({"logging_level": "DEBUG"}))
            loaded = sc.load_setup_config()
            assert loaded["logging_level"] == "DEBUG"
        finally:
            sc._DATA_DIR = original_data_dir
            sc._SETUP_PATH = original_setup_path

    def test_web_ui_config_overrides_legacy(self, tmp_path):
        import smartboiler.setup_config as sc
        original_data_dir = sc._DATA_DIR
        original_setup_path = sc._SETUP_PATH
        sc._DATA_DIR = str(tmp_path)
        sc._SETUP_PATH = str(tmp_path / "smartboiler_setup.json")
        try:
            (tmp_path / "options.json").write_text(json.dumps({"logging_level": "DEBUG"}))
            (tmp_path / "smartboiler_setup.json").write_text(
                json.dumps({"logging_level": "ERROR", "boiler_switch_entity_id": "switch.x"})
            )
            loaded = sc.load_setup_config()
            assert loaded["logging_level"] == "ERROR"
        finally:
            sc._DATA_DIR = original_data_dir
            sc._SETUP_PATH = original_setup_path


# ── is_setup_complete ────────────────────────────────────────────────────────

class TestIsSetupComplete:
    def test_returns_false_when_switch_empty(self):
        assert is_setup_complete({"boiler_switch_entity_id": ""}) is False

    def test_returns_false_when_switch_whitespace(self):
        assert is_setup_complete({"boiler_switch_entity_id": "   "}) is False

    def test_returns_false_when_key_missing(self):
        assert is_setup_complete({}) is False

    def test_returns_true_when_switch_set(self):
        assert is_setup_complete({"boiler_switch_entity_id": "switch.boiler"}) is True

    def test_loads_from_file_when_no_arg(self, tmp_path):
        import smartboiler.setup_config as sc
        original_data_dir = sc._DATA_DIR
        original_setup_path = sc._SETUP_PATH
        sc._DATA_DIR = str(tmp_path)
        sc._SETUP_PATH = str(tmp_path / "smartboiler_setup.json")
        try:
            (tmp_path / "smartboiler_setup.json").write_text(
                json.dumps({"boiler_switch_entity_id": "switch.heater"})
            )
            assert sc.is_setup_complete() is True
        finally:
            sc._DATA_DIR = original_data_dir
            sc._SETUP_PATH = original_setup_path
