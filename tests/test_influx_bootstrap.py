"""Tests for mode-specific InfluxDB bootstrap configuration."""

from smartboiler.influx_bootstrap import InfluxBootstrapper


class _DummyStore:
    def get(self, key, default=None):
        return default

    def set(self, key, value) -> None:
        self.last = (key, value)


def test_standard_mode_uses_standard_influx_fields():
    bootstrapper = InfluxBootstrapper(
        options={
            "operation_mode": "standard",
            "influxdb_host": "influx.local",
            "influxdb_db": "homeassistant",
            "influxdb_standard_relay_entity_id": "switch.boiler",
            "influxdb_standard_power_entity_id": "sensor.power",
            "influxdb_standard_flow_entity_id": "sensor.flow",
            "influxdb_standard_water_temp_entity_id": "sensor.outlet",
            "influxdb_standard_case_tmp_entity_id": "sensor.case",
        },
        store=_DummyStore(),
        thermal_model=None,
    )

    assert bootstrapper.is_configured() is True
    assert bootstrapper.relay_entity == "switch.boiler"
    assert bootstrapper.power_entity == "sensor.power"
    assert bootstrapper.flow_entity == "sensor.flow"
    assert bootstrapper.water_tmp_entity == "sensor.outlet"
    assert bootstrapper.case_tmp_entity == "sensor.case"


def test_simple_mode_uses_simple_influx_fields():
    bootstrapper = InfluxBootstrapper(
        options={
            "operation_mode": "simple",
            "influxdb_host": "influx.local",
            "influxdb_db": "homeassistant",
            "influxdb_simple_relay_entity_id": "switch.boiler",
            "influxdb_simple_power_entity_id": "sensor.power",
            "influxdb_simple_case_tmp_entity_id": "sensor.case",
            "influxdb_simple_inlet_tmp_entity_id": "sensor.inlet",
            "influxdb_simple_outlet_tmp_entity_id": "sensor.outlet",
        },
        store=_DummyStore(),
        thermal_model=None,
    )

    assert bootstrapper.is_configured() is True
    assert bootstrapper.relay_entity == "switch.boiler"
    assert bootstrapper.power_entity == "sensor.power"
    assert bootstrapper.case_tmp_entity == "sensor.case"
    assert bootstrapper.inlet_tmp_entity == "sensor.inlet"
    assert bootstrapper.outlet_tmp_entity == "sensor.outlet"
    assert bootstrapper.flow_entity == ""


def test_flat_influx_fields_remain_a_fallback():
    bootstrapper = InfluxBootstrapper(
        options={
            "operation_mode": "simple",
            "influxdb_host": "influx.local",
            "influxdb_db": "homeassistant",
            "influxdb_relay_entity_id": "switch.boiler",
            "influxdb_power_entity_id": "sensor.power",
            "influxdb_case_tmp_entity_id": "sensor.case",
            "influxdb_inlet_tmp_entity_id": "sensor.inlet",
            "influxdb_water_temp_entity_id": "sensor.outlet",
        },
        store=_DummyStore(),
        thermal_model=None,
    )

    assert bootstrapper.relay_entity == "switch.boiler"
    assert bootstrapper.power_entity == "sensor.power"
    assert bootstrapper.case_tmp_entity == "sensor.case"
    assert bootstrapper.inlet_tmp_entity == "sensor.inlet"
    assert bootstrapper.outlet_tmp_entity == "sensor.outlet"
