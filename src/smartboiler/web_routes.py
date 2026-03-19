# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# Flask route handlers for the SmartBoiler dashboard.
# Imported at the bottom of web_server.py to register routes with the app instance.
# Do not import this module directly — import web_server instead.

import logging
from datetime import datetime

from flask import Response, abort, jsonify, redirect, render_template, request

# Imported from web_server after app and helpers are fully defined (avoids circular import)
from smartboiler.web_server import (  # noqa: E402
    app,
    _rate_limited,
    _get_state,
    _get_extra,
    _calendar_manager,
)
from smartboiler.setup_config import load_setup_config, save_setup_config, is_setup_complete

logger = logging.getLogger(__name__)


# ── Security headers ───────────────────────────────────────────────────────


@app.after_request
def add_security_headers(response: Response) -> Response:
    """Add security headers to all responses."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'"
    )
    return response


# ── Core routes ────────────────────────────────────────────────────────────


@app.route("/")
def index() -> str:
    if _rate_limited():
        abort(429)
    if not is_setup_complete():
        return redirect("setup")
    return render_template("dashboard.html")


@app.route("/setup")
def setup_wizard() -> str:
    return render_template("setup.html", settings_mode=False)


@app.route("/settings")
def settings_page() -> str:
    return render_template("setup.html", settings_mode=True)


# ── Setup / config API ─────────────────────────────────────────────────────


@app.route("/api/ha/entities")
def api_ha_entities_full() -> Response:
    """Return all HA entity IDs grouped by domain — used by the setup wizard."""
    if _rate_limited():
        abort(429)
    try:
        from smartboiler.ha_client import HAClient
        client = HAClient()
        states = client.get_all_states()
        entities = []
        for s in states:
            eid = s.get("entity_id", "")
            if not eid:
                continue
            domain = eid.split(".")[0]
            friendly = (
                s.get("attributes", {}).get("friendly_name") or eid
            )
            entities.append({"entity_id": eid, "name": friendly, "domain": domain})
        entities.sort(key=lambda e: e["entity_id"])
        return jsonify(entities)
    except Exception as e:
        logger.error("api_ha_entities_full error: %s", e)
        return jsonify([])


@app.route("/api/config", methods=["GET"])
def api_get_config() -> Response:
    """Return current setup configuration."""
    if _rate_limited():
        abort(429)
    return jsonify(load_setup_config())


@app.route("/api/config", methods=["POST"])
def api_save_config() -> Response:
    """Save setup configuration and signal the controller to (re)start."""
    if _rate_limited():
        abort(429)
    try:
        data = request.get_json(force=True) or {}
        save_setup_config(data)
        return jsonify({"ok": True})
    except ValueError as e:
        # Validation errors from save_setup_config — return as 400
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        logger.error("api_save_config error: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/test/influxdb", methods=["POST"])
def api_test_influxdb() -> Response:
    """Test InfluxDB connection and return available entity names."""
    if _rate_limited():
        abort(429)
    try:
        body = request.get_json(force=True) or {}
        host = str(body.get("host", "")).strip()
        port = int(body.get("port", 8086))
        db = str(body.get("db", "homeassistant")).strip()
        username = str(body.get("username", "")).strip() or None
        password = str(body.get("password", "")).strip() or None

        if not host:
            return jsonify({"ok": False, "error": "Host is required"}), 400

        from influxdb import InfluxDBClient  # type: ignore
        client = InfluxDBClient(
            host=host, port=port, database=db,
            username=username, password=password,
            timeout=8,
        )
        # Ping to check connectivity
        client.ping()

        # The HA InfluxDB integration schema:
        #   measurement = unit_of_measurement  (e.g. "°C", "W", "state")
        #   tag entity_id = HA entity id       (e.g. "sensor.boiler_temp")
        #   field value   = numeric value
        #
        # Build a list of "measurement::entity_id" strings so the user
        # can pick the right combination in the setup wizard.
        entities = []
        for m in client.get_list_measurements():
            mname = m["name"]
            try:
                rows = client.query(
                    f'SHOW TAG VALUES FROM "{mname}" WITH KEY = "entity_id"'
                )
                for row in rows.get_points():
                    eid = row.get("value", "")
                    if eid:
                        entities.append({
                            "measurement": mname,
                            "entity_id": eid,
                            "display": f"{mname} :: {eid}",
                        })
            except Exception:
                pass  # skip measurements with no entity_id tag

        entities.sort(key=lambda e: e["display"])
        return jsonify({"ok": True, "entities": entities})
    except Exception as e:
        logger.warning("InfluxDB test failed: %s", e)
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/status")
def api_status() -> Response:
    if _rate_limited():
        abort(429)
    try:
        state = _get_state()
        safe_keys = {
            "boiler_temp", "relay_on", "boiler_status", "set_tmp", "min_tmp",
            "heating_until", "last_legionella", "predictor_has_data",
            "forecast_24h", "plan_slots", "spot_prices_today",
            "hdo_schedule", "sys_info", "calendar_events", "calendar_entity_id",
        }
        return jsonify({k: v for k, v in state.items() if k in safe_keys})
    except Exception as e:
        logger.error("api_status error: %s", e)
        abort(500)


@app.route("/api/ping")
def api_ping() -> Response:
    return jsonify({"ok": True})


@app.route("/api/history")
def api_history() -> Response:
    if _rate_limited():
        abort(429)
    period = request.args.get("period", "7d")
    if period not in ("1d", "7d", "30d", "90d"):
        period = "7d"
    try:
        return jsonify(_get_extra("history", {"period": period}))
    except Exception as e:
        logger.error("api_history error: %s", e)
        abort(500)


@app.route("/api/accuracy")
def api_accuracy() -> Response:
    if _rate_limited():
        abort(429)
    try:
        return jsonify(_get_extra("accuracy", {}))
    except Exception as e:
        logger.error("api_accuracy error: %s", e)
        abort(500)


@app.route("/api/predictor")
def api_predictor() -> Response:
    if _rate_limited():
        abort(429)
    try:
        return jsonify(_get_extra("predictor", {}))
    except Exception as e:
        logger.error("api_predictor error: %s", e)
        abort(500)


@app.route("/api/entities")
def api_entities() -> Response:
    """Return entity list for UI-based entity selection (name + id only)."""
    if _rate_limited():
        abort(429)
    state = _get_state()
    entities = state.get("available_entities", [])
    safe = [
        {
            "entity_id": e.get("entity_id", ""),
            "name": e.get("friendly_name") or e.get("entity_id", ""),
        }
        for e in entities
        if e.get("entity_id")
    ]
    return jsonify(safe)


# ── Calendar routes ────────────────────────────────────────────────────────


@app.route("/api/calendar/events", methods=["GET"])
def api_calendar_events() -> Response:
    """Return upcoming calendar events (next 7 days)."""
    if _rate_limited():
        abort(429)
    if _calendar_manager is None:
        return jsonify([])
    try:
        days = min(int(request.args.get("days", 7)), 30)
        return jsonify(_calendar_manager.upcoming_events_json(days=days))
    except Exception as e:
        logger.error("api_calendar_events error: %s", e)
        abort(500)


@app.route("/api/calendar/events", methods=["POST"])
def api_calendar_create() -> Response:
    """Create a boiler calendar event.

    Body JSON:
      { "event_type": "vacation_min"|"vacation_off"|"boost_max"|"boost_temp",
        "start": "2026-03-15 08:00",
        "end":   "2026-03-22 20:00",
        "target_temp": 65  // only for boost_temp
      }
    """
    if _rate_limited():
        abort(429)
    if _calendar_manager is None:
        return jsonify({"ok": False, "error": "Calendar not configured"}), 503
    try:
        body = request.get_json(force=True) or {}
        event_type = str(body.get("event_type", ""))
        if event_type not in ("vacation_min", "vacation_off", "boost_max", "boost_temp"):
            return jsonify({"ok": False, "error": "Invalid event_type"}), 400

        start_raw = str(body.get("start", ""))
        end_raw   = str(body.get("end",   ""))
        try:
            start = datetime.strptime(start_raw, "%Y-%m-%d %H:%M")
            end   = datetime.strptime(end_raw,   "%Y-%m-%d %H:%M")
        except ValueError:
            return jsonify({"ok": False, "error": "Invalid date format; use YYYY-MM-DD HH:MM"}), 400

        if end <= start:
            return jsonify({"ok": False, "error": "end must be after start"}), 400

        target_temp = None
        if event_type == "boost_temp":
            try:
                target_temp = float(body.get("target_temp", 0))
                if not (30 <= target_temp <= 95):
                    raise ValueError
            except (ValueError, TypeError):
                return jsonify({"ok": False, "error": "target_temp must be 30–95"}), 400

        ok = _calendar_manager.create_event(event_type, start, end, target_temp)
        return jsonify({"ok": ok})
    except Exception as e:
        logger.error("api_calendar_create error: %s", e)
        abort(500)


@app.route("/api/calendar/events/<path:event_id>", methods=["DELETE"])
def api_calendar_delete(event_id: str) -> Response:
    """Delete a calendar event by its uid."""
    if _rate_limited():
        abort(429)
    if _calendar_manager is None:
        return jsonify({"ok": False, "error": "Calendar not configured"}), 503
    try:
        ok = _calendar_manager.delete_event(event_id)
        return jsonify({"ok": ok})
    except Exception as e:
        logger.error("api_calendar_delete error: %s", e)
        abort(500)
