# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# Flask application setup for the SmartBoiler dashboard.
# Exposed via HA ingress on port 8099.
# Security: HA Supervisor handles auth before forwarding; we add rate limiting + input validation.
#
# Module layout:
#   web_server.py    — app init, state providers, rate limiting, run_dashboard()
#   web_routes.py    — all @app.route() handlers
#   web_templates.py — embedded HTML/CSS/JS dashboard template

import logging
import threading
from typing import Any, Callable, Dict, Optional

from flask import Flask

logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── State providers (injected by controller before starting) ──────────────

_state_provider: Optional[Callable[[], Dict]] = None
_extra_provider: Optional[Callable[[str, Dict], Dict]] = None
_calendar_manager: Optional[Any] = None


def set_state_provider(provider: Callable[[], Dict]) -> None:
    """Inject a callable that returns current system state for the dashboard."""
    global _state_provider
    _state_provider = provider


def set_extra_provider(provider: Callable[[str, Dict], Dict]) -> None:
    """Inject a callable for extended data queries (history, accuracy, predictor)."""
    global _extra_provider
    _extra_provider = provider


def set_calendar_manager(cm: Any) -> None:
    """Inject the CalendarManager so calendar routes can use it."""
    global _calendar_manager
    _calendar_manager = cm


def _get_state() -> Dict:
    if _state_provider is None:
        return {}
    try:
        return _state_provider()
    except Exception as e:
        logger.error("State provider error: %s", e)
        return {}


def _get_extra(endpoint: str, params: Dict) -> Dict:
    if _extra_provider is None:
        return {}
    try:
        return _extra_provider(endpoint, params) or {}
    except Exception as e:
        logger.error("Extra provider error (%s): %s", endpoint, e)
        return {}


# ── Rate limiting (simple token bucket, no external dependencies) ─────────

_request_counts: Dict[str, int] = {}
_request_lock = threading.Lock()
RATE_LIMIT = 120  # max requests per IP per sliding window 


def _check_rate_limit(ip: str, limit: int = RATE_LIMIT) -> bool:
    with _request_lock:
        count = _request_counts.get(ip, 0)
        if count >= limit:
            return False
        _request_counts[ip] = count + 1
    return True


def _reset_rate_limits() -> None:
    with _request_lock:
        _request_counts.clear()


def _rate_limited() -> bool:
    from flask import request
    ip = request.remote_addr or "unknown"
    if not _check_rate_limit(ip):
        logger.warning("Rate limit exceeded for %s", ip)
        return True
    return False


# ── Server start ──────────────────────────────────────────────────────────


def run_dashboard(
    host: str = "0.0.0.0",
    port: int = 8099,
    debug: bool = False,
) -> None:
    """Start the Flask dashboard. Call this in a daemon thread."""
    import time

    def _rate_reset_loop():
        while True:
            time.sleep(60)
            _reset_rate_limits()

    t = threading.Thread(target=_rate_reset_loop, daemon=True)
    t.start()

    app.run(host=host, port=port, debug=debug, use_reloader=False)


# ── Register routes ───────────────────────────────────────────────────────
# Imported last so that `app` and all helpers above are already defined when
# web_routes.py imports them.  The import is a side-effect: decorators in
# web_routes.py register every route with the `app` instance above.

from smartboiler import web_routes  # noqa: F401, E402
