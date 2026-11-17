# SolaX Cloud Developer API client (read-only): battery SoC, PV power and
# grid power of one inverter, for users without a SolaX integration in HA.
#
# API (https://developer.solaxcloud.com):
#   POST {base}/openapi/auth/oauth/token        form: client_id, client_secret,
#        grant_type=client_credentials          → result.access_token (code 0)
#   GET  {base}/openapi/v2/device/realtime_data  ?snList=SN&deviceType=1&businessType=1
#        header Authorization: bearer <token>   → result[0] fields (code 10000)
#
# Fields used: batterySOC (%), MPPTTotalInputPower (W, PV), gridPower (W).
# The inverter sends data to the cloud about every 5 minutes, and the API is
# rate-limited, so readings are cached for CACHE_SECONDS.

import logging
import time
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

REGIONS = {
    "eu": "https://openapi-eu.solaxcloud.com",
    "cn": "https://openapi-cn.solaxcloud.com",
    "in": "https://openapi-in.solaxcloud.com",
}
CODE_TOKEN_OK = 0
CODE_DATA_OK = 10000
DEVICE_TYPE_INVERTER = 1
CACHE_SECONDS = 60
TOKEN_REFRESH_MARGIN_S = 3600   # renew the ~30-day token an hour before it expires


class SolaxError(Exception):
    """API call failed (HTTP error, bad credentials, unexpected response)."""


class SolaxCloud:
    """Reads realtime values of one inverter from SolaX Cloud."""

    def __init__(self, client_id: str, client_secret: str, inverter_sn: str,
                 region: str = "eu", business_type: int = 1, timeout: int = 15):
        if region not in REGIONS:
            raise ValueError(f"region must be one of {', '.join(REGIONS)}")
        self.base_url = REGIONS[region]
        self.client_id = client_id
        self.client_secret = client_secret
        self.inverter_sn = inverter_sn
        self.business_type = business_type
        self.timeout = timeout
        self._token: Optional[str] = None
        self._token_expires = 0.0
        self._data: Dict = {}
        self._data_time = 0.0

    # ── Auth ─────────────────────────────────────────────────────────────

    def _get_token(self) -> str:
        if self._token and time.time() < self._token_expires - TOKEN_REFRESH_MARGIN_S:
            return self._token
        resp = requests.post(
            f"{self.base_url}/openapi/auth/oauth/token",
            data={"client_id": self.client_id, "client_secret": self.client_secret,
                  "grant_type": "client_credentials"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        result = body.get("result") or {}
        if body.get("code") != CODE_TOKEN_OK or not result.get("access_token"):
            raise SolaxError(f"SolaX login failed: {body.get('message', body.get('code'))}")
        self._token = result["access_token"]
        self._token_expires = time.time() + float(result.get("expires_in", 3600))
        return self._token

    # ── Data ─────────────────────────────────────────────────────────────

    def realtime(self, max_age_s: float = CACHE_SECONDS) -> Dict:
        """All realtime fields of the inverter (cached). Raises SolaxError."""
        if self._data and time.time() - self._data_time < max_age_s:
            return self._data
        try:
            resp = requests.get(
                f"{self.base_url}/openapi/v2/device/realtime_data",
                params={"snList": self.inverter_sn, "deviceType": DEVICE_TYPE_INVERTER,
                        "businessType": self.business_type},
                headers={"Authorization": f"bearer {self._get_token()}"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            body = resp.json()
        except requests.RequestException as e:
            raise SolaxError(f"SolaX request failed: {e}") from e
        if body.get("code") != CODE_DATA_OK:
            self._token = None   # an expired or revoked token is the usual cause
            raise SolaxError(f"SolaX data error: {body.get('message', body.get('code'))}")
        rows = body.get("result") or []
        if not rows:
            raise SolaxError(f"SolaX returned no data for inverter {self.inverter_sn}")
        self._data, self._data_time = rows[0], time.time()
        return self._data

    def _number(self, field: str) -> Optional[float]:
        try:
            value = self.realtime().get(field)
            return None if value is None else float(value)
        except (SolaxError, TypeError, ValueError) as e:
            logger.warning("SolaX %s unavailable: %s", field, e)
            return None

    def battery_soc(self) -> Optional[float]:
        """Battery state of charge in %."""
        return self._number("batterySOC")

    def pv_power_w(self) -> Optional[float]:
        """Total PV (MPPT input) power in W."""
        return self._number("MPPTTotalInputPower")

    def grid_power_w(self) -> Optional[float]:
        """Grid power in W as reported by SolaX (sign depends on the model;
        check it with the wizard's Test button and use pv_surplus_inverted)."""
        return self._number("gridPower")


def from_config(cfg: dict) -> Optional[SolaxCloud]:
    """Client when Client ID, Secret and inverter SN are configured, else None."""
    cid = str(cfg.get("solax_client_id", "") or "").strip()
    secret = str(cfg.get("solax_client_secret", "") or "").strip()
    sn = str(cfg.get("solax_inverter_sn", "") or "").strip()
    if not (cid and secret and sn):
        return None
    return SolaxCloud(cid, secret, sn, region=cfg.get("solax_region", "eu") or "eu")
