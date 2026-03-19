"""
Stub out heavy / network-bound libraries before any smartboiler module is imported.
Only stubs for libraries still referenced by old modules (boiler.py, event_checker.py)
that may be imported transitively during test collection.
"""
import sys
from unittest.mock import MagicMock

_STUBS = [
    # Legacy — kept because boiler.py / event_checker.py still exist
    "influxdb",
    "keras",
    "keras.models",
    "keras.layers",
    "keras.callbacks",
    "keras.optimizers",
    "keras.backend",
    "sklearn",
    "sklearn.preprocessing",
    "skforecast",
    "skforecast.ForecasterAutoreg",
    "geopy",
    "geopy.distance",
    "haversine",
    "h5py",
    "googleapiclient",
    "googleapiclient.discovery",
    "google",
    "google.auth",
    "google.auth.transport",
    "google.auth.transport.requests",
    "google.oauth2",
    "google.oauth2.credentials",
    "google_auth_oauthlib",
    "google_auth_oauthlib.flow",
    "statsmodels",
    "statsmodels.tsa",
    "statsmodels.tsa.statespace",
    "statsmodels.tsa.statespace.sarimax",
]

for _mod in _STUBS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
