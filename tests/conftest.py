"""
Stub out heavy / network-bound libraries before any smartboiler module is imported.
This allows unit tests to run without installing influxdb, keras, sklearn, etc.
"""
import sys
from unittest.mock import MagicMock

_STUBS = [
    # InfluxDB
    "influxdb",
    # Keras / TensorFlow
    "keras",
    "keras.models",
    "keras.layers",
    "keras.callbacks",
    "keras.optimizers",
    "keras.backend",
    # scikit-learn
    "sklearn",
    "sklearn.preprocessing",
    # skforecast
    "skforecast",
    "skforecast.ForecasterAutoreg",
    # geopy
    "geopy",
    "geopy.distance",
    # haversine
    "haversine",
    # h5py
    "h5py",
    # Google API client
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
    # statsmodels
    "statsmodels",
    "statsmodels.tsa",
    "statsmodels.tsa.statespace",
    "statsmodels.tsa.statespace.sarimax",
]

for _mod in _STUBS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
