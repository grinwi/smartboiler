"""Tests for StateStore persistence: round trips and crash-safe writes."""
import json
from unittest.mock import patch

import pandas as pd
import pytest

from smartboiler.state_store import StateStore, atomic_write


def test_json_state_round_trip(tmp_path):
    StateStore(str(tmp_path)).set("answer", 42)
    assert StateStore(str(tmp_path)).get("answer") == 42


def test_pickle_round_trip(tmp_path):
    StateStore(str(tmp_path)).save_pickle("thing", {"a": [1, 2]})
    assert StateStore(str(tmp_path)).load_pickle("thing") == {"a": [1, 2]}


def test_consumption_history_round_trip(tmp_path):
    idx = pd.date_range(pd.Timestamp.now().floor("h") - pd.Timedelta(hours=2), periods=3, freq="h")
    df = pd.DataFrame({"consumed_kwh": [0.1, 0.2, 0.3], "relay_on": [0, 1, 0], "power_w": [0, 2000, 0]}, index=idx)
    store = StateStore(str(tmp_path))
    store.save_consumption_history(df)
    pd.testing.assert_frame_equal(store.load_consumption_history(), df, check_freq=False)


def test_atomic_write_replaces_content(tmp_path):
    target = tmp_path / "f.json"
    atomic_write(target, b"old")
    atomic_write(target, b"new")
    assert target.read_bytes() == b"new"


def test_failed_write_keeps_previous_file_and_leaves_no_temp(tmp_path):
    target = tmp_path / "state.json"
    target.write_text(json.dumps({"keep": True}))
    with patch("smartboiler.state_store.os.replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            atomic_write(target, b'{"keep": false}')
    assert json.loads(target.read_text()) == {"keep": True}
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_store_survives_failed_save(tmp_path):
    store = StateStore(str(tmp_path))
    store.set("a", 1)
    with patch("smartboiler.state_store.os.replace", side_effect=OSError("disk full")):
        store.set("a", 2)  # logged, not raised
    assert StateStore(str(tmp_path)).get("a") == 1
