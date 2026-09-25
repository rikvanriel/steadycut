"""Camera registry: measured data comes from the user's config, never the code.

A body is usable only once its axis map has been measured and stored; a body
with no stored map is reported as needing a measurement rather than silently
inheriting another camera's numbers.
"""
import json

import pytest

from steadycut.ingest.cameras import registry as R
from steadycut.ingest.cameras import store
from steadycut.ingest.cameras.profiles import AxisMap

X4_AXIS = {"yaw": [1, -1], "pitch": [2, -1], "roll": [0, -1]}


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    """Point the user's config and cache directories at a temp dir."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path


def test_unmeasured_body_is_unknown_and_says_where_to_look(cfg) -> None:
    entry = R.describe("NEVER-SEEN-0000")
    assert entry["known"] is False
    assert "cameras.json" in entry["action"]


def test_measured_body_resolves_and_its_map_parses(cfg) -> None:
    R.record_measured("S1", model="Insta360 X4", fw="v1", axis_map=X4_AXIS)
    entry = R.describe("S1")
    assert entry["known"] is True and entry["model"] == "Insta360 X4"
    assert R.axis_map_for("S1") == AxisMap(yaw=(1, -1), pitch=(2, -1),
                                          roll=(0, -1))


def test_registered_body_without_a_map_is_not_usable(cfg) -> None:
    R.record_measured("S2", model="Insta360 X4")
    entry = R.describe("S2")
    assert entry["known"] is False
    assert "axis map" in entry["action"]


def test_measurements_land_in_the_config_file(cfg) -> None:
    R.record_measured("S3", axis_map=X4_AXIS)
    assert store.registry_path().exists()
    assert "S3" in json.loads(store.registry_path().read_text())


def test_corrupt_stored_map_is_treated_as_unmeasured(cfg) -> None:
    store.upsert("S4", {"axis_map": {"yaw": [1, -1]}})  # pitch/roll missing
    assert R.axis_map_for("S4") is None
    assert R.describe("S4")["known"] is False


def test_file_camera_uses_the_cache_without_a_live_parse(cfg) -> None:
    # A cache hit must never touch telemetry_parser (the live call is the
    # OOM-killed one): seed the cache and the measurement, then identify.
    f = cfg / "clip.insv"
    f.write_bytes(b"\x00" * 1024)
    key = f"{f.resolve()}|1024|{int(f.stat().st_mtime)}"
    R.record_measured("IBMYA2410M3EB5", model="Insta360 X4",
                      axis_map=X4_AXIS)
    store.store_file_cache(key, {"serial_number": "IBMYA2410M3EB5",
                                 "model": "Insta360 X4", "fw": "v9",
                                 "offset_len": 16})
    got = R.file_camera(f)
    assert got["serial"] == "IBMYA2410M3EB5" and got["cached"] is True
    assert got["known"] is True
