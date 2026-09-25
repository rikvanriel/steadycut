"""The camera registry is the user's file, not a source-tree constant.

Measured data must live where a package install can write it and where a
distribution can ship defaults, so the paths follow the XDG directories and
the file cache is kept separate from the measurements.
"""
import json
from pathlib import Path

from steadycut.ingest.cameras import store


def test_registry_path_follows_xdg_config_home(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert store.registry_path() == tmp_path / "steadycut" / "cameras.json"
    monkeypatch.delenv("XDG_CONFIG_HOME")
    assert store.registry_path() == (
        Path.home() / ".config" / "steadycut" / "cameras.json")


def test_upsert_round_trips_and_merges(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert store.entry("S1") is None
    store.upsert("S1", {"model": "Cam", "axis_map": {"yaw": [1, -1]}})
    merged = store.upsert("S1", {"fw": "v2"})
    assert merged["model"] == "Cam" and merged["fw"] == "v2"
    on_disk = json.loads(store.registry_path().read_text())
    assert on_disk["S1"]["axis_map"]["yaw"] == [1, -1]


def test_file_cache_is_not_the_measurement_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    store.store_file_cache("k", {"serial_number": "S"})
    assert store.load_file_cache()["k"]["serial_number"] == "S"
    assert store.registry_path() != store.file_cache_path()
    assert store.load_registry() == {}


def test_unreadable_registry_reads_as_empty(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    store.registry_path().parent.mkdir(parents=True, exist_ok=True)
    store.registry_path().write_text("{ this is not json")
    assert store.load_registry() == {}
