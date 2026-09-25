"""Measurement lifespan: once per body, and re-taken when firmware changes.

A body is measured once and every later file from it reuses that measurement.
Firmware is the exception: the axis convention (and the per-unit calibration
in the trailer) can move with a firmware update, so a measurement records the
firmware it was taken under and a mismatch triggers exactly one re-measurement.
"""
import pytest

from steadycut.ingest.cameras import registry as R

X4_AXIS = {"yaw": [1, -1], "pitch": [2, -1], "roll": [0, -1]}


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path


def test_same_serial_reuses_one_measurement_across_files(cfg) -> None:
    R.record_measured("S1", model="Insta360 X4", fw="v1.9.21",
                      axis_map=X4_AXIS)
    # Three different files from the same body: one stored measurement serves
    # all of them, and nothing is keyed by file path.
    for _ in range(3):
        assert R.axis_map_for("S1", "v1.9.21") is not None
    assert len(R.store.load_registry()) == 1


def test_firmware_change_invalidates_the_measurement(cfg) -> None:
    R.record_measured("S1", model="Insta360 X4", fw="v1.9.21",
                      axis_map=X4_AXIS)
    assert R.axis_map_for("S1", "v1.9.21") is not None
    # A file shot after a firmware update must not reuse the old convention.
    assert R.axis_map_for("S1", "v1.10.0") is None
    reason = R.stale_reason("S1", "v1.10.0")
    assert reason and "firmware changed" in reason


def test_unknown_firmware_still_uses_the_measurement(cfg) -> None:
    """A file whose fw could not be read must not force a re-measurement."""
    R.record_measured("S1", fw="v1.9.21", axis_map=X4_AXIS)
    assert R.axis_map_for("S1", None) is not None
    assert R.stale_reason("S1", None) is None


def test_remeasurement_updates_the_stored_firmware(cfg) -> None:
    R.record_measured("S1", fw="v1.9.21", axis_map=X4_AXIS)
    assert R.axis_map_for("S1", "v1.10.0") is None
    R.record_measured("S1", fw="v1.10.0", axis_map=X4_AXIS)
    assert R.axis_map_for("S1", "v1.10.0") is not None
