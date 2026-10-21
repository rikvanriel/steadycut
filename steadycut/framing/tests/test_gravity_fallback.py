"""Gravity fallback: level pitch when the ladder finds no wearer mass.

Fail-old/pass-new anchor (real footage, not in this file): May19 t=30
refused before ("never reached its target: 0.59..0.79 against 0.90") and
renders after via this fallback. These units pin the mechanism: sign,
gate, window, and failure modes.
"""
import numpy as np

from steadycut.framing import ladder
from steadycut.ingest.cameras.profiles import AxisMap

MAP = AxisMap(yaw=(1, -1), pitch=(2, -1), roll=(0, -1))
# cols order in tilt_from_gravity is [pitch, yaw, roll] = raw[2, 1, 0],
# so raw[0] is body-z (forward), raw[1] body-y (up), raw[2] body-x.


class _IMU:
    def __init__(self, accel, rate=100.0):
        self.accel = np.asarray(accel, dtype=float)
        self.time_s = np.arange(len(accel)) / rate


class _Profile:
    axis_map = MAP


def _imu_like(monkeypatch, accel, rate=100.0):
    import steadycut.ingest.telemetry as T

    monkeypatch.setattr(T, "read_telemetry",
                        lambda *a, **k: _IMU(accel, rate))


def _level_accel(tilt_deg=0.0, n=400):
    g = 9.81
    t = np.radians(tilt_deg)
    raw = [g * np.sin(t), g * np.cos(t), 0.0]
    return [raw] * n


def test_level_camera_commands_zero_pitch(monkeypatch) -> None:
    _imu_like(monkeypatch, _level_accel(0.0))
    pitch, info = ladder.gravity_fallback("src", 0.0, _Profile())
    assert info["usable"] is True
    assert pitch == 0.0


def test_down_tilt_commands_negative_pitch(monkeypatch) -> None:
    # Camera pitched down 10 deg reads positive tilt; framing counts down
    # as negative, so the fallback negates.
    _imu_like(monkeypatch, _level_accel(10.0))
    pitch, info = ladder.gravity_fallback("src", 0.0, _Profile())
    assert info["usable"] is True
    assert pitch is not None and pitch < 0
    assert abs(pitch + 10.0) < 0.01


def test_sustained_maneuver_is_unusable(monkeypatch) -> None:
    _imu_like(monkeypatch, [[1.0, 2.0, 0.5]] * 400)  # |g| ~ 2.3
    pitch, info = ladder.gravity_fallback("src", 0.0, _Profile())
    assert pitch is None and info["usable"] is False
    assert "|g|" in info["reason"]


def test_empty_window_is_unusable(monkeypatch) -> None:
    _imu_like(monkeypatch, _level_accel(0.0, n=10), rate=100.0)
    pitch, info = ladder.gravity_fallback("src", 50.0, _Profile())
    assert pitch is None and info["usable"] is False


def test_missing_telemetry_is_unusable(monkeypatch) -> None:
    import steadycut.ingest.telemetry as T

    def _boom(*a, **k):
        raise RuntimeError("no trailer")

    monkeypatch.setattr(T, "read_telemetry", _boom)
    pitch, info = ladder.gravity_fallback("src", 0.0, _Profile())
    assert pitch is None and info["usable"] is False
