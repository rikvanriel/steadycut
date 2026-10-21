"""Stride pins: stride=1 is the old behaviour exactly; stride>1 shortens.

The bench (recorded, real footage) is the peak-holds validation: stride 5
reads -60ms vs -50ms and -30ms vs -20ms on the frozen windows. These units
pin the mechanism: identical bins at stride 1, fewer rows at stride 2.
"""
import numpy as np

from steadycut.core import pipeline as P


class _IMU:
    def __init__(self, rate=1000.0, n=3000):
        self.time_s = np.arange(n) / rate
        self.gyro = np.zeros((n, 3))
        self.gyro[:, 0] = 1.0  # constant pitch rate


def _fake_telemetry(monkeypatch, imu):
    import steadycut.ingest.telemetry as T

    monkeypatch.setattr(T, "read_telemetry", lambda *a, **k: imu)


def test_stride1_bins_match_hand_computed(monkeypatch) -> None:
    _fake_telemetry(monkeypatch, _IMU())
    axis_map = {"pitch": (0, 1)}
    got = P.gyro_pitch_per_frame("src", 1.0, 4, fps=100.0,
                                 axis_map=type("A", (), axis_map)(),
                                 stride=1)
    # constant 1.0 rate over full bins -> 1.0/100 per frame, every bin
    assert np.allclose(got, 0.01)


def test_stride_widens_bins_but_same_rate(monkeypatch) -> None:
    _fake_telemetry(monkeypatch, _IMU())
    axis_map = type("A", (), {"pitch": (0, 1)})()
    a = P.gyro_pitch_per_frame("src", 1.0, 20, fps=100.0,
                               axis_map=axis_map, stride=1)
    b = P.gyro_pitch_per_frame("src", 1.0, 4, fps=100.0,
                               axis_map=axis_map, stride=5)
    assert np.allclose(a[:20:5], b, equal_nan=True)


def test_strided_track_returns_fewer_rows(tmp_path) -> None:
    import cv2

    rng = np.random.default_rng(0)
    out = str(tmp_path / "noise.mp4")
    w = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    base = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
    for i in range(12):
        w.write(np.roll(base, 2 * i, axis=1))
    w.release()
    from steadycut.stabilization.far_track import track

    d1, _ = track(out, stride=1)
    d2, _ = track(out, stride=2)
    assert len(d1) == 11  # one row per interval, always
    assert len(d2) == 5
