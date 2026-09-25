"""Tests for the gyro -> camera-path stage.

These use synthetic telemetry so they run without camera footage and, more to
the point, so the answer is known: a signal made of a slow drift plus a fast
shake lets us assert that the fast part is cancelled and the slow part is not.
"""

import numpy as np

from steadycut.ingest.cameras.profiles import AxisMap
from steadycut.stabilization.path import build_path

# A measured axis map is required input (it lives in the user's registry, not
# in the source), so these tests state the X4's measured mapping explicitly:
# yaw, pitch and roll sit on gyro columns 1, 2 and 0, all negative because the
# rear-first stack order rotates the stitched sphere 180 degrees against the
# camera axes. The synthetic signal below puts its rate in column 1, i.e. yaw.
MAP = AxisMap(yaw=(1, -1), pitch=(2, -1), roll=(0, -1))

RATE_HZ = 1000.0  # the X4's real IMU rate; the bug this guards against was
                  # deriving the filter window from the video frame rate


def _signal(duration=4.0, shake_hz=5.0, shake_deg=8.0, drift_hz=0.1, drift_deg=30.0):
    t = np.arange(0.0, duration, 1.0 / RATE_HZ)
    shake = shake_deg * np.sin(2 * np.pi * shake_hz * t)
    drift = drift_deg * np.sin(2 * np.pi * drift_hz * t)
    angle = shake + drift
    rate = np.gradient(angle, t)  # deg/s, what the gyro would report
    gyro = np.stack([np.zeros_like(rate), rate, np.zeros_like(rate)], axis=1)
    return t, gyro, shake, drift


def _amp(x, t, hz):
    """Amplitude of the component at `hz`."""
    return 2 * abs(np.sum(x * np.exp(-2j * np.pi * hz * t))) / len(x)


def test_cancels_fast_shake():
    t, gyro, shake, _ = _signal()
    path = build_path(t, gyro, cutoff_s=1.0, axis_map=MAP)
    residual = np.array([p.yaw for p in path])
    # The path is ADDED to the gyro-derived angle with this sign convention
    # (see build_path); the residual shake is therefore integral minus path.
    applied = np.cumsum(gyro[:, 1] * np.gradient(t)) - residual
    assert _amp(applied, t, 5.0) < 0.35 * _amp(shake, t, 5.0)


def test_preserves_slow_gaze():
    t, gyro, _, drift = _signal()
    path = build_path(t, gyro, cutoff_s=1.0, axis_map=MAP)
    residual = np.array([p.yaw for p in path])
    assert _amp(residual, t, 0.1) < 0.35 * _amp(drift, t, 0.1)


def test_window_follows_imu_rate_not_frame_rate():
    """Regression: a frame-rate-derived window (30 samples = 0.03 s) leaves the
    5 Hz shake untouched, because it is no longer 'fast' at that scale."""
    t, gyro, shake, _ = _signal()
    path = build_path(t, gyro, cutoff_s=1.0, axis_map=MAP)
    residual = np.array([p.yaw for p in path])
    # The path is ADDED to the gyro-derived angle with this sign convention
    # (see build_path); the residual shake is therefore integral minus path.
    applied = np.cumsum(gyro[:, 1] * np.gradient(t)) - residual
    assert _amp(applied, t, 5.0) < 0.2 * _amp(shake, t, 5.0)


def test_gain_scales_the_correction():
    t, gyro, _, _ = _signal()
    full = np.array([p.yaw for p in build_path(t, gyro, cutoff_s=1.0, gain=1.0, axis_map=MAP)])
    half = np.array([p.yaw for p in build_path(t, gyro, cutoff_s=1.0, gain=0.5, axis_map=MAP)])
    assert np.allclose(half, full / 2, atol=1e-6)


def test_infer_mapping_recovers_a_known_mapping():
    """The axis inference must recover a mapping that was constructed in.

    Synthetic motion is built from three independent signals, each placed in a
    different gyro column, so the answer is known in advance.
    """
    from steadycut.calibration import calibrate

    rng = np.random.default_rng(0)
    n = 300
    signals = [rng.standard_normal(n) for _ in range(3)]
    # yaw in column 1, pitch in column 2, roll in column 0 - the X4 layout
    motion = np.column_stack([signals[1], signals[2]])
    roll = signals[0]
    gyro = np.column_stack([signals[0], signals[1], signals[2]])

    best, _ = calibrate.infer_mapping(motion, roll, gyro)
    assert best["dx (yaw)"] == 1
    assert best["dy (pitch)"] == 2
    assert best["tilt (roll)"] == 0
