"""End-to-end combination: profile -> ClipSpec -> gyro path -> command file.

The camera profile must actually steer the numeric path: with a permuted axis
map the same telemetry lands in DIFFERENT correction channels. That is the
property that makes a second camera's calibration pure data.
"""
from dataclasses import replace

import numpy as np

from steadycut.ingest.cameras.insta360_x4 import PROFILE as X4
from steadycut.ingest.cameras.profiles import AxisMap

# A profile as the registry hands it over: the model profile with the measured
# axis map for this body attached. The map is no longer a source-tree default,
# so a test that exercises the path must supply one.
X4M = replace(X4, axis_map=AxisMap(yaw=(1, -1), pitch=(2, -1), roll=(0, -1)))
from steadycut.ingest.telemetry import Telemetry
from steadycut.core.pipeline import ClipSpec, build_gyro_path, gyro_pitch_per_frame
from steadycut.render.reframe import build_command_file


def _telemetry(n_s=10.0, rate=1000.0) -> Telemetry:
    """Yaw-rate shake in column 1, a weaker signal in column 0 (the X4's
    yaw and roll columns); column 2 silent."""
    t = np.arange(0.0, n_s, 1.0 / rate)
    gyro = np.zeros((len(t), 3))
    gyro[:, 1] = 8.0 * np.sin(2 * np.pi * 5.0 * t)
    gyro[:, 0] = 3.0 * np.sin(2 * np.pi * 3.0 * t)
    accel = np.tile([3.3, 8.71, 2.29], (len(t), 1))
    return Telemetry(time_s=t, gyro=gyro, accel=accel, model="test")


def _patch_telemetry(monkeypatch):
    import steadycut.ingest.telemetry as T
    monkeypatch.setattr(T, "read_telemetry",
                        lambda path, profile=None: _telemetry())


def test_profile_path_and_commands_agree(monkeypatch) -> None:
    _patch_telemetry(monkeypatch)
    spec = ClipSpec(source="fake.insv", start=1.0, duration=2.0,
                    framing_pitch=-12.0, profile=X4M)
    pts = build_gyro_path(spec)
    assert len(pts) > 10
    # Pitch carries the framing offset; the correction itself rides yaw.
    median_pitch = float(np.median([p.pitch for p in pts]))
    assert abs(median_pitch - (-12.0)) < 2.0
    exc_yaw = max(abs(p.yaw) for p in pts)
    exc_pitch = max(abs(p.pitch + 12.0) for p in pts)
    assert exc_yaw > exc_pitch
    # And the path converts to a real sendcmd file.
    txt = build_command_file(pts, fps=30.0)
    assert "v360 yaw" in txt and "v360 h_fov" in txt


def test_permuted_map_moves_the_channels(monkeypatch) -> None:
    _patch_telemetry(monkeypatch)
    spec = ClipSpec(source="fake.insv", start=1.0, duration=2.0,
                    framing_pitch=-12.0, profile=X4M)
    pts_x4 = build_gyro_path(spec)
    # Swap the yaw and roll COLUMNS: the strong signal must move from the yaw
    # channel to the roll channel.
    variant = replace(X4M, axis_map=AxisMap(yaw=(0, -1), pitch=(2, -1),
                                            roll=(1, -1)))
    pts_swap = build_gyro_path(replace(spec, profile=variant))
    assert max(abs(p.yaw) for p in pts_swap) < max(abs(p.yaw) for p in pts_x4)
    assert max(abs(p.roll) for p in pts_swap) > max(abs(p.roll) for p in pts_x4)


def test_gyro_pitch_per_frame_follows_the_map(monkeypatch) -> None:
    _patch_telemetry(monkeypatch)
    # X4 pitch column (2) is silent: means ~0. A map naming column 0 sees the
    # 3 deg/s signal: clearly non-zero.
    x4 = gyro_pitch_per_frame("fake.insv", 0.0, 100, fps=100.0,
                              axis_map=X4M.axis_map)
    assert np.nanmax(np.abs(x4)) < 0.05
    variant = replace(X4M, axis_map=AxisMap(yaw=(1, -1), pitch=(0, -1),
                                            roll=(2, -1)))
    other = gyro_pitch_per_frame("fake.insv", 0.0, 100, fps=100.0,
                                 axis_map=variant.axis_map)
    assert np.nanmax(np.abs(other)) > 0.02


def test_a_profile_without_a_measured_map_is_refused(monkeypatch) -> None:
    """Measure-first end to end: no map means no path, never another camera's."""
    import pytest as _pytest
    _patch_telemetry(monkeypatch)
    spec = ClipSpec(source="fake.insv", start=1.0, duration=2.0,
                    framing_pitch=-12.0, profile=X4)   # X4.axis_map is None
    with _pytest.raises(ValueError, match="no measured axis map"):
        build_gyro_path(spec)
