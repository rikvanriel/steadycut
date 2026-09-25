"""Known-answer tests for the axis-map inference (no renders needed).

The inference is the part of first-use discovery that can be wrong silently:
a wrong column or a wrong sign still produces a plausible-looking map, and the
stabiliser then applies it. So it is tested against synthetic motion built
from a known layout.
"""
import numpy as np

from steadycut.calibration import discover as D


def _case(sy: float = -1.0, sp: float = -1.0, sr: float = -1.0):
    """Motion built so yaw/pitch/roll sit on columns 1/2/0 of the gyro.

    The signs default to negative, which is the X4's measured relationship:
    every channel correlates negatively with its gyro column, because the
    rear-first stack order rotates the stitched sphere 180 degrees against the
    camera axes.
    """
    rng = np.random.default_rng(0)
    gyro = rng.standard_normal((200, 3))
    motion = np.column_stack([sy * gyro[:, 1], sp * gyro[:, 2]])
    roll = sr * gyro[:, 0]
    return motion, roll, gyro


def test_infer_map_recovers_columns_and_signs() -> None:
    motion, roll, gyro = _case()
    axis_map, corr = D.infer_map(motion, roll, gyro)
    assert axis_map["yaw"][0] == 1 and axis_map["yaw"][1] == -1
    assert axis_map["pitch"][0] == 2 and axis_map["pitch"][1] == -1
    assert axis_map["roll"][0] == 0 and axis_map["roll"][1] == -1
    assert abs(corr["yaw"][1]) > 0.9, "the winning column must be decisive"


def test_sign_follows_the_measured_correlation() -> None:
    """The sign is measured, never assumed: a positive correlation gives +1."""
    motion, roll, gyro = _case(sy=1.0)
    axis_map, _ = D.infer_map(motion, roll, gyro)
    assert axis_map["yaw"] == [1, 1]


def test_a_motionless_window_is_refused() -> None:
    """No motion means no correlation, and no correlation means no map."""
    gyro = np.zeros((100, 3))
    axis_map, corr = D.infer_map(np.zeros((100, 2)), np.zeros(100), gyro)
    assert D._weak(axis_map, corr) is not None
    assert all(np.isfinite(r) for rs in corr.values() for r in rs)


def test_a_weak_channel_is_refused() -> None:
    axis_map = {"yaw": [1, -1], "pitch": [2, -1], "roll": [0, -1]}
    corr = {"yaw": [0.1, -0.9, 0.1], "pitch": [0.1, 0.2, -0.3],   # pitch weak
            "roll": [-0.9, 0.1, 0.1]}
    reason = D._weak(axis_map, corr)
    assert reason and "pitch" in reason


def test_two_channels_on_one_column_is_refused() -> None:
    axis_map = {"yaw": [1, -1], "pitch": [1, -1], "roll": [0, -1]}
    corr = {"yaw": [0.1, -0.9, 0.1], "pitch": [0.1, -0.9, 0.1],
            "roll": [-0.9, 0.1, 0.1]}
    reason = D._weak(axis_map, corr)
    assert reason and "same gyro column" in reason


def test_strength_is_the_weakest_channel() -> None:
    axis_map = {"yaw": [1, -1], "pitch": [2, -1], "roll": [0, -1]}
    corr = {"yaw": [0.1, -0.95, 0.1], "pitch": [0.1, 0.2, -0.6],
            "roll": [-0.9, 0.1, 0.1]}
    assert abs(D._strength(axis_map, corr) - 0.6) < 1e-12
