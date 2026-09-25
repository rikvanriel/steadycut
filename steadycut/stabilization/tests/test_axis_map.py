"""The axis map is data: golden equality for the measured X4 map, and generic
behaviour for any measured map.

The map is REQUIRED input now (it is a per-body measurement stored in the
user's registry, not a source-tree default), so the tests state it explicitly.
Golden values were captured from the pre-refactor implementation (hardcoded
reorder [2, 1, 0] and literal sign negations) on a deterministic analytic
fixture, under exactly this layout. The refactor must reproduce them
BIT-FOR-BIT: same operations, same order, so the sums and spot values compare
equal to within float noise (1e-9; numpy/BLAS builds differ in the last ulps,
routing errors cannot).
"""
import numpy as np
import pytest

from steadycut.ingest.cameras.profiles import AxisMap
from steadycut.stabilization.orientation import correction_axes
from steadycut.stabilization.path import build_path, build_path_quat

# The map measured on the rider's X4 body: yaw, pitch and roll on gyro columns
# 1, 2 and 0, all negative because the rear-first stack order rotates the
# stitched sphere 180 degrees against the camera axes. Stored per serial in the
# user's registry; the golden numbers below were captured under it.
X4 = AxisMap(yaw=(1, -1), pitch=(2, -1), roll=(0, -1))

N = 3000
T = np.arange(N) / 1000.0
GYRO = np.column_stack([
    10.0 * np.sin(2 * np.pi * 3.0 * T),        # column 0
    7.0 * np.sin(2 * np.pi * 5.0 * T + 0.3),   # column 1
    4.0 * np.cos(2 * np.pi * 7.0 * T),         # column 2
])

# Captured from the hardcoded implementation before the refactor.
QUAT_SUMS = (4.503166286702077, 0.8211782363205391, 6.243354241474243)
QUAT_SPOTS = (-0.09097025584126757, 0.1507189516160126,      # yaw  0, 1500
              -0.016589042595096287, -0.002082901986082414,  # pitch 0, 1500
              -0.1261239882818161, 0.2495059087294909)       # roll 0, 1500
PATH_SUMS = (0.0371761699812353, 5.93275428784068e-16, 0.03360644616117212)
PATH_SPOTS = (-0.012081059984453954, 0.008401053208413833,   # yaw 0, 1500
              -0.00015308934795236584,                        # pitch 1500
              0.007653438785930611)                           # roll 1500


def test_x4_map_reproduces_golden_exactly() -> None:
    y, p, r = correction_axes(T, GYRO, 0.05, axis_map=X4)
    assert (y.sum(), p.sum(), r.sum()) == QUAT_SUMS
    assert (y[0], y[1500], p[0], p[1500], r[0], r[1500]) == QUAT_SPOTS


def test_no_map_is_refused_rather_than_defaulted() -> None:
    """Measure-first: a caller with no map must supply one, never inherit."""
    with pytest.raises(ValueError, match="no measured axis map"):
        correction_axes(T, GYRO, 0.05)
    with pytest.raises(ValueError, match="no measured axis map"):
        build_path(T, GYRO, None, cutoff_s=0.05)
    with pytest.raises(ValueError, match="no measured axis map"):
        build_path_quat(T, GYRO, corner_hz=0.5)


def test_sign_flip_flips_every_channel() -> None:
    x4 = AxisMap(yaw=(1, -1), pitch=(2, -1), roll=(0, -1))
    pos = AxisMap(yaw=(1, 1), pitch=(2, 1), roll=(0, 1))
    neg = correction_axes(T, GYRO, 0.05, axis_map=x4)
    flip = correction_axes(T, GYRO, 0.05, axis_map=pos)
    for n, f in zip(neg, flip):
        assert np.allclose(n, -f, atol=1e-12)


def test_map_follows_a_permuted_layout() -> None:
    """A DIFFERENT measured layout must move the channels with the columns.

    Motion is injected per column; with a map claiming column 2 is yaw, the
    yaw output must track the column-2 signal, not the column-1 one. This is
    what makes a second camera's calibration expressible as pure data.
    """
    # column 2 carries a 5 Hz oscillation; columns 0 and 1 are flat.
    g = np.zeros((N, 3))
    g[:, 2] = GYRO[:, 1]
    am = AxisMap(yaw=(2, 1), pitch=(0, 1), roll=(1, 1))
    y, p, r = correction_axes(T, g, 0.05, axis_map=am)
    # The correction of a single live axis must be far larger than the
    # channels the map routes to flat columns.
    assert np.abs(y).max() > 10 * max(np.abs(p).max(), np.abs(r).max())


def test_build_path_routes_columns_from_the_map() -> None:
    x4 = AxisMap(yaw=(1, -1), pitch=(2, -1), roll=(0, -1))
    pts = build_path(T, GYRO, None, cutoff_s=0.05, axis_map=x4)
    # Sums accumulate float ulps that differ between numpy/BLAS builds, so the
    # golden values are compared with a tolerance far below any routing error
    # (a mis-routed axis changes these by O(1), not 1e-9).
    assert (sum(q.yaw for q in pts), sum(q.pitch for q in pts),
            sum(q.roll for q in pts)) == pytest.approx(PATH_SUMS, abs=1e-9)
    assert (pts[0].yaw, pts[1500].yaw, pts[1500].pitch,
            pts[1500].roll) == pytest.approx(PATH_SPOTS, abs=1e-9)


def test_build_path_quat_threads_the_map() -> None:
    """The map is the only source of routing, and it is deterministic."""
    a = build_path_quat(T, GYRO, corner_hz=0.5, axis_map=X4)
    b = build_path_quat(T, GYRO, corner_hz=0.5, axis_map=X4)
    assert [(q.yaw, q.pitch, q.roll) for q in a] == [(q.yaw, q.pitch, q.roll)
                                                     for q in b]
    # A permuted map must route differently: same telemetry, other channels.
    permuted = build_path_quat(T, GYRO, corner_hz=0.5,
                               axis_map=AxisMap(yaw=(2, -1), pitch=(1, -1),
                                                roll=(0, -1)))
    assert [q.yaw for q in permuted] != [q.yaw for q in a]
