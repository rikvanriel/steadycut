"""Pins for gravity levelling: formula, application, and the no-op case."""
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, "/source/upstream/steadycut")
from steadycut.ingest.cameras.profiles import AxisMap                # noqa: E402
from steadycut.stabilization.level import (                          # noqa: E402
    apply_shift, measure_shift, shift_from_gravity)

AXIS = AxisMap(yaw=(1, -1), pitch=(2, -1), roll=(0, -1))
# body frame = raw cols [pitch, yaw, roll] = (right, up, forward)


def test_level_mount_roll():
    # Camera rolled +90 deg (right side down): gravity reads -1g on body-x.
    g = np.zeros(3)
    g[AXIS.pitch[0]] = -9.81
    g[AXIS.yaw[0]] = 0.0
    assert shift_from_gravity(g, AXIS) == 90.0


def test_upright_mount_is_noop_shift():
    g = np.zeros(3)
    g[AXIS.yaw[0]] = 9.81
    assert shift_from_gravity(g, AXIS) == 0.0


def test_inverted_mount_reads_180():
    g = np.zeros(3)
    g[AXIS.yaw[0]] = -9.81
    assert abs(shift_from_gravity(g, AXIS)) == 180.0


def test_measure_shift_uses_clip_start():
    t = np.arange(0, 100.0, 0.01)
    g0 = np.zeros(3)
    g0[AXIS.yaw[0]] = 9.81
    imu = SimpleNamespace(time_s=t, accel=np.tile(g0, (len(t), 1)))
    got = measure_shift(imu, 10.0, 60.0, AXIS)
    assert got["shift_deg"] == 0.0
    assert got["span_s"] == 30.0
    assert got["n"] == 3000


def test_apply_shift_adds_to_every_roll():
    from steadycut.render.reframe import ControlPoint
    pts = [ControlPoint(t=0.0, roll=1.0), ControlPoint(t=0.1, roll=-2.0)]
    out = apply_shift(pts, 66.6)
    assert [p.roll for p in out] == [67.6, 64.6]
    # originals untouched
    assert [p.roll for p in pts] == [1.0, -2.0]


def test_apply_zero_shift_is_identity():
    from steadycut.render.reframe import ControlPoint
    pts = [ControlPoint(t=0.0, roll=1.0)]
    assert apply_shift(pts, 0.0)[0].roll == 1.0
