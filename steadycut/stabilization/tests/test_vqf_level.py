"""Synthetics for vqf_level.level_shifts (rev2 validation a/b/c).

Frame: Map() identity (pitch=(0,1), roll=(1,1), yaw=(2,1)); VQF frame
(X,Y,Z) = (pitch, roll, gravity) columns. Body roll = rotation about
VQF-Y: gyro [0, w, 0], accel R_y(th)(0,0,g). The settled static formula
gives shift = -th; the tests assert VQF reproduces it through motion
(no hand-derived sign anywhere in the module).
(a) rigid +-10 deg oscillation: shifts track -th (corr, amplitude).
(b) coordinated turn + 1 g lateral: VQF shifts stable while the same
    formula on RAW accel (fail-old) swings ~45 deg.
(c) static 30 deg tilt: shifts sit at -30 deg (tilt is tilt).
"""
import sys

import numpy as np

sys.path.insert(0, "/source/upstream/steadycut")
from steadycut.stabilization import vqf_level as VL
from steadycut.stabilization.level import shift_from_gravity


class Map:
    pitch = (0, 1)
    roll = (1, 1)
    yaw = (2, 1)


def test_rigid_roll_tracks():
    dt, n = 0.01, 2000
    t = np.arange(n) * dt
    th = np.radians(10.0 * np.sin(2 * np.pi * 0.05 * t))
    thd = np.radians(10.0 * 2 * np.pi * 0.05 * np.cos(2 * np.pi * 0.05 * t))
    g = 9.81
    gyro = np.column_stack([np.zeros(n), np.degrees(thd), np.zeros(n)])
    accel = np.column_stack([g * np.sin(th), np.zeros(n), g * np.cos(th)])
    s, _ = VL.level_shifts(t, gyro, accel, Map())
    thd_deg = np.degrees(th)
    c = float(np.corrcoef(s[50:-50], thd_deg[50:-50])[0, 1])
    # opposes the tilt (settled-formula direction, measured -th); -0.8
    # not -1.0: VQF's bias estimator absorbs part of slow oscillations
    # (known limit, documented in vqf_level)
    assert c < -0.8, c
    assert 6.0 < float(np.percentile(np.abs(s[50:-50]), 95)) < 12.0


def test_coordinated_turn_no_bank():
    dt, n = 0.01, 2000
    t = np.arange(n) * dt
    g = 9.81
    gyro = np.column_stack([np.full(n, 1.2), np.full(n, 0.5),
                            np.full(n, 6.9)])
    rng = np.random.default_rng(0)
    # 1 g lateral in column 0 (the formula's gx): fail-old swings ~45 deg
    accel = np.column_stack([np.full(n, g) + rng.normal(0, 0.15, n),
                             rng.normal(0, 0.15, n),
                             np.full(n, g) + rng.normal(0, 0.15, n)])
    s, info = VL.level_shifts(t, gyro, accel, Map())
    naive = np.array([shift_from_gravity(accel[i], Map())
                      for i in range(n)])
    assert float(np.percentile(np.abs(s - np.median(s)), 95)) < 10.0, info
    # fail-old: the same formula on raw accel sits banked ~45 deg
    assert abs(float(np.median(naive))) > 20.0, float(np.median(naive))


def test_static_tilt_carried():
    dt, n = 0.01, 500
    t = np.arange(n) * dt
    g = 9.81
    th = np.radians(30.0)
    gyro = np.zeros((n, 3))
    accel = np.column_stack([np.full(n, g * np.sin(th)), np.zeros(n),
                             np.full(n, g * np.cos(th))])
    s, _ = VL.level_shifts(t, gyro, accel, Map())
    assert abs(float(np.median(s[100:-100])) + 30.0) < 3.0
