"""Synthetics for vqf_level.level_shifts (rev2 validation a/b/c).

Fixtures are RAW-sensor terms with the real X4 map (yaw=(1,-1),
pitch=(2,-1), roll=(0,-1)): gravity lives in raw column 1 (reads +g at
rest, as on footage). Identity maps hide the frame logic under test.
(a) rigid +-10 deg oscillation about raw-x: shifts track the tilt
    through the settled formula (corr + amplitude pin the direction).
(b) yawing at 7 deg/s for 20 s with 1 g body-fixed lateral: the rate
    gate fires and holds pre-turn level while the same formula on RAW
    accel (fail-old) sits banked ~45 deg.
(c) static 30 deg tilt about raw-x: shifts sit at the formula value
    (tilt is tilt; head-look is not special).
"""
import sys

import numpy as np

sys.path.insert(0, "/source/upstream/steadycut")
from steadycut.stabilization import vqf_level as VL
from steadycut.stabilization.level import shift_from_gravity


class Map:
    yaw = (1, -1)
    pitch = (2, -1)
    roll = (0, -1)


def test_rigid_roll_tracks():
    dt, n = 0.01, 2000
    t = np.arange(n) * dt
    th = np.radians(10.0 * np.sin(2 * np.pi * 0.05 * t))
    thd = np.radians(10.0 * 2 * np.pi * 0.05 * np.cos(2 * np.pi * 0.05 * t))
    g = 9.81
    gyro = np.column_stack([np.degrees(thd), np.zeros(n), np.zeros(n)])
    accel = np.column_stack([np.zeros(n), g * np.cos(th), g * np.sin(th)])
    s, _ = VL.level_shifts(t, gyro, accel, Map())
    thd_deg = np.degrees(th)
    c = float(np.corrcoef(s[50:-50], thd_deg[50:-50])[0, 1])
    # direction presence at full amplitude (sign asserted below
    # against the static formula — no new convention)
    assert abs(c) > 0.8, c
    assert 6.0 < float(np.percentile(np.abs(s[50:-50]), 95)) < 12.0
    # same sign as the static formula on the same tilt (no new convention)
    st = np.array([shift_from_gravity(
        np.array([0.0, g * np.cos(a), g * np.sin(a)]), Map())
        for a in th[50:-50:50]])
    assert np.sign(np.median(s[50:-50:50])) == np.sign(np.median(st))


def test_coordinated_turn_no_bank():
    dt, n = 0.01, 2000
    t = np.arange(n) * dt
    g = 9.81
    # yawing at 7 deg/s for 20 s (sustained: the rate gate must fire).
    # Yaw is the vendor column 1 here only as filler; the gate reads the
    # VQF yaw rate, which spins under body yaw motion.
    gyro = np.column_stack([np.zeros(n), np.full(n, 7.0), np.zeros(n)])
    rng = np.random.default_rng(0)
    accel = np.column_stack([rng.normal(0, 0.15, n),
                             np.full(n, g) + rng.normal(0, 0.15, n),
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
    accel = np.column_stack([np.zeros(n), np.full(n, g * np.cos(th)),
                             np.full(n, g * np.sin(th))])
    s, _ = VL.level_shifts(t, gyro, accel, Map())
    st = shift_from_gravity(
        np.array([0.0, g * np.cos(th), g * np.sin(th)]), Map())
    assert abs(float(np.median(s[100:-100])) - st) < 3.0
