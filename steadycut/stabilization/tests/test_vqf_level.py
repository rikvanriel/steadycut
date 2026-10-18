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
import numpy as np

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
    s, _ps, _ = VL.level_shifts(t, gyro, accel, Map())
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
    s, _ps, info = VL.level_shifts(t, gyro, accel, Map())
    naive = np.array([shift_from_gravity(accel[i], Map())
                      for i in range(n)])
    assert float(np.percentile(np.abs(s - np.median(s)), 95)) < 10.0, info
    # the gate did work (not VQF alone): it fired, and the ungated raw
    # spread exceeds the held output (review R2)
    assert info["n_turns"] >= 1, info
    assert info["raw_p95"] > float(
        np.percentile(np.abs(s - np.median(s)), 95)), info
    # fail-old: the same formula on raw accel sits banked ~45 deg
    assert abs(float(np.median(naive))) > 20.0, float(np.median(naive))


def test_bridge_spin_about_vertical():
    # (d) rigid spin about WORLD vertical, fixed 15-deg body tilt, calm
    # both sides: the roll-slot shift swings through the turn while the
    # world reference is fixed. The bridge must track it (<4 deg);
    # the old constant hold fails by the swing amplitude.
    dt = 0.01
    calm_s, turn_s = 3.0, 12.0
    n_c, n_t = int(calm_s / dt), int(turn_s / dt)
    n = 2 * n_c + n_t
    t = np.arange(n) * dt
    yaw_rate = 10.0  # deg/s about world vertical through the turn
    psi = np.zeros(n)
    psi[n_c:n_c + n_t] = np.radians(yaw_rate) * np.arange(n_t) * dt
    tilt = np.radians(15.0)
    Rx = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)],
                   [0, np.sin(tilt), np.cos(tilt)]])
    up0 = Rx @ np.array([0.0, 0.0, 1.0])
    # spin psi about world z: the tilted up-vector rotates by -psi in body
    up_body = np.array([[up0[0] * np.cos(p) + up0[1] * np.sin(p),
                         -up0[0] * np.sin(p) + up0[1] * np.cos(p), up0[2]]
                        for p in psi])
    g = 9.81
    # up-positive convention (pipeline-wide: rest raw[1] = +g); raw via
    # the code's post-fix frame body = (-raw[2], -raw[0], +raw[1])
    acc_body = g * up_body
    # raw columns via the X4 signed permutation (Map signs pre-det-fix;
    # level_shifts derives the handedness itself)
    gyro = np.zeros((n, 3))
    accel = np.zeros((n, 3))
    w_body = np.degrees(np.radians(yaw_rate)) * up_body
    w_body[:n_c] = 0.0
    w_body[n_c + n_t:] = 0.0
    # raw columns via the code's post-fix frame:
    # body = (-raw[2], -raw[0], +raw[1])
    gyro[:, 2] = -w_body[:, 0]
    gyro[:, 0] = -w_body[:, 1]
    gyro[:, 1] = +w_body[:, 2]
    accel[:, 2] = -acc_body[:, 0]
    accel[:, 0] = -acc_body[:, 1]
    accel[:, 1] = +acc_body[:, 2]
    s, _ps, info = VL.level_shifts(t, gyro, accel, Map())
    true = np.array([shift_from_gravity(accel[i], Map())
                     for i in range(n)])
    mid = slice(n_c, n_c + n_t)
    err = float(np.median(np.abs(s[mid] - true[mid])))
    assert info["n_bridged"] >= 1, info
    assert err < 4.0, (err, info)
    # fail-old: the constant hold misses the swing
    hold_err = float(np.median(np.abs(
        np.median(s[:n_c]) - true[mid])))
    assert hold_err > 2 * err, (hold_err, err)


def test_slalom_cycle_cancels_lateral():
    # (f) two-leg slalom, yaw +-20 deg/s, lateral +-3 m/s^2 alternating
    # with the legs, fixed 15-deg tilt, calm ends: VQF absorbs each
    # leg's lateral as gravity (per-leg anchors lie), but the cycle
    # mean cancels it. Shifts must track TRUE body-gravity (<6 deg);
    # per-leg innov anchors fail by the lateral tilt (~17 deg).
    dt = 0.01
    n_c, n_t = 300, 1000
    n = 2 * n_c + 2 * n_t
    t = np.arange(n) * dt
    yaw_rate = 20.0
    psi = np.zeros(n)
    psi[n_c:n_c + n_t] = np.radians(yaw_rate) * np.arange(n_t) * dt
    psi[n_c + n_t:n_c + 2 * n_t] = (
        np.radians(yaw_rate) * n_t * dt
        - np.radians(yaw_rate) * np.arange(n_t) * dt)
    tilt = np.radians(15.0)
    Rx = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)],
                   [0, np.sin(tilt), np.cos(tilt)]])
    up0 = Rx @ np.array([0.0, 0.0, 1.0])
    up_body = np.array([[up0[0] * np.cos(p) + up0[1] * np.sin(p),
                         -up0[0] * np.sin(p) + up0[1] * np.cos(p), up0[2]]
                        for p in psi])
    g = 9.81
    lat = np.zeros(n)
    lat[n_c:n_c + n_t] = 3.0
    lat[n_c + n_t:n_c + 2 * n_t] = -3.0
    # lateral along body-x (centripetal, alternating with the legs)
    acc_body = g * up_body + np.column_stack(
        [lat, np.zeros(n), np.zeros(n)])
    w_body = np.zeros((n, 3))
    dpsi = np.gradient(psi, dt)
    for i in range(3):
        w_body[:, i] = dpsi * up_body[:, i]
    w_body[:n_c] = 0.0
    w_body[n_c + 2 * n_t:] = 0.0
    class Map:
        yaw = (1, -1)
        pitch = (2, -1)
        roll = (0, -1)
    gyro = np.zeros((n, 3))
    accel = np.zeros((n, 3))
    gyro[:, 2] = -np.degrees(w_body[:, 0])
    gyro[:, 0] = -np.degrees(w_body[:, 1])
    gyro[:, 1] = +np.degrees(w_body[:, 2])
    accel[:, 2] = -acc_body[:, 0]
    accel[:, 0] = -acc_body[:, 1]
    accel[:, 1] = +acc_body[:, 2]
    s, _ps, info = VL.level_shifts(t, gyro, accel, Map())
    # truth: lateral-free gravity through the post-fix frame
    # body = (-raw[2], -raw[0], +raw[1]) -> raw = (-b1, +b2, -b0)
    true = np.array([shift_from_gravity(
        np.array([-g * u[1], g * u[2], -g * u[0]]), Map())
        for u in up_body])
    mid = slice(n_c, n_c + 2 * n_t)
    err = float(np.median(np.abs(s[mid] - true[mid])))
    # flip anchors cut the lateral-tilt error (~17 deg per leg) to
    # ~6 deg; the rest is gyro-carry drift plus flip-instant transients
    assert err < 8.0, (err, info)


def test_pitch_axis_complements_roll():
    # (g) full-tilt: static 20-deg pitch-forward tilt reads ~20 deg in
    # the pitch channel and ~0 in roll; roll formula untouched.
    from steadycut.stabilization.level import tilt_from_gravity
    g = 9.81
    p, r = tilt_from_gravity(np.array(
        [g * np.sin(np.radians(20.0)), g * np.cos(np.radians(20.0)),
         0.0]), Map())
    assert abs(p - 20.0) < 1.0, (p, r)
    assert abs(r) < 1.0, (p, r)


def test_up_convention_matches_static():
    # (e) z-up pin: on calm tilted data the VQF-derived body-up must
    # agree with the static accel direction (validates _body_up_from_R
    # including the det-derived handedness on the X4 map).
    dt, n = 0.01, 400
    t = np.arange(n) * dt
    tilt = np.radians(20.0)
    g = 9.81
    gyro = np.zeros((n, 3))
    accel = np.column_stack([np.zeros(n),
                             np.full(n, g * np.cos(tilt)),
                             np.full(n, g * np.sin(tilt))])
    s, _ps, _ = VL.level_shifts(t, gyro, accel, Map())
    assert abs(float(np.median(s)) - float(shift_from_gravity(
        np.array([0.0, g * np.cos(tilt), g * np.sin(tilt)]),
        Map()))) < 1.0
    # B1: path times are clip-relative; interpolating them against
    # absolute IMU times parks every point before xp[0] (one constant
    # from before the window). resample_shifts subtracts start.
    imu_t = np.arange(0.0, 20.0, 0.01) + 240.0
    sh = np.sin(imu_t)  # varying: a constant output proves the bug
    path_t = np.arange(-15.0, 30.0, 1.0)  # clip-relative, start=240
    got = VL.resample_shifts(path_t, imu_t, sh, 240.0)
    assert float(np.std(got)) > 0.1, "constant: clock mismatch"
    # and it tracks the right values, not just varies
    want = np.interp(path_t, imu_t - 240.0, sh)
    assert np.allclose(got, want)


def test_static_tilt_carried():
    dt, n = 0.01, 500
    t = np.arange(n) * dt
    g = 9.81
    th = np.radians(30.0)
    gyro = np.zeros((n, 3))
    accel = np.column_stack([np.zeros(n), np.full(n, g * np.cos(th)),
                             np.full(n, g * np.sin(th))])
    s, _ps, _ = VL.level_shifts(t, gyro, accel, Map())
    st = shift_from_gravity(
        np.array([0.0, g * np.cos(th), g * np.sin(th)]), Map())
    assert abs(float(np.median(s[100:-100])) - st) < 3.0
