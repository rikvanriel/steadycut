"""Time-varying roll leveling from VQF (gravity-leveling rev2, step 2/3).

Split of duties (axis-swamp avoidance):
- VQF supplies the ATTITUDE (gyro-carried, sees through turn
  centripetal via earth-frame lowpass + offline forward-backward).
- `shift_from_gravity` (eye-settled on Jan +66.6 / Feb -104, NOT
  derived) supplies the CONVENTION: per-sample shift from VQF-predicted
  body gravity, same formula as the static path. On straights this
  reduces to the static shift by construction; in turns the prediction
  ( unlike raw accel ) carries no centripetal.
- Freeze through sustained turns (VQF yaw RATE via turn_state with 2.5 s
  entry + backward dilation, raised-cosine rejoin) holds pre-turn level
  through sustained one-way lateral that VQF itself admits. No gyro
  integration anywhere (coning/bias diverge tens of degrees on real
  data); yaw rate is gyro-dominated and drift-free. Documented leak:
  carve with counter-rotated head reads straight (accepted, rare).
No Euler extraction, no hand-derived signs anywhere.

Known limit (pinned in tests): VQF's bias estimator absorbs part of
SLOW oscillations (~25% at 0.05 Hz) — slow rolls under-corrected,
never invented. Eye decides magnitude on footage.
"""
import numpy as np


def quat_to_mat(q):
    """3x3 body->world from (w, x, y, z)."""
    q = np.asarray(q, float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))


def resample_shifts(path_t, imu_t, shifts, start):
    """Shifts onto ControlPoint timestamps. path_t is CLIP-RELATIVE
    (imu_time_s - start + sync offset); imu_t is absolute. Mixing the
    clocks parks every point left of xp[0] and applies one constant
    from before the window (review R2 B1) — subtract start here.
    """
    import numpy as _np
    pt = _np.asarray(path_t, float)
    xp = _np.asarray(imu_t, float) - float(start)
    return _np.interp(pt, xp, _np.asarray(shifts, float),
                      left=shifts[0], right=shifts[-1])


def level_shifts(t, gyro, accel, axis_map, sel=None):
    """Per-sample roll-slot shifts (deg) from VQF-predicted body gravity.

    Returns (shifts, info); info carries turn_frac/n_turns for the cert
    (rev2: duty-cycle visibility) and the p95 shift size.

    sel: optional boolean mask scoping turn_frac/refusal to the delivery
    window. Pass WIDE arrays (attitude needs context: q0 from a violent
    10 s window poisons the whole prediction) with sel on the window;
    freeze spans are computed wide so rejoins at window edges are sane.
    """
    import vqf
    from steadycut.stabilization.level import shift_from_gravity
    t = np.asarray(t, float)
    gyro = np.asarray(gyro, float)
    accel = np.asarray(accel, float)
    dt = float(np.median(np.diff(t)))
    p_col, p_sign = axis_map.pitch
    r_col, r_sign = axis_map.roll
    g_col, g_sign = axis_map.yaw  # gravity lives in the vendor yaw column
    # Handedness fix (review R2): the map frame (signed permutation of
    # raw columns) is left-handed for the X4 map (det -1: the signs serve
    # the 180-deg-flipped sphere). VQF needs a proper right-handed frame
    # with Z up, so negate Z iff the signed frame has det -1. Per-body
    # safe: a det +1 map keeps its signs; |roll|>90 still refuses below.
    _perm = [p_col, r_col, g_col]
    _inv = sum(1 for i in range(3) for j in range(i + 1, 3)
               if _perm[i] > _perm[j])
    if p_sign * r_sign * g_sign * (-1) ** _inv < 0:
        g_sign = -g_sign

    def _build(ps, rs, gs):
        g = np.radians(np.column_stack((
            ps * gyro[:, p_col], rs * gyro[:, r_col],
            gs * gyro[:, g_col])))
        a = np.column_stack((
            ps * accel[:, p_col], rs * accel[:, r_col],
            gs * accel[:, g_col]))
        return g, a

    gyr, acc = _build(p_sign, r_sign, g_sign)
    out = vqf.offlineVQF(gyr, acc, None, dt)
    R = quat_to_mat(np.asarray(out["quat6D"], float))  # body -> world
    # absurdity guard: a bike/ski camera never rides upside-down; if the
    # winning branch still reads |roll| > 90 deg the frame is inverted
    # (map signs are renderer-frame) — refuse, do not ship it.
    _w, _x, _y, _z = np.asarray(out["quat6D"], float).T
    _roll = np.degrees(np.arctan2(2 * (_w * _x + _y * _z),
                                  1 - 2 * (_x * _x + _y * _y)))
    # body components of world-up: v_b with R v_b = e_z, i.e. row 2 of R
    up_body = np.einsum("...ji,...j->...i", R, np.array([0.0, 0.0, 1.0]))
    # VQF frame (X,Y,Z) = signed (pitch, roll, gravity) columns -> raw
    gvec = np.empty_like(up_body)
    gvec[:, p_col] = up_body[:, 0] / p_sign
    gvec[:, r_col] = up_body[:, 1] / r_sign
    gvec[:, g_col] = up_body[:, 2] / g_sign
    raw = np.array([shift_from_gravity(gvec[i], axis_map)
                    for i in range(len(t))])
    w, x, y, z = np.asarray(out["quat6D"], float).T
    yaw_w = np.degrees(np.gradient(np.unwrap(np.arctan2(
        2 * (w * z + x * y), 1 - 2 * (y * y + z * z))), dt))
    # freeze gate: SUSTAINED world yaw rate (turn_state: 2.5 s entry, so
    # spikes and glances never enter), backward-dilated for the entry.
    # No gyro integration anywhere: coning and bias make a predicted-up
    # innov diverge tens of degrees on real data (measured), while yaw
    # RATE is gyro-dominated and drift-free. Documented leaks (review R2):
    # a carve with counter-rotated head reads straight (accepted, rare);
    # sub-threshold rate with real lateral leaks (5.5 dps + 2.0 m/s2
    # leaks ~8.6 of ~11.5 deg — wide GS turns live there; gate on
    # v*omega when speed exists, else accept).
    import steadycut.stabilization.orientation as _O
    back = int(round(2.5 / dt))
    turn_m = _O.turn_state(t, yaw_w)
    # dilate each contiguous span backward (entry coverage), offline luxury
    idx = np.flatnonzero(turn_m)
    if len(idx):
        cuts = np.flatnonzero(np.diff(idx) > 1)
        starts = np.concatenate([[idx[0]], idx[cuts + 1]])
        for s in starts:
            turn_m[max(0, s - back):s] = True
    frozen = turn_m
    # hold pre-freeze value through each span, raised-cosine rejoin
    shifts = raw.copy()
    n = len(t)
    i = 0
    spans = []
    while i < n:
        if not frozen[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and frozen[j + 1]:
            j += 1
        spans.append((i, j))
        i = j + 1
    for a, b in spans:
        # hold the MEDIAN over the preceding ~1 s (a single unsmoothed
        # sample can sit on vibration; review R2)
        lo = max(0, a - int(round(1.0 / dt)))
        pre = raw[lo:a] if a > lo else np.array([])
        hold = float(np.median(pre)) if len(pre) else (
            raw[b + 1] if b + 1 < n else float(np.median(raw)))
        shifts[a:b + 1] = hold
        m = min(n - b - 1, int(round(2.0 / dt)))
        if m > 1 and b + 1 < n:
            wgt = 0.5 - 0.5 * np.cos(np.pi * np.arange(m) / m)
            shifts[b + 1:b + 1 + m] = ((1 - wgt) * hold + wgt * raw[b + 1:b + 1 + m])
    # level reference carries no shake: the gyro path owns fast motion.
    # Zero-phase lowpass at level timescales (vibration-speed noise in
    # VQF attitude otherwise lands straight in the roll slots: measured
    # 24 deg/frame pump on Mar05 from +-8 deg unsmoothed shifts).
    from steadycut.stabilization.orientation import _smooth_1d
    shifts = _smooth_1d(shifts, t, 3.0)
    if sel is None:
        lo, hi = 0, n
    else:
        idx = np.flatnonzero(np.asarray(sel, bool))
        lo, hi = (int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, 0)
    frozen_n = sum(min(b, hi - 1) - max(a, lo) + 1
                   for a, b in spans if b >= lo and a < hi)
    scope = float(max(1, hi - lo))
    turn_frac = float(frozen_n / scope)
    info = {"turn_frac": turn_frac,
            "n_turns": len(spans),
            "shift_p95": float(np.percentile(np.abs(shifts[lo:hi]), 95))
            if hi > lo else 0.0,
            "raw_p95": float(np.percentile(np.abs(raw[lo:hi]), 95))
            if hi > lo else 0.0,
            "rest_frac": float(np.mean(out["restDetected"]))}
    if turn_frac > 0.5:
        # open-loop most of the run (no steady second anywhere):
        # refuse rather than pump. Caller falls back to static.
        info["refused"] = ("turn_frac %.2f > 0.50: no gravity reference "
                           "in this window" % turn_frac)
    if abs(float(np.median(_roll))) > 90.0:
        info["refused"] = ("roll median %.0f deg: inverted attitude" %
                           float(np.median(_roll)))
    return shifts, info
