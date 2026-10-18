"""Time-varying roll leveling from VQF (gravity-leveling rev2, step 2/3).

Split of duties (axis-swamp avoidance):
- VQF supplies the ATTITUDE (gyro-carried, sees through turn
  centripetal via earth-frame lowpass + offline forward-backward).
- `shift_from_gravity` (eye-settled on Jan +66.6 / Feb -104, NOT
  derived) supplies the CONVENTION: per-sample shift from VQF-predicted
  body gravity, same formula as the static path. On straights this
  reduces to the static shift by construction; in turns the prediction
  ( unlike raw accel ) carries no centripetal.
- Innovation freeze (gyro-predicted up, UNION turn segments,
  backward-dilated, raised-cosine rejoin) holds pre-turn level through
  sustained one-way lateral that VQF itself admits.
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


def level_shifts(t, gyro, accel, axis_map):
    """Per-sample roll-slot shifts (deg) from VQF-predicted body gravity.

    Returns (shifts, info); info carries turn_frac/n_turns for the cert
    (rev2: duty-cycle visibility) and the p95 shift size.
    """
    import vqf
    from steadycut.calibration import mount
    from steadycut.stabilization import orientation as O
    from steadycut.stabilization.level import shift_from_gravity
    t = np.asarray(t, float)
    gyro = np.asarray(gyro, float)
    accel = np.asarray(accel, float)
    dt = float(np.median(np.diff(t)))
    p_col, p_sign = axis_map.pitch
    r_col, r_sign = axis_map.roll
    g_col, g_sign = axis_map.yaw  # gravity lives in the vendor yaw column
    gyr = np.radians(np.column_stack((
        p_sign * gyro[:, p_col], r_sign * gyro[:, r_col],
        g_sign * gyro[:, g_col])))
    acc = np.column_stack((
        p_sign * accel[:, p_col], r_sign * accel[:, r_col],
        g_sign * accel[:, g_col]))
    out = vqf.offlineVQF(gyr, acc, None, dt)
    R = quat_to_mat(np.asarray(out["quat6D"], float))  # body -> world
    # body components of world-up: v_b with R v_b = e_z, i.e. row 2 of R
    up_body = np.einsum("...ji,...j->...i", R, np.array([0.0, 0.0, 1.0]))
    # VQF frame (X,Y,Z) = signed (pitch, roll, gravity) columns -> raw
    gvec = np.empty_like(up_body)
    gvec[:, p_col] = up_body[:, 0] / p_sign
    gvec[:, r_col] = up_body[:, 1] / r_sign
    gvec[:, g_col] = up_body[:, 2] / g_sign
    raw = np.array([shift_from_gravity(gvec[i], axis_map)
                    for i in range(len(t))])
    # freeze gate: GYRO-predicted up (zero accel influence — VQF-blended
    # attitude would self-mask sustained lateral by tilting toward it).
    # O.integrate from an accel-leveled start; only the horizontal
    # MAGNITUDE is read, which is convention-safe under mirroring.
    q0 = O.level_start(t, acc, np.degrees(gyr))
    q_gyro = O.integrate(t, np.degrees(gyr), q0)
    Rg = quat_to_mat(q_gyro)
    a_world = np.einsum("...ij,...j->...i", Rg, acc)
    innov_h = np.hypot(a_world[:, 0], a_world[:, 1])
    w, x, y, z = np.asarray(out["quat6D"], float).T
    yaw_w = np.gradient(np.unwrap(np.arctan2(
        2 * (w * z + x * y), 1 - 2 * (y * y + z * z))), dt)
    segs = mount.turn_segments(t, np.degrees(yaw_w))
    frozen = (innov_h > 2.0)
    back = int(round(2.5 / dt))
    turn_m = np.zeros(len(t), bool)
    for a, b, _s in segs:
        turn_m[max(0, a - back):b + 1] = True
    frozen = frozen | turn_m
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
        hold = raw[a - 1] if a > 0 else raw[b + 1]
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
    frozen_n = sum(b - a + 1 for a, b in spans)
    turn_frac = float(frozen_n / max(1, n))
    info = {"turn_frac": turn_frac,
            "n_turns": len(spans),
            "shift_p95": float(np.percentile(np.abs(shifts), 95)),
            "rest_frac": float(np.mean(out["restDetected"])),
            "innov_p95": float(np.percentile(innov_h, 95))}
    if turn_frac > 0.5:
        # open-loop most of the run (no steady second anywhere):
        # refuse rather than pump. Caller falls back to static.
        info["refused"] = ("turn_frac %.2f > 0.50: no gravity reference "
                           "in this window" % turn_frac)
    return shifts, info
