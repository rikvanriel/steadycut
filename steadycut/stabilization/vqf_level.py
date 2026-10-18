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

    def _build(ps, rs, gs):
        g = np.radians(np.column_stack((
            ps * gyro[:, p_col], rs * gyro[:, r_col],
            gs * gyro[:, g_col])))
        a = np.column_stack((
            ps * accel[:, p_col], rs * accel[:, r_col],
            gs * accel[:, g_col]))
        return g, a

    gyr, acc = _build(p_sign, r_sign, g_sign)
    # Frame fix (measured on MTB + ski, two bodies, same sensor): the
    # map's Z sign is renderer-frame (sphere flipped 180 deg) — VQF with
    # map signs converges with body-Z pointing DOWN (up_z -0.96) on every
    # window. Negate Z always, then break the remaining yaw-180 tie
    # (X vs Y negation) by pitch agreement with the label-pinned online
    # estimator. Both branches describe the same tilt; the pin picks the
    # representation, and |roll|>90 refuses if neither is upright.
    from steadycut.stabilization import attitude_vqf as _AV
    try:
        pin = float(np.median(_AV.attitude(
            np.degrees(gyr), acc, dt).pitch))
    except Exception:
        pin = None
    g_sign = -g_sign
    gyr, acc = _build(p_sign, r_sign, g_sign)
    out = vqf.offlineVQF(gyr, acc, None, dt)

    def _pitchmed(o):
        q = np.asarray(o["quat6D"], float)
        w, x, y, z = q.T
        return float(np.median(-np.degrees(np.arcsin(
            np.clip(2 * (w * y - z * x), -1, 1)))))
    if pin is not None and (_pitchmed(out) > 0) != (pin > 0):
        # other yaw branch: negate X&Y (180 deg about Z, proper); Z stays
        # negated (map Z is renderer-frame — re-flipping it restores the
        # inverted attitude and shifts read +-150 deg)
        gyr, acc = _build(-p_sign, -r_sign, g_sign)
        out = vqf.offlineVQF(gyr, acc, None, dt)
        p_sign, r_sign = -p_sign, -r_sign
    # (g_sign already negated above unconditionally: map Z is renderer-frame)
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
    # RATE is gyro-dominated and drift-free. Documented leak: a carve
    # with counter-rotated head reads straight here (sensor rate ~0)
    # while lateral persists — rare, accepted; VQF lowpass covers the
    # alternating case and the freeze covers the sustained one.
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
        hold = raw[a - 1] if a > 0 else (raw[b + 1] if b + 1 < n
                                         else float(np.median(raw)))
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
