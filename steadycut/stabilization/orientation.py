"""Camera orientation as quaternions, and the correction as a relative rotation.

path.py integrates each gyro axis into an Euler angle independently, filters
each independently, and applies each independently. Gyroflow's technical docs
describe why that is wrong: filtering Euler axes separately produces
"non-linear behavior in the resulting orientation". Its pipeline is

    integrate gyro -> quaternion orientation per sample
    -> low pass in quaternion space via Slerp, forward and reverse
    -> correction = rotation between smoothed and raw, a RELATIVE rotation

The relative form matters here beyond correctness: the absolute roll carries a
mount offset of about 20 degrees, and a correction expressed as a rotation
between two orientations cannot mis-handle that offset, whereas an absolute
angle path can and does.
"""
import numpy as np


def _mul(a, b):
    """Hamilton product, for arrays of quaternions in (w, x, y, z) order."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def _conj(q):
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def _normalise(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def from_rotvec(rotvec):
    """Quaternion from a rotation vector (axis * angle, radians)."""
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    half = angle / 2.0
    scale = np.where(angle > 1e-12, np.sin(half) / np.maximum(angle, 1e-12), 0.5)
    return np.concatenate([np.cos(half), rotvec * scale], axis=-1)


def to_rotvec(q):
    """Rotation vector (axis * angle, radians) of a quaternion."""
    q = _normalise(q)
    w = np.clip(q[..., 0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(np.maximum(1.0 - w * w, 0.0))
    scale = np.where(s > 1e-12, angle / np.maximum(s, 1e-12), 0.0)
    return q[..., 1:] * scale[..., None]


def integrate(t, gyro, q0=None):
    """Quaternion orientation per sample, from angular rate in degrees/second.

    Each step composes a small rotation about the instantaneous axis, so the
    axes are combined in the correct order instead of being accumulated
    independently as Euler angles.

    q0 is the start attitude (default identity). Pass an accel-leveled q0
    when the absolute tilt matters (e.g. projecting rates about world
    vertical): an identity start tilts the reference by the unknown mount
    itself.
    """
    rate = np.radians(gyro)
    dt = np.gradient(t)[:, None]
    deltas = from_rotvec(rate * dt)
    out = np.zeros((rate.shape[0], 4))
    q = np.array([1.0, 0.0, 0.0, 0.0]) if q0 is None else np.asarray(q0, float)
    for i in range(rate.shape[0]):
        q = _normalise(_mul(q, deltas[i]))
        out[i] = q
    return out


def _slerp_step(q_from, q_to, alpha):
    """Move a fraction alpha from q_from towards q_to, along the short arc."""
    rel = _mul(_conj(q_from), q_to)
    return _normalise(_mul(q_from, from_rotvec(to_rotvec(rel) * alpha)))


def smooth(q, t, tau_s):
    """Slerp low pass, run forward then in reverse to remove the delay.

    A single causal pass introduces a lag of order tau_s, which would then show
    up as a spurious correction. Running the same filter backwards over the
    result cancels that lag, so a constant-rate rotation comes through
    unchanged - which is the property we want, since a steady pan is the rider
    moving deliberately.
    """
    if tau_s <= 0:
        return q.copy()
    dt = np.median(np.diff(t))
    alpha = 1.0 - np.exp(-dt / tau_s)
    out = q.copy()
    for i in range(1, len(q)):
        out[i] = _slerp_step(out[i - 1], q[i], alpha)
    back = out.copy()
    for i in range(len(q) - 2, -1, -1):
        back[i] = _slerp_step(back[i + 1], out[i], alpha)
    return back


def to_euler(q):
    """ZYX Euler angles (yaw about z, pitch about y, roll about x), degrees."""
    q = _normalise(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    return np.degrees(np.stack([yaw, pitch, roll], axis=-1))


def to_euler_zxy(q):
    """Angles (y2, p, r) with R = R_z(r) R_x(p) R_y(y2), degrees.

    This is the decomposition the v360 render path needs: measuring real
    renders with single markers and with combined rotations shows the view
    v360 builds from a (yaw, pitch, roll) command acts on sphere content as
    R_z(roll) R_x(pitch) R_y(-yaw) -- yaw negated, pitch and roll as given
    (0.8 px median on fresh angles against 25+ px for every other sign and
    order combination tried; single-axis probes confirm pitch and roll signs
    individually, yaw negation at 0.55 px against 145 px un-negated).

    That differs from a plain reading of vf_v360.c `calculate_rotation`
    (Y quaternion times X times Z), which would predict R_y R_x R_z with no
    flips. The renders are ground truth here and they disagree with that
    reading -- whether the flip lives in the yaw option handling, the
    view/content duality of `rotate`, or the frame handedness is not
    established, so this function documents the MEASURED behavior and the
    render test below locks it. Do not "simplify" this toward the source
    reading without re-running that test.

    Returns [y2, p, r]; the caller sends yaw = -y2. Keeping the negation at
    the call site makes the measured flip visible instead of burying it in
    the math.

    The orders coincide for single-axis rotations and differ only once two
    axes are non-zero together, so single-axis probes cannot choose an order
    -- the pitch+roll combination is the cheapest input that can (114 px at
    p95 for the neighbouring ZYX order against 1.4 px here).

    Degenerate at pitch = +-90 deg, where y2 and r are no longer separable;
    the gimbal-lock branch holds r at 0 and takes y2 from the remaining
    rotation, which reconstructs the same rotation.
    """
    q = _normalise(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # For R = R_z(r) R_x(p) R_y(y2):
    #   R[2][1] = sin(p), R[2][0]/R[2][2] give y2, R[0][1]/R[1][1] give r.
    # In quaternion terms R[2][1] = 2(yz + wx), R[2][0] = 2(xz - wy),
    # R[2][2] = 1-2(x^2+y^2), R[0][1] = 2(xy - wz), R[1][1] = 1-2(x^2+z^2).
    sp = np.clip(2.0 * (y * z + w * x), -1.0, 1.0)
    pitch = np.arcsin(sp)
    y2 = np.arctan2(-2.0 * (x * z - w * y), 1.0 - 2.0 * (x * x + y * y))
    roll = np.arctan2(-2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z))
    locked = np.abs(np.abs(sp) - 1.0) < 1e-9
    if np.any(locked):
        # At the poles only y2 +/- r is observable. Hold r at 0 and take y2
        # from R[0][0]/R[1][0]: for p = +90 those read (cos y2, sin y2),
        # for p = -90 (cos y2, -sin y2).
        r00 = 1.0 - 2.0 * (y * y + z * z)
        r10 = 2.0 * (x * y + w * z)
        y2_lock = np.where(sp > 0.0, np.arctan2(r10, r00),
                           np.arctan2(-r10, r00))
        y2 = np.where(locked, y2_lock, y2)
        roll = np.where(locked, 0.0, roll)
    return np.degrees(np.stack([y2, pitch, roll], axis=-1))


def to_euler_zyx(q):
    """ZYX Euler angles (about Z, then Y, then X), degrees.

    Distinct from `to_euler`, which decomposes as R_z(yaw) R_y(pitch) R_x(roll).
    This one gives the angles in the order R_z(a) R_y(b) R_x(c) and returns them
    as [a, b, c], so a caller asking for (roll, yaw, pitch) can take them in
    that order. v360 composes its three numbers that way, so this is the
    decomposition that matches it.

    Degenerate at c = +-90 deg, where b and c are no longer separable; the
    gimbal-lock branch returns b with c = 0, which is one of the two valid
    solutions and differs from the other only by a rotation about the locked
    axis.
    """
    q = _normalise(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # For R = R_z(a) R_y(b) R_x(c): R[2][0] = -sin(b), R[1][0]/R[0][0] give a,
    # and R[2][1]/R[2][2] give c. Note the two off-diagonal terms are
    # R[2][0] = 2(xz - wy) and R[1][0] = 2(xy + wz) -- swapping those two makes
    # a pure Z rotation come back as a pure Y one, which is what the first
    # version of this did.
    b = np.arcsin(np.clip(2.0 * (w * y - x * z), -1.0, 1.0))
    a = np.arctan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))
    cb = np.cos(b)
    locked = np.abs(cb) < 1e-9
    c = np.where(locked, 0.0,
                 np.arctan2(2.0 * (y * z + w * x),
                            1.0 - 2.0 * (x * x + y * y)))
    del cb, locked
    return np.degrees(np.stack([a, b, c], axis=-1))


def _smooth_1d(x, t, tau_s):
    """Scalar twin of smooth(): exponential low pass, forward then reversed
    to remove the delay, so a constant-rate sweep comes through unchanged.
    tau_s may be a per-sample array (adaptive leash: fast in turns, slow
    otherwise); the same alphas run both directions."""
    x = np.asarray(x, float)
    n = len(x)
    if np.all(np.asarray(tau_s) <= 0):
        return x.copy()
    dt = float(np.median(np.diff(t)))
    tau = np.broadcast_to(np.asarray(tau_s, float), (n,)).copy()
    tau[tau <= 0] = np.inf
    alpha = 1.0 - np.exp(-dt / tau)
    out = x.copy()
    for i in range(1, n):
        out[i] = out[i - 1] + alpha[i] * (x[i] - out[i - 1])
    back = out.copy()
    for i in range(n - 2, -1, -1):
        back[i] = back[i + 1] + alpha[i] * (out[i] - back[i + 1])
    return back


def _zxy_to_quat(y2, p, r):
    """Quaternions for R = R_z(r) R_x(p) R_y(y2), radians. Inverse of the
    to_euler_zxy decomposition (up to the gimbal branch, which holds r at 0
    and is exact there too)."""
    y2, p, r = np.broadcast_arrays(y2, p, r)
    cy, sy = np.cos(y2 / 2), np.sin(y2 / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    # q = qz(r) * qx(p) * qy(y2), Hamilton product, (w,x,y,z).
    qy = np.stack([cy, np.zeros_like(cy), sy, np.zeros_like(cy)], -1)
    qx = np.stack([cp, sp, np.zeros_like(cp), np.zeros_like(cp)], -1)
    qz = np.stack([cr, np.zeros_like(cr), np.zeros_like(cr), sr], -1)
    return _normalise(_mul(_mul(qz, qx), qy))


def smooth_channels(q_raw, t, tau_yaw_s, tau_other_s):
    """Trend with yaw on its own leash: decompose to v360 Euler, unwrap,
    filter yaw with tau_yaw_s and pitch/roll with tau_other_s (same zero-lag
    forward-backward shape as smooth), recompose. This is how a travel policy
    holds a heading through look-arounds while keeping production shake
    removal on pitch/roll -- done on angles before composition, never by
    splicing slots from two different smoothings (whose Euler components do
    not compose: measured 77 px rms on a spliced probe against 7.7 for the
    composed equivalent).

    Gimbal caveat: at pitch +-90 deg y2/r are not separable (same branch as
    to_euler_zxy). Ski/MTB pitch never goes there; a policy that needs the
    poles wants a different decomposition, not a wider tolerance here.

    tau_yaw_s may be a (slow_s, fast_s) pair: the leash runs fast during
    sustained turns (IMU turn detector below) and slow otherwise, so a
    144-degree carved turn is followed instead of ending up behind it,
    while sub-2.5 s look-arounds never engage the fast leash and stay
    held. A scalar is the legacy fixed leash, bit for bit.
    """
    e = to_euler_zxy(q_raw)
    y = np.unwrap(np.radians(e[:, 0]))
    p = np.unwrap(np.radians(e[:, 1]))
    r = np.unwrap(np.radians(e[:, 2]))
    if isinstance(tau_yaw_s, tuple):
        slow_s, fast_s = tau_yaw_s
        # body yaw rate from consecutive attitudes (robust where Euler
        # angles flip: violent rolls make to_euler_zxy discontinuous and
        # its gradient crosses zero constantly, so the detector never
        # engages). Signed lowpass: shake cancels, sustained turns pass.
        dq = relative(q_raw[:-1], q_raw[1:])
        dt = float(np.median(np.diff(t)))
        wy = to_rotvec(dq)[:, 1] / dt  # rad/s about body-Y (yaw channel)
        wyaw = _smooth_1d(np.degrees(wy), t[:-1], 2.0)
        pad = np.zeros(len(t))
        pad[:len(wyaw)] = wyaw
        pad[len(wyaw):] = wyaw[-1]
        turning = turn_state(t, pad)
        tau = np.where(turning, fast_s, slow_s)
    else:
        tau = tau_yaw_s
    return _zxy_to_quat(_smooth_1d(y, t, tau),
                        _smooth_1d(p, t, tau_other_s),
                        _smooth_1d(r, t, tau_other_s))


def relative(q_raw, q_smooth):
    """Rotation from the raw orientation to the smoothed one.

    Applying this on top of the raw orientation yields the smoothed
    orientation, so it is the rotation the virtual camera needs. Expressed this
    way it is a difference between two orientations and cannot be wrong about
    any absolute offset, such as the mount angle.
    """
    return _normalise(_mul(_conj(q_raw), q_smooth))


def level_start(t, accel_body, gyro_body=None):
    """Start attitude with body pitch/roll leveled from the accelerometer.

    Minimal rotation mapping the mean specific-force direction to +z.
    Uses the straightest window (lowest body-rate magnitude — frame
    invariant, needs no attitude): a mean taken in a turn reversal would
    bake ~1 g of transient into the vertical. Falls back to the first
    3 s. Yaw stays arbitrary (identity) — only the tilt matters for a
    world-vertical reference.
    """
    t = np.asarray(t, float)
    a = np.asarray(accel_body, float)
    n = len(a)
    dt = float(np.median(np.diff(t))) if n > 1 else 0.01
    i0, i1 = 0, min(n, max(1, int(round(3.0 / dt))))
    if gyro_body is not None:
        g = np.asarray(gyro_body, float)
        mag = np.linalg.norm(g, axis=1)
        w = max(1, int(round(1.0 / dt)))
        best, bestm = None, 4.0
        for i in range(0, max(1, n - w), w // 2):
            m = float(mag[i:i + w].mean())
            if m < bestm:
                bestm, best = m, (i, i + w)
        if best is not None:
            i0, i1 = best
    v = a[i0:i1].mean(axis=0)
    v = v / (np.linalg.norm(v) + 1e-12)
    # minimal rotation v -> +z: axis v x z, angle acos(v.z)
    axis = np.array([v[1], -v[0], 0.0])
    s = float(np.linalg.norm(axis))
    if s < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / s
    ang = float(np.arccos(np.clip(v[2], -1.0, 1.0)))
    return np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * axis])


def turn_state(t, rate, enter_s=2.5, exit_s=1.0, enter_dps=6.0,
               exit_dps=3.0, pad_s=1.0):
    """Sustained-turn mask with hysteresis: enter after enter_s above
    enter_dps, exit after exit_s below exit_dps, dilated by pad_s so the
    fast leash covers entries/exits. Brief glances (< enter_s) never
    enter; the forward-backward filter smooths the edges that remain."""
    t = np.asarray(t, float)
    r = np.abs(np.asarray(rate, float))
    n = len(t)
    dt = float(np.median(np.diff(t))) if n > 1 else 0.01
    ne, nx, npad = (max(1, int(round(enter_s / dt))),
                    max(1, int(round(exit_s / dt))),
                    int(round(pad_s / dt)))
    on = np.zeros(n, bool)
    state, c_on, c_off = False, 0, 0
    for i in range(n):
        if r[i] > enter_dps:
            c_on += 1
            c_off = 0
        elif r[i] < exit_dps:
            c_off += 1
            c_on = 0
        else:
            c_off = 0
            # dead band: hold run-up, keep counting exit only when low
        if not state and c_on >= ne:
            state = True
        if state and c_off >= nx:
            state = False
        on[i] = state
    if npad > 0 and on.any():
        # dilate each contiguous block (not the bbox: inter-turn glides
        # keep the slow leash so glances between turns are still held)
        w = np.ones(2 * npad + 1)
        on = np.convolve(on.astype(float), w, mode="same") > 0
    return on


def correction_axes(t, gyro, tau_s, axis_map=None, yaw_tau_s=None):
    """Yaw, pitch and roll correction in degrees, from the quaternion pipeline.

    `axis_map` states, per channel, which gyro column carries it and with what
    sign; both are measured on the model (calibrate.py for columns,
    evaluate.py --per-channel for signs) and default to the X4 map. The vector
    is reordered to a body frame of (right, up, forward) before integrating.
    The correction is read back as a rotation vector rather than as Euler
    angles: the components map straight onto the three axes, which avoids the
    singularities and order dependence of Euler extraction for the small
    corrections that stabilisation produces.

    `yaw_tau_s` puts yaw on its own leash (travel policies): the trend is
    built by smooth_channels instead of smooth, holding a heading through
    look-arounds. None (default) is byte-identical to the old path.
    """
    if axis_map is None:
        raise ValueError(
            "no measured axis map: pass the one the camera registry loaded "
            "for this body; an unknown body is measured from its own footage "
            "on first use")
    am = axis_map
    body = gyro[:, [am.pitch[0], am.yaw[0], am.roll[0]]]
    q_raw = integrate(t, body)
    if yaw_tau_s is None:
        q_smooth = smooth(q_raw, t, tau_s)
    else:
        q_smooth = smooth_channels(q_raw, t, yaw_tau_s, tau_s)
    rv = np.degrees(to_rotvec(relative(q_raw, q_smooth)))
    # Signs come from the axis map, measured per model: the stitched sphere is
    # rotated 180 degrees relative to the camera axes by the fisheye stacking
    # order, so the correction that reads correctly on paper makes the image
    # worse. Driving each channel alone: yaw -23.6%, pitch -25.6% and roll
    # -16.9% on their own metrics, against +30 to +49% the other way -- for the
    # X4 that is sign -1 on every channel, carried in its profile rather than
    # hardcoded here.
    #
    # Validated again with the corrected stitch: this path improves
    # every one of five windows with no regressions, mean -17.4 percent.
    #
    # An earlier note here claimed this path performed worse under the corrected
    # stitch and advised preferring build_path. That measurement came from
    # evaluate.py while it was using a time constant six times too slow (0.3 s
    # against the measured optimum near 0.05 s), so the apparent failure was that
    # parameter and not this mapping or the stitch. Once both tools were made to
    # use the production value they agreed to the digit.
    #
    # The time constant matters more than anything else here. It must come from the
    # MEASURED optimum corner frequency. An earlier note recorded a flat plateau
    # from 0.05 to 1.5 Hz with 3.18 Hz outside it, measured per axis on 4 s windows
    # with phase correlation; re-measuring the FAR FIELD (the landscape, which
    # moves only by rotation) with the ORB estimator on 20 s windows of two clips
    # gives a different answer: lowering the corner from 0.5 to 0.15 Hz improved
    # landscape amplitude on every window (60.5 -> 71.2 percent of raw removed, on
    # average), and 0.05 Hz bought no further amplitude while costing the near
    # field. The production default is therefore 0.15 Hz, which also agrees with
    # the behaviour the rider asked for: a lower corner follows the rider's head
    # turns less and the direction of travel more. Do not hand-restate the value in
    # a tool that is meant to evaluate production - read it from the production
    # default.
    return (am.yaw[1] * rv[:, 1], am.pitch[1] * rv[:, 0], am.roll[1] * rv[:, 2])
