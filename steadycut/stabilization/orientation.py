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


def integrate(t, gyro):
    """Quaternion orientation per sample, from angular rate in degrees/second.

    Each step composes a small rotation about the instantaneous axis, so the
    axes are combined in the correct order instead of being accumulated
    independently as Euler angles.
    """
    rate = np.radians(gyro)
    dt = np.gradient(t)[:, None]
    deltas = from_rotvec(rate * dt)
    out = np.zeros((rate.shape[0], 4))
    q = np.array([1.0, 0.0, 0.0, 0.0])
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


def relative(q_raw, q_smooth):
    """Rotation from the raw orientation to the smoothed one.

    Applying this on top of the raw orientation yields the smoothed
    orientation, so it is the rotation the virtual camera needs. Expressed this
    way it is a difference between two orientations and cannot be wrong about
    any absolute offset, such as the mount angle.
    """
    return _normalise(_mul(_conj(q_raw), q_smooth))


def correction_axes(t, gyro, tau_s, axis_map=None):
    """Yaw, pitch and roll correction in degrees, from the quaternion pipeline.

    `axis_map` states, per channel, which gyro column carries it and with what
    sign; both are measured on the model (calibrate.py for columns,
    evaluate.py --per-channel for signs) and default to the X4 map. The vector
    is reordered to a body frame of (right, up, forward) before integrating.
    The correction is read back as a rotation vector rather than as Euler
    angles: the components map straight onto the three axes, which avoids the
    singularities and order dependence of Euler extraction for the small
    corrections that stabilisation produces.
    """
    if axis_map is None:
        raise ValueError(
            "no measured axis map: pass the one the camera registry loaded "
            "for this body; an unknown body is measured from its own footage "
            "on first use")
    am = axis_map
    body = gyro[:, [am.pitch[0], am.yaw[0], am.roll[0]]]
    q_raw = integrate(t, body)
    q_smooth = smooth(q_raw, t, tau_s)
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
