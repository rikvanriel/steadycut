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
    # MEASURED optimum corner frequency, and that was re-measured once the v360
    # command fault was fixed: residual motion is flat from 0.05 to 1.5 Hz and the
    # previously recorded 3.18 Hz sits outside that plateau, ~30 percent worse on
    # rotation. The production default is now 0.5 Hz (path.build_path_quat), chosen
    # mid-plateau and for behaviour: a lower corner follows the rider's head turns
    # less and the direction of travel more. Do not hand-restate the value in a tool
    # that is meant to evaluate production - read it from the production default.
    return (am.yaw[1] * rv[:, 1], am.pitch[1] * rv[:, 0], am.roll[1] * rv[:, 2])
