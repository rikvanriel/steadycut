"""Gravity-gated attitude tracker (Gap-1 fix attempt).

Design from the file-backed stabilizer docs (insta360-rs stabilization.md):
- init: average a QUALIFIED still gravity interval for initial tilt
- integrate with actual intervals, midpoint rates, bias removed
- gravity correction gated: |a| within 0.15 g of 1 g AND angle from predicted
  up under 20 deg; rejected samples leave gyro propagation active
- correction is tilt-only (gravity gives no heading): small rotation taking
  predicted body-up toward measured body-up, applied at a fraction per sample

Frame: telemetry-parser normalized frame shared by gyro+accel (no remap; the
mounting rotation is Gap-3, a separate step). Correction output uses the same
v360 axis mapping as orientation.correction_axes so renders are comparable.
"""
import numpy as np

from steadycut.stabilization import orientation as O

G = 9.81
MAG_GATE = 0.15 * G
ANGLE_GATE_DEG = 20.0


def estimate_bias(t, gyro, accel, lo=0.0, hi=1.0):
    m = (t >= lo) * (t < hi)
    return gyro[m].mean(axis=0), accel[m].mean(axis=0), m.sum()


def track(t, gyro, accel, bias, tilt_gain=0.02):
    """Attitude quaternions body->world, one per sample. Yaw starts at 0."""
    g = (gyro - bias) * np.pi / 180.0
    # init tilt from mean accel over first second: body-up = -a/|a|... accel
    # measures specific force (up positive when still), so up_body = a/|a|
    up0 = accel[:1000].mean(axis=0)
    up0 = up0 / np.linalg.norm(up0)
    # quaternion taking body-up to world +Z: rotate about axis cross(up0, z)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(up0, z)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        q = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        ang = np.arcsin(min(s, 1.0))
        q = np.concatenate([[np.cos(ang / 2)], axis / s * np.sin(ang / 2)])
    qs = np.zeros((len(t), 4))
    qs[0] = q / np.linalg.norm(q)
    n_acc = 0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if not 0 < dt < 0.05:
            qs[i] = qs[i - 1]
            continue
        # midpoint rate, substep cap 0.1 rad
        wmid = 0.5 * (g[i - 1] + g[i])
        ang = np.linalg.norm(wmid) * dt
        nsub = max(1, int(np.ceil(ang / 0.1)))
        q = qs[i - 1]
        for _ in range(nsub):
            dq = O.from_rotvec((wmid * (dt / nsub))[None, :])[0]
            q = O._normalise(O._mul(q, dq))
        # gated tilt correction
        a = accel[i]
        am = np.linalg.norm(a)
        if abs(am - G) < MAG_GATE and am > 1e-9:
            meas_up = a / am  # body frame
            # predicted body-up: world +Z expressed in body = conj(q) * Z * q
            zq = np.array([0.0, 0.0, 0.0, 1.0])
            pred = O._mul(O._mul(O._conj(q), zq), q)[1:]
            pred = pred / np.linalg.norm(pred)
            cosang = np.clip(float(pred @ meas_up), -1.0, 1.0)
            if np.degrees(np.arccos(cosang)) < ANGLE_GATE_DEG:
                corr_axis = np.cross(pred, meas_up)
                if np.linalg.norm(corr_axis) > 1e-12:
                    tiltdq = O.from_rotvec(
                        (corr_axis / np.linalg.norm(corr_axis)
                         * np.arccos(cosang) * tilt_gain)[None, :])[0]
                    q = O._normalise(O._mul(q, tiltdq))
                    n_acc += 1
        qs[i] = q
    return qs, n_acc


def correction_axes(t, gyro, accel, tau_s, bias, tilt_gain=0.02):
    """Same contract as orientation.correction_axes: yaw, pitch, roll in deg."""
    q_raw = track(t, gyro, accel, bias, tilt_gain)[0]
    q_smooth = O.smooth(q_raw, t, tau_s)
    rv = np.degrees(O.to_rotvec(O.relative(q_raw, q_smooth)))
    # NOTE: orientation.correction_axes reorders to body [2,1,0] before
    # integrating. Here integration is in the parser frame directly, where
    # gyro[0]=roll about X, gyro[1]=yaw about Y, gyro[2]=pitch about Z, so
    # the rotvec components read back as yaw=-rv_y, pitch=-rv_z, roll=-rv_x
    # (negated for the same sphere-180 reason as the orientation pipeline).
    return -rv[:, 1], -rv[:, 2], -rv[:, 0]
