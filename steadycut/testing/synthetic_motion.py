"""Plausible synthetic riding motion, and the IMU that measures it.

A pure sinusoid is not a plausible ride and would not exercise the interesting
paths. Four superposed components, all deterministic:

    body sway     +-20 deg  0.3 Hz pitch/yaw    the slow motion smoothing follows
    heading turn  +-25 deg  0.15 Hz yaw         a sustained CURVE, not a jolt
    jolts          3-8 deg  8 impulses < 40 ms  the transient to be rejected
    roll          +-12 deg  0.4 Hz              all three axes at once

The derivative of that attitude is the synthetic IMU, and it is exact: it
integrates back to the attitude with no noise and no bias. That is the property
the whole end-to-end test rests on, because it makes the reference computable
rather than estimated -- so any deviation in the rendered pixels is
attributable to the pipeline rather than to the sensor.

Quaternions are built through scipy. Assembling `(0, y, p, r)` by hand -- the
obvious shortcut, treating the Euler triple as a quaternion -- yields a
zero-norm quaternion at t=0, which is exactly the kind of thing a synthetic test
must not contain and which fails as NaN several stages later.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from steadycut.ingest.telemetry import Telemetry
from steadycut.testing.synthetic_profile import IMU_HZ

G = 9.81


def attitude_euler(t: np.ndarray) -> np.ndarray:
    """(pitch, yaw, roll) in degrees at times `t` (seconds)."""
    t = np.asarray(t, dtype=float)
    pitch = 20.0 * np.sin(2 * np.pi * 0.30 * t)
    yaw = 25.0 * np.sin(2 * np.pi * 0.15 * t)
    roll = 12.0 * np.sin(2 * np.pi * 0.40 * t)

    # Jolts: short, asymmetric, and placed away from the curve turns so they
    # are not mistaken for them. Deterministic spacing, not random.
    for k in range(8):
        t0 = 0.35 + 0.9 * k
        d = t - t0
        amp = 3.0 + 5.0 * (k % 3) / 2.0
        env = np.where(np.abs(d) < 0.02, np.exp(-(d ** 2) / (2 * 0.008 ** 2)), 0.0)
        pitch = pitch + amp * env
        roll = roll + 0.4 * amp * env
    return np.column_stack([pitch, yaw, roll])


def quats(t: np.ndarray) -> np.ndarray:
    """(w, x, y, z) unit quaternions for the attitude, body convention.

    `attitude_euler` works in degrees because that is the unit every other
    stage of this pipeline speaks; scipy's `from_euler` takes RADIANS and has
    no unit argument. Passing degrees straight through makes the attitude 57.3x
    too large -- 0.0377 deg is read as 0.0377 rad = 2.16 deg -- which then
    integrates to an absurd ~30000 deg/s "rate" that looks like a spike in the
    synthetic IMU rather than the conversion error it is.
    """
    pitch, yaw, roll = np.radians(attitude_euler(t)).T
    xyzw = Rot.from_euler("zyx", np.column_stack([roll, yaw, pitch])).as_quat()
    return np.column_stack([xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]])


def body_rates(q: np.ndarray, dt: float) -> np.ndarray:
    """Body-frame angular velocity in deg/s that integrates to `q`.

    Taken as consecutive body-frame deltas rather than a numerical derivative
    of the Euler triple, so it is the rate that actually reproduces the
    attitude -- differentiating the angles and integrating them back would not
    be, because the composition is not linear.
    """
    r = Rot.from_quat(q[:, [1, 2, 3, 0]])
    rel = r[:-1].inv() * r[1:]
    rates = np.zeros((len(q), 3))
    rates[1:] = rel.as_rotvec() / dt
    return np.degrees(rates)


def accel_from_attitude(t: np.ndarray) -> np.ndarray:
    """Gravity as the accelerometer would see it under the attitude.

    Not zeros: the loader's gravity gate exists precisely because a mis-scaled
    or all-zero accel silently poisons an attitude estimate, and a synthetic
    stream should satisfy the gate rather than skirt it.
    """
    _, _, roll = attitude_euler(t).T
    r = np.radians(roll)
    return np.column_stack([np.zeros_like(r),
                            G * np.sin(r),
                            G * np.cos(r)])


def synthetic_stream(duration: float = 4.0, hz: float = IMU_HZ):
    """(time_s, gyro_deg_s, accel_m_s2) on the video's clock."""
    n = int(round(duration * hz))
    t = np.arange(n) / hz
    q = quats(t)
    gyro = body_rates(q, 1.0 / hz)
    return t, gyro, accel_from_attitude(t)


def synthetic_telemetry(duration: float = 4.0,
                        hz: float = IMU_HZ) -> Telemetry:
    t, gyro, accel = synthetic_stream(duration, hz)
    return Telemetry(time_s=t, gyro=gyro, accel=accel, model="synthetic")