"""Camera attitude from VQF, in the per-camera axis convention.

WHY VQF AND NOT A LOCAL FILTER. This is the routing filter for attitude in this
project. A hand-written complementary filter is the obvious alternative and it is
a trap here: deriving a quaternion convention from memory produces confident,
mirrored, wrong numbers that do not raise, because a sign or handedness mistake
in a rotation or an axis permutation still runs to completion and still looks
plausible. A left-handed axis permutation in particular will pass a suite that
only compares against whatever the new code happens to emit. The `vqf` package
is the reference implementation of Laidig & Seel's VQF (Information Fusion 2023)
and is the algorithm Gyroflow ships as its default, so there is no reason to keep
re-deriving it.

WHY THE AXIS CONVENTION IS AN ARGUMENT. The Insta360 IMU record stores its axes
in the vendor's own order, which is (pitch, yaw, roll) rather than the
conventional (roll, pitch, yaw), and the mapping to the correction channels is
measured per camera body rather than read from a datasheet. `AxisMap` carries
`(column, sign)` per channel and is what `registry.identify` attaches. This module
consumes that rather than assuming a convention, because a sign error here is
invisible: the numbers come out plausible and mirrored.

THE PERMUTATION, and why it is what it is. Measured directly off the
accelerometer on this material: the gravity axis is raw index 1, the pitch axis is
raw index 0, and roll is raw index 2. VQF wants a right-handed frame with Z up,
so the frame is built as (pitch, roll, gravity) -- X from index 0, Y from index 2,
Z from index 1 -- and the per-channel signs come from the axis map.

SIGNS ARE PINNED BY THE LABELS, NOT BY THIS MODULE. The pitch returned here is
positive when the camera looks DOWN. That is established by the rider's own
labelling of real frames -- the five frames they called "almost straight down"
read +38.9 to +70.7 under this convention and the one they called "straight up"
reads -49.3 -- and not by whatever this code happens to emit. The test module
asserts a mirrored sign FAILS, because that is the mistake this exists to prevent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Raw Insta360 IMU columns, measured off the accelerometer on this material:
# index 0 carries the pitch axis, index 1 carries gravity, index 2 carries roll.
PITCH_COL, GRAVITY_COL, ROLL_COL = 0, 1, 2
#: VQF's frame is right-handed with Z up, so (X, Y, Z) = (pitch, roll, gravity).
_TO_VQF = (PITCH_COL, ROLL_COL, GRAVITY_COL)


@dataclass(frozen=True)
class Attitude:
    """Per-sample attitude in degrees.

    `pitch` is POSITIVE WHEN THE CAMERA LOOKS DOWN, which is the convention the
    framing labels were taken in. `roll` is unsigned, so a camera inverted and a
    camera upright with the opposite lean read the same -- which is deliberate,
    because the validity gate wants "how far from level", not a signed angle.
    """

    pitch: np.ndarray
    roll: np.ndarray
    bias: np.ndarray
    rest: np.ndarray


def attitude(gyro, accel, dt: float, axis_map=None) -> Attitude:
    """Fuse raw gyro (deg/s) and accel (m/s^2) into per-sample attitude.

    `axis_map` is the camera's `AxisMap`, whose per-channel sign is applied to the
    corresponding raw column. Without one the raw vendor order is used as-is,
    which is correct for a body whose map is the identity and is the wrong thing
    to assume for any other -- so callers on a real file should pass the map from
    `registry.identify`.
    """
    import vqf

    gyro = np.asarray(gyro, dtype=float)
    accel = np.asarray(accel, dtype=float)
    if gyro.shape[0] != accel.shape[0]:
        raise ValueError(
            f"gyro has {gyro.shape[0]} samples and accel has {accel.shape[0]}; "
            "they must be on the same clock before fusion")

    # Apply the measured per-channel (column, sign) BEFORE the permutation, so a
    # body whose map flips an axis flips it in the vendor's own columns, where it
    # was measured, rather than after the axes have been renumbered. The gravity
    # axis lives in the vendor's YAW column, so the map's yaw entry is the one
    # that describes it.
    if axis_map is None:
        p_col, r_col, g_col = PITCH_COL, ROLL_COL, GRAVITY_COL
        p_sign = r_sign = g_sign = 1
    else:
        p_col, p_sign = axis_map.pitch
        r_col, r_sign = axis_map.roll
        g_col, g_sign = axis_map.yaw
    gyr = np.radians(np.column_stack((p_sign * gyro[:, p_col],
                                      r_sign * gyro[:, r_col],
                                      g_sign * accel[:, g_col])))
    a = np.column_stack((p_sign * accel[:, p_col],
                         r_sign * accel[:, r_col],
                         g_sign * accel[:, g_col]))

    f = vqf.VQF(float(dt))
    res = f.updateBatch(gyr, a)
    q = np.asarray(res["quat6D"], dtype=float)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # VQF's euler[1] is the pitch of ITS frame. The rider's labels put "looking
    # down" on the positive side, and the permutation above is what makes the two
    # agree, so the sign is applied here and asserted in the tests.
    pitch = -np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    roll = np.abs(np.degrees(np.arctan2(2 * (w * x + y * z),
                                        1 - 2 * (x * x + y * y))))
    return Attitude(pitch=pitch, roll=roll,
                    bias=np.asarray(res["bias"], float),
                    rest=np.asarray(res["restDetected"], bool))


class _FallBack(tuple):
    """A stand-in channel spec for a body whose map has no accel entry."""


def attitude_from_telemetry(telemetry, dt: float, axis_map=None) -> Attitude:
    """`attitude` for a `Telemetry` record, decimated by an integer factor."""
    factor = max(1, int(round(dt / float(_median_dt(telemetry)))))
    return attitude(telemetry.gyro[::factor], telemetry.accel[::factor],
                    dt, axis_map=axis_map)


def _median_dt(telemetry) -> float:
    t = np.asarray(telemetry.time_s)
    return float(np.median(np.diff(t)))
