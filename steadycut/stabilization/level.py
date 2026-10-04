"""Gravity levelling: the static roll offset gyro stabilisation cannot see.

The gyro path is deliberately RELATIVE (raw -> smooth): it follows slow
motion as its trend, so a constant mount roll sails through untouched.
Only gravity knows "down". This module measures the static roll offset
from the accelerometer median and returns it as an ADDITIVE shift on the
renderer's roll slot.

Sign convention (empirical, settled by eye on two recordings, NOT derived):
    shift = degrees(atan2(-gx, gy)),  new_roll = old_roll + shift
on body-frame (right, up, forward) median gravity. Jan ski (+66.6 deg) and
Feb ski (-104 deg) both render level with this form; the naive
atan2(gx, -gy) reads exactly 180 off (verified: rendered inverted).

No rest gate: a 30 s riding median is stable to ~1-3 deg (measured on MTB
0521 across 25 min), and true rest does not exist mid-ride (zero 2 s rest
windows in 30 min of MTB). Linear accel averages out; gravity persists.

Gyro drift does NOT force re-anchoring: the pipeline's own quaternion
integrator agrees with gravity medians to ~2 deg over 25 min (drift_check).
One shift per clip suffices; a mid-recording helmet shift (crash) is the
only case that wants re-measurement, and that is future work, not drift.
"""
import numpy as np


def shift_from_gravity(gvec, axis_map) -> float:
    """Additive roll shift (degrees) for a body-frame gravity median.

    gvec: median accel in RAW IMU columns; axis_map reorders to body
    (right, up, forward), the same frame correction_axes integrates in.
    """
    cols = [axis_map.pitch[0], axis_map.yaw[0], axis_map.roll[0]]
    gx, gy, _ = (float(gvec[c]) for c in cols)
    return float(np.degrees(np.arctan2(-gx, gy)))


def measure_shift(imu, start: float, duration: float, axis_map,
                  window_s: float = 30.0) -> dict:
    """Shift + diagnostics over the clip span (capped at window_s each end).

    Measures on the first window_s seconds of the clip: the helmet angle at
    clip start is what the clip needs, and the drift check shows it holds
    for tens of minutes afterward.
    """
    span = min(duration, window_s)
    m = (imu.time_s >= start) & (imu.time_s < start + span)
    g = np.median(imu.accel[m], axis=0)
    return {"shift_deg": shift_from_gravity(g, axis_map),
            "g_norm": float(np.linalg.norm(g)),
            "span_s": float(span), "n": int(m.sum())}


def apply_shift(points, shift_deg: float):
    """Return points with the levelling shift added to every roll slot."""
    import dataclasses
    return [dataclasses.replace(p, roll=p.roll + shift_deg) for p in points]
