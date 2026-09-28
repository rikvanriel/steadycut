"""A gravity-levered attitude filter for the LEVEL axis, in the shape the
shipping tool uses, and the two corrections this footage demands.

WHY THIS SHAPE. Three hand-rolled filters failed on this footage before, and the
skill records why each one failed, so the shape here is chosen against those
failures rather than by taste:

* `attitude.py` in this package gates the correction on a 20-degree INNOVATION.
  That rejects exactly the samples that would fix a large error, so it accepts
  0.6-1.4 percent of them, the correction is inert, and what remains is a
  gyro-only propagation drifting 41-78 degrees from measured gravity.
* A ROBUST WEIGHT (`1/(1+(theta/theta0)^2)`) fails in the same way more gently:
  a scale set at 15-90 degrees down-weights the first correction, when the error
  is largest, to 0.03-0.5, and the attitude never leaves its start.

The filter used here has NEITHER. The correction is a fixed multiplicative pull
on the angular RATE, so a large error always gets a large pull and the filter
can re-anchor from any starting attitude. Per sample:

    accWorld  = rotate(q, a_hat)          # measured up, expressed in world
    corrWorld = cross(accWorld, world_up) # = sin(theta) in the world frame
    weight    = 1/tau
    omega    += weight * rotate(conj(q), corrWorld)
    q         = q * rate_to_quat(omega, dt)

`corrWorld` is a cross product of two UNIT vectors, so its magnitude is
sin(theta) and `weight * sin(theta)` is a rate. The pull therefore vanishes
exactly at level, which is what makes it a fixed point rather than a drift.

TWO CORRECTIONS TO THE TRANSCRIPTION, both forced by measured facts about THIS
body, and both of them the difference between working and silently doing nothing:

1. **The magnitude window is relative to the window's own median.** The
   transcription gates on `0.81 < |a|^2/g^2 < 1.21`, which assumes |a| reads
   1 g correctly. This body reads a median 10.55 m/s^2, about 7 percent high,
   which parks |a|^2/g^2 at 1.16 -- against the TOP edge -- and the gate then
   rejects EVERY sample. The correction switches itself off with no error
   anywhere, which is the same failure as the hard innovation gate by a
   different route. Scaling by the window's own median makes the window a test
   for LINEAR ACCELERATION, which is what it is for, and leaves the body's
   calibration error as a constant scale that cancels.

2. **tau is long.** 1.7 s is Gyroflow's value and it suits a short bounce. Here
   the contaminant is 0.3-0.5 g of sustained lateral acceleration while leaning
   through a corner, which tilts apparent gravity by 17-27 degrees; a long tau
   averages that transient out while the gyro carries everything fast. The
   project's own measurements put the traced plateau at 12-20 degrees with
   tau 1.7, so the sweep is 40/20/10/5/1.7 and the choice is measured, not
   assumed.

WHAT THIS DOES NOT CLAIM. A small INNOVATION is not the target here: a slow
anchor legitimately has a large innovation while it is still converging, and
treating that as a fault is what made the gated variants unusable. The target is
the levelled bank on rendered video. Judged by the innovation, this filter looks
worse than a gate that does nothing at all.
"""
from __future__ import annotations

import numpy as np

from steadycut.stabilization import orientation as O

G = 9.81
WORLD_UP = np.array([0.0, 0.0, 1.0])

# The magnitude window, as a tolerance on |a| about the window's own median.
# A riding body deviates from its own median by tens of percent under
# acceleration; a gate this tight is a test for a still interval, which is not
# what this footage offers.
MAG_TOL = 0.25

# First `warm_s` seconds pull hard, as the transcription does, so a bad initial
# attitude is cleared before the slow anchor takes over.
WARM_S = 1.5
WARM_TAU = 0.1


def _rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    qv = np.array([x, y, z])
    t2 = 2.0 * np.cross(qv, v)
    return v + w * t2 + np.cross(qv, t2)


def magnitude_window(accel: np.ndarray, tol: float = MAG_TOL) -> np.ndarray:
    """Which samples are free of gross LINEAR acceleration.

    Relative to the window's own median, never to 1 g: this body reads about
    7 percent high and an absolute gate rejects everything. What is left is a
    test for acceleration, which is the question being asked.
    """
    a = np.linalg.norm(accel, axis=1)
    med = float(np.median(a))
    if not np.isfinite(med) or med <= 1e-9:
        return np.zeros(len(a), dtype=bool)
    lo, hi = med * (1.0 - tol), med * (1.0 + tol)
    return (a >= lo) & (a <= hi) & (a > 1e-9)


def level_attitude(t, gyro, accel, tau_s: float = 20.0, q0=None,
                   sign: float = 1.0, mag_tol: float = MAG_TOL,
                   warm_s: float = WARM_S, warm_tau: float = WARM_TAU):
    """Body-to-world attitude per sample, from gyro plus a gravity re-anchor.

    `gyro` is in the project's parser frame and DEGREES PER SECOND, the same
    units `orientation.integrate` takes. `accel` is specific force in m/s^2 on
    the sensor axes, which for this body are already the parser frame.

    `sign` exists for the control test and for nothing else: a flipped
    correction sign must DIVERGE, and the divergence is the evidence that the
    filter is strong enough to reveal its own sign error rather than settling at
    a plausible plateau. Production callers leave it at +1.
    """
    t = np.asarray(t, dtype=float)
    gyro = np.asarray(gyro, dtype=float)
    accel = np.asarray(accel, dtype=float)
    n = len(t)
    if n == 0:
        return np.zeros((0, 4))

    ok = magnitude_window(accel, mag_tol)
    q = np.array([1.0, 0.0, 0.0, 0.0]) if q0 is None else np.asarray(q0, float)
    q = O._normalise(q)
    out = np.empty((n, 4))
    out[0] = q
    t0 = float(t[0])

    for i in range(1, n):
        dt = float(t[i] - t[i - 1])
        if not 0.0 < dt < 0.5:
            # A gap in the samples is not a rotation; hold the attitude rather
            # than integrating an unknown rate across it.
            out[i] = q
            continue

        # Gyro propagation, midpoint rate between the samples that bracket dt.
        rate = 0.5 * (gyro[i - 1] + gyro[i]) * np.pi / 180.0
        if ok[i]:
            a_hat = accel[i] / np.linalg.norm(accel[i])
            acc_world = _rot(q, a_hat)
            corr_world = np.cross(acc_world, WORLD_UP)
            tau = warm_tau if (t[i] - t0) < warm_s else tau_s
            # corr_world is a cross product of unit vectors, so its magnitude
            # is sin(theta): this pull is a RATE, and it vanishes at level.
            rate = rate + sign * (1.0 / tau) * _rot(O._conj(q), corr_world)
        dq = O.from_rotvec((rate * dt)[None, :])[0]
        q = O._normalise(O._mul(q, dq))
        out[i] = q
    return out


def world_tilt(q: np.ndarray, view_axis=None) -> np.ndarray:
    """Roll of the camera about its VIEW axis, relative to the world vertical.

    The bank a viewer sees is the roll about the axis the camera is looking
    along, not about a body axis -- and the distinction matters, because a filter
    that cannot observe yaw drifts the view axis and then reports a "bank" that
    is really yaw drift. Taking the axis as an ARGUMENT is what keeps the two
    apart: pass the DELIVERED view axis and this measures the delivered bank.
    """
    x = np.array([1.0, 0.0, 0.0]) if view_axis is None else np.asarray(
        view_axis, dtype=float)
    x = x / np.linalg.norm(x)
    up = np.array([0.0, 1.0, 0.0])
    out = np.empty(len(q))
    for i, qi in enumerate(q):
        v = _rot(qi, x)
        u = _rot(qi, up)
        perp = WORLD_UP - float(WORLD_UP @ v) * v
        n = np.linalg.norm(perp)
        if n < 1e-9:
            out[i] = 0.0            # looking straight up or down: undefined
            continue
        perp = perp / n
        side = np.cross(v, perp)
        out[i] = np.degrees(np.arctan2(float(u @ side), float(u @ perp)))
    return out
