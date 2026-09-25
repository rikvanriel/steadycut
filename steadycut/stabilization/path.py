"""Turn head-motion telemetry into a camera path that stabilises the world.

The camera is bolted to the head, so the recorded sphere rotates with it. Three
rendering choices follow, and telling them apart is the whole job:

    crop fixed at the camera's own forward  -> raw, all shake visible
    crop counter-rotates the head fully     -> world locked, gaze thrown away
    crop counter-rotates only the fast part -> shake removed, slow gaze kept

The third is what we want, and it is the one the plan calls follow_gaze.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PathPoint:
    t: float
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    fov: float = 100.0


def _zero_phase_lowpass(x: np.ndarray, window: int) -> np.ndarray:
    """Smooth with a symmetric kernel.

    Convolution with a symmetric kernel has no phase shift, which is the
    discrete equivalent of the filtfilt trick the plan calls for. It matters:
    a causal filter would lag the pan, so the camera would start turning after
    the rider already had.
    """
    window = max(1, int(window) | 1)
    if window == 1 or x.size <= window:
        return x.copy()
    kernel = np.hanning(window)
    kernel /= kernel.sum()
    pad = window // 2
    padded = np.pad(x, pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: x.size]


def _integrate(rate_deg_s: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Turn an angular rate into a running angle, in degrees."""
    return np.cumsum(rate_deg_s * dt)


def build_path(
    t: np.ndarray,
    gyro: np.ndarray,
    accel: np.ndarray | None = None,
    *,
    cutoff_s: float = 0.3,
    gain: float = 1.0,
    level_roll: bool = False,
    roll_gain: float = 1.0,
    axis_map=None,
) -> list[PathPoint]:
    """Counter-rotate the fast part of the head motion.

    `cutoff_s` is the period, in seconds, below which motion counts as shake
    and gets cancelled; `gain` scales how much of the cancelling is applied.
    Larger cutoff means more of the head's motion is treated as shake.

    The default of 0.3 s is measured, not guessed. Sweeping it on real mountain
    bike footage gives a clear minimum there (shake 3.27 against 4.39 raw, a
    25.5 percent reduction), with 0.5 s slightly worse and 1.0 s barely better
    than raw. Cancelling motion slower than about 1 s actively destabilises the
    image: the landscape ends up less steady than with no correction at all.
    This matches the gyro spectrum, whose shake energy sits in the 2-10 Hz band,
    i.e. periods of 0.1-0.5 s, so the optimum falls inside it.

    The practical reading for framing: low-frequency head motion is where the
    rider is *looking*, and it should be followed rather than cancelled. That is
    what keeps the view pointed where they were going.
    """
    if t.size < 3:
        return []

    dt = np.gradient(t)
    angles = np.stack(
        [_integrate(gyro[:, axis], dt) for axis in range(gyro.shape[1])]
    )

    # The window must be expressed in IMU samples.  Deriving it from the
    # video frame rate instead makes it orders of magnitude too short on a
    # 1 kHz sensor, which leaves the shake component near zero and silently
    # disables the whole correction.
    dt_median = float(np.median(np.diff(t)))
    sample_rate = 1.0 / dt_median if dt_median > 0 else 1.0
    window = max(3, int(round(cutoff_s * sample_rate)))
    window += 1 - window % 2  # odd length keeps the filter centred
    path = np.empty_like(angles)
    for axis in range(angles.shape[0]):
        slow = _zero_phase_lowpass(angles[axis], window)
        # Sign convention: the correction is ADDED to the gyro-derived angle.
        # It is counter-intuitive but measured. Each channel was driven on its
        # own and judged by the image motion it is meant to cancel:
        #
        #   channel        one sign        the other
        #   yaw (dx)       +48.8%          -23.6%
        #   pitch (dy)     +46.0%          -25.6%
        #   roll (rot)     +30.0%          -16.9%
        #
        # The sign that helps is the OPPOSITE of the one that reads correctly on
        # paper. The reason is the stitched sphere's orientation: the two fisheye
        # inputs are stacked rear-then-front (see reframe.py), which rotates the
        # sphere by 180 degrees relative to the camera's own axes, so a rotation
        # the gyro reports arrives at the render with the opposite sense.
        path[axis] = (angles[axis] - slow) * gain

    # Measured on both clips by correlating image motion against each axis:
    # axis 1 tracks horizontal image motion (r = -0.80 on MTB, -0.98 on ski) and
    # so is yaw; axis 2 tracks *vertical* image motion (r = -0.83) and so is
    # pitch; axis 0 is what remains, roll.  The ski clip alone was ambiguous -
    # axis 2 correlated with both image axes - and reading it as roll put the
    # largest motion (vertical, 5x horizontal) beyond reach of the correction.
    # The COLUMNS come from the camera profile's axis map now; the SIGN table
    # above stays in this formulation (the quaternion path in correction_axes
    # is where map signs are applied -- the two paths' sign conventions differ
    # because this one subtracts the smooth component).
    if axis_map is None:
        raise ValueError(
            "no measured axis map: pass the one the camera registry loaded "
            "for this body; an unknown body is measured from its own footage "
            "on first use")
    yaw_a, pitch_a, roll_a = axis_map.yaw[0], axis_map.pitch[0], axis_map.roll[0]

    # Horizon levelling is not a frequency filter like the others: the camera
    # should hold level in absolute terms, so this cancels roll outright using
    # a reference that does not drift.  The gyro gives the fast part, gravity
    # pins the long-term average (see complementary_roll) -- gravity alone is
    # unusable here because ride acceleration corrupts it.
    if level_roll:
        roll_ref = complementary_roll(t, gyro[:, roll_a], accel)
        path[roll_a] = -roll_ref * roll_gain

    return [
        PathPoint(
            t=float(t[i]),
            yaw=float(path[yaw_a][i]),
            pitch=float(path[pitch_a][i]),
            roll=float(path[roll_a][i]),
        )
        for i in range(t.size)
    ]


def thin(points: list[PathPoint], every_s: float = 0.5) -> list[PathPoint]:
    """Keep every nth point; sendcmd interpolates the rest."""
    if not points:
        return []
    out = [points[0]]
    for point in points:
        if point.t - out[-1].t >= every_s:
            out.append(point)
    if out[-1] is not points[-1]:
        out.append(points[-1])
    return out


def stabilise(
    source: str,
    output: str,
    *,
    start: float = 0.0,
    duration: float = 6.0,
    cutoff_s: float = 0.3,
    gain: float = 1.0,
    thin_s: float = 0.5,
) -> list:
    """Render a clip with the fast part of the head motion cancelled."""
    from steadycut.render.reframe import render
    from steadycut.ingest.telemetry import read_telemetry

    imu = read_telemetry(source)
    window = (imu.time_s >= start) & (imu.time_s <= start + duration)
    points = build_path(
        imu.time_s[window] - start,
        imu.gyro[window],
        cutoff_s=cutoff_s,
        gain=gain,
    )
    if not points:
        raise SystemExit("no telemetry in that window")
    args = render(
        source,
        output,
        thin(points, thin_s),
        start=start,
        duration=duration,
    )
    return args


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Gyro-stabilised reframe")
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--cutoff", type=float, default=1.5,
                        help="seconds; motion faster than this counts as shake")
    parser.add_argument("--gain", type=float, default=1.0)
    args = parser.parse_args()
    stabilise(
        args.source, args.output, start=args.start, duration=args.duration,
        cutoff_s=args.cutoff, gain=args.gain,
    )


if __name__ == "__main__":
    _main()


def gravity_roll(accel: np.ndarray) -> np.ndarray:
    """Roll implied by gravity, in degrees.

    Only trustworthy at low frequency: the accelerometer measures proper
    acceleration, so hard riding corrupts it.  Its virtue is that it does not
    drift, which is exactly what integrating the gyro cannot offer.
    """
    # Gravity sits along +y on this camera: the mean accelerometer vector over
    # 30 s of riding is [+3.30 +8.71 +2.29] with |a| = 9.59 m/s^2, i.e. the
    # down axis is y.  Roll is rotation about the view axis z, so it is
    # atan2(ax, ay).  An earlier attempt to pick this by correlating against
    # measured image tilt chose atan2(ay, az); that correlation was worthless
    # because the tilt estimator is unreliable on real footage (see tilt.py),
    # and it produced nonsense references of 38..115 degrees.
    ax, ay = accel[:, 0], accel[:, 1]
    return np.degrees(np.arctan2(ax, ay))


def complementary_roll(t: np.ndarray, rate_deg_s: np.ndarray,
                       accel: np.ndarray, tau_s: float = 4.0) -> np.ndarray:
    """Blend gyro integration (fast, drifts) with gravity (slow, noisy).

    Standard complementary filter: the gyro carries the high frequencies, the
    accelerometer pins the long-term average so the horizon cannot slowly
    rotate away.
    """
    dt = np.gradient(t)
    ref = gravity_roll(accel)
    alpha = dt / max(tau_s, 1e-9)
    out = np.empty_like(rate_deg_s)
    est = ref[0]
    for i in range(len(t)):
        est += rate_deg_s[i] * dt[i]
        if alpha[i] > 1.0:
            alpha[i] = 1.0
        est = (1.0 - alpha[i]) * est + alpha[i] * ref[i]
        out[i] = est
    return out - np.mean(out)


def build_path_quat(
    t: np.ndarray,
    gyro: np.ndarray,
    *,
    # Re-measured on the FIXED renderer. Until reset_rot=1 was set, v360 silently
    # discarded most of each command file, so every earlier sweep here described a
    # configuration the code did not produce. Residual motion is FLAT from 0.05 to
    # 1.5 Hz, and the old 3.18 Hz sits OUTSIDE that plateau - rotation 0.62 against
    # 0.42, about 30 percent worse, with dy 8.64 against 8.47.
    #
    # 0.5 Hz sits mid-plateau and is chosen for behaviour as well as number: a lower
    # corner follows the rider's head turns less and the direction of travel more,
    # which is what the rider asked for. Checked on a 4 s window, so the penalty for
    # a corner low enough to cancel intentional turns is not visible here - it has a
    # period longer than the window. Re-check on a longer clip before going lower.
    corner_hz: float = 0.5,
    fov: float = 100.0,
    every_s: float = 0.05,
    axis_map=None,
) -> list[PathPoint]:
    """Camera path from the quaternion orientation pipeline.

    Preferred over `build_path`'s independent per-axis filtering: the gyro is
    integrated into a quaternion orientation, low passed in orientation space,
    and the correction applied as the RELATIVE rotation between the smoothed and
    raw orientations. That avoids filtering Euler axes separately, which produces
    non-linear behaviour in the resulting orientation.

    Parameterised by CORNER FREQUENCY rather than a time constant. The two are
    not interchangeable: an exponential with time constant tau has a corner at
    1/(2*pi*tau), so a time constant copied from a moving-average window cancels
    far too much. Measured optimum is about 3 Hz.
    """
    from steadycut.stabilization import orientation

    tau = 1.0 / (2.0 * np.pi * max(corner_hz, 1e-6))
    yaw, pitch, roll = orientation.correction_axes(t, gyro, tau, axis_map)
    points = [
        PathPoint(t=float(a), yaw=float(b), pitch=float(c), roll=float(d), fov=fov)
        for a, b, c, d in zip(t, yaw, pitch, roll)
    ]
    return thin(points, every_s)
