"""Motion telemetry: one neutral currency, one loader per format.

`Telemetry` is what every downstream stage consumes (time on the video's own
clock, gyro in deg/s, accel in m/s^2). `load(path, profile)` dispatches on the
camera profile's telemetry_format; with no profile it falls back to the .insv
trailer, which is the only format that exists in this repo today.

Every loader's output passes the same generic gates before use -- gravity
median near 9.81 m/s^2 (the cheapest mis-scale catcher: GoPro-style scale
factors live in-band and a wrong one poisons a whole calibration) and, when
the profile states a nominal rate, a 10x rate sanity check.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

GRAVITY_NOMINAL = 9.81     # m/s^2
GRAVITY_TOLERANCE = 1.0    # real X4 medians: 9.65 and 10.22 (pre-roll head load)


@dataclass(frozen=True)
class Telemetry:
    """Gyroscope and accelerometer samples on the video's timeline.

    Timestamps are seconds relative to the first video frame and may start
    slightly negative, because the IMU begins recording before the encoder
    does.  Gyro is deg/s, accel is m/s^2, both already scaled by the
    parser for this camera model.
    """

    time_s: np.ndarray
    gyro: np.ndarray
    accel: np.ndarray
    model: str

    @property
    def rate_hz(self) -> float:
        if len(self.time_s) < 2:
            return 0.0
        span = float(self.time_s[-1] - self.time_s[0])
        return (len(self.time_s) - 1) / span if span > 0 else 0.0

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0]) if len(self.time_s) else 0.0


class NoTelemetryError(RuntimeError):
    """The file carries no usable IMU data."""


# Parsed-telemetry cache keyed by (path, size, mtime): the same pattern the
# camera registry uses for identity. Without it the CLI's sync scan re-parses
# the ~2M-sample trailer once per lag (~29 full dumps per run), and the
# transient per-sample dicts are what got processes OOM-killed. The cache
# holds only the resulting ARRAYS (~100 MB per file), bounded FIFO.
_CACHE: dict = {}
_CACHE_MAX = 4


def _cache_key(path) -> tuple:
    p = Path(path)
    st = p.stat()
    return (str(p.resolve()), st.st_size, int(st.st_mtime))


def load(path: Path | str, profile=None) -> Telemetry:
    """Extract IMU samples per the profile's format, with the gates applied."""
    model = profile.model if profile is not None else "unknown"
    fmt = profile.telemetry_format if profile is not None else "insv_trailer"
    if fmt == "none":
        raise NoTelemetryError(f"{Path(path).name}: {model} has no embedded IMU")
    elif fmt in ("gpmf", "dji", "external"):
        raise NotImplementedError(
            f"{fmt} telemetry loader is not built yet (plan milestone D/E)")
    elif fmt != "insv_trailer":
        raise NoTelemetryError(f"unknown telemetry format {fmt!r}")

    try:
        key = _cache_key(path)
    except OSError:
        key = None
    telemetry = _CACHE.get(key) if key is not None else None
    if telemetry is None:
        from steadycut.ingest.telemetry import insv
        telemetry = insv.load(path)
        if key is not None:
            if len(_CACHE) >= _CACHE_MAX:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[key] = telemetry
    check_scales(telemetry, profile)
    return telemetry


# The historical name, kept so every existing caller keeps working.
read_telemetry = load


def check_scales(telemetry: Telemetry, profile=None) -> None:
    """Refuse telemetry whose units cannot be trusted.

    The gravity gate catches mis-scale (a parser that returned rad/s or g
    instead of deg/s and m/s^2 reads wildly off 9.81). The rate gate only runs
    when the profile states a nominal rate.
    """
    g = gravity_check(telemetry)
    if not (GRAVITY_NOMINAL - GRAVITY_TOLERANCE
            <= g <= GRAVITY_NOMINAL + GRAVITY_TOLERANCE):
        raise ValueError(
            f"accel scale looks wrong: median |a| = {g:.2f} m/s^2 "
            f"(expected ~{GRAVITY_NOMINAL}); refusing before it poisons a "
            "calibration")
    if profile is not None and profile.imu_rate_nominal_hz:
        rate = telemetry.rate_hz
        nominal = profile.imu_rate_nominal_hz
        if not (nominal / 10.0 <= rate <= nominal * 10.0):
            raise ValueError(
                f"imu rate {rate:.1f} Hz is outside 10x of the profile's "
                f"nominal {nominal:.0f} Hz")


def gravity_check(telemetry: Telemetry, sample_count: int = 3000) -> float:
    """Median accelerometer magnitude, which should sit near 9.81 m/s^2.

    A wildly different value means the samples were misparsed or misscaled,
    so this is the cheapest end-to-end check that the numbers are real.
    """
    head = telemetry.accel[:sample_count]
    if not len(head):
        return float("nan")
    return float(np.median(np.linalg.norm(head, axis=1)))


def covers_video(telemetry: Telemetry, video_duration_s: float,
                 tolerance_s: float = 2.0) -> bool:
    """Whether telemetry spans the footage, within a tolerance.

    Short telemetry means any camera path derived from it would run out
    part-way through the clip, so callers should refuse rather than
    silently freeze the camera.
    """
    return telemetry.time_s[-1] >= video_duration_s - tolerance_s
