"""The .insv trailer loader: Insta360 IMU samples via telemetry-parser.

Insta360 stores IMU samples in a trailer appended to the .insv file, and
telemetry-parser already knows how to find and scale them -- including the
offset that puts timestamps on the video's own clock, which is the fiddly
part.  So this module is a thin, well-tested wrapper rather than a parser:
it converts to plain arrays, states the units, and reports whether the data
actually covers the footage.

The IMU sits on the rider's helmet, so these samples describe *head*
orientation -- not the bike's attitude and not the direction of travel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import telemetry_parser

from steadycut.ingest.telemetry import NoTelemetryError, Telemetry


def load(path: Path | str) -> Telemetry:
    """Extract IMU samples from an .insv trailer, or raise if it has none."""
    path = Path(path)
    parser = telemetry_parser.Parser(str(path))
    samples = parser.normalized_imu()
    if not samples:
        raise NoTelemetryError(f"{path.name}: no IMU samples")

    times, gyros, accels = [], [], []
    for sample in samples:
        gyro = sample.get("gyro")
        accel = sample.get("accl")
        stamp = sample.get("timestamp_ms")
        if gyro is None or accel is None or stamp is None:
            continue
        times.append(stamp / 1000.0)
        gyros.append(gyro)
        accels.append(accel)

    if not times:
        raise NoTelemetryError(f"{path.name}: IMU record present but unreadable")

    return Telemetry(
        time_s=np.asarray(times, dtype=np.float64),
        gyro=np.asarray(gyros, dtype=np.float64),
        accel=np.asarray(accels, dtype=np.float64),
        model=parser.model or "unknown",
    )
