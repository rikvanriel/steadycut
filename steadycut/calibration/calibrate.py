"""Work out a camera's gyro axis mapping from one of its own recordings.

The axis order is a hardware and firmware convention, so it should be the same
on every body of a model - but that is an assumption until it is checked on a
secondcamera.  This measuresthe image motion of a short baseline render and
correlates it against each gyro axis:

    horizontal image motion -> yaw
    vertical image motion   -> pitch
    in-plane rotation       -> roll

Run it on one file from another body and compare the mapping it reports.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def measure(source: str, start: float, duration: float, workdir: Path):
    """Render a static view and return image motion plus gyro on frame times."""
    from steadycut.metrics import shake
    from steadycut.metrics import rotate
    from steadycut.render.reframe import render, source_fps
    from steadycut.ingest.telemetry import read_telemetry

    clip = workdir / "_calib_base.mp4"
    render(source, clip, [__import__("path").PathPoint(t=0.0)], start=start,
           duration=duration, size=(960, 540), preset="ultrafast")

    frames = shake.frames(str(clip))
    motion = np.array([shake.shift(frames[i], frames[i + 1])
                       for i in range(len(frames) - 1)])
    # rotate.rotation_series uses the ORB estimator, validated against a
    # known rotation. tilt.tilt_series was used here originally; it failed
    # that validation and measures on a cropped frame, the same class of error.
    roll = np.asarray(rotate.rotation_series(str(clip))[0])

    imu = read_telemetry(source)
    fps = source_fps(source)
    frame_t = start + np.arange(len(motion)) / fps
    gyro = np.column_stack([
        np.interp(frame_t, imu.time_s, imu.gyro[:, a]) for a in range(3)
    ])
    return motion, roll, gyro, fps


def infer_mapping(motion, roll, gyro):
    """Pick, for each image motion, the gyro axis that explains it best.

    Split out from the reporting so the inference can be tested against
    synthetic data with a known answer.
    """
    rs_all = {}
    best = {}
    for target, series in (("dx (yaw)", motion[:, 0]),
                           ("dy (pitch)", motion[:, 1]),
                           ("tilt (roll)", roll)):
        rs = [float(np.corrcoef(gyro[:, a], series)[0, 1]) for a in range(3)]
        rs_all[target] = rs
        best[target] = int(np.argmax(np.abs(rs)))
    return best, rs_all


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source")
    ap.add_argument("--start", type=float, default=300.0)
    ap.add_argument("--duration", type=float, default=3.0)
    args = ap.parse_args()

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        motion, roll, gyro, fps = measure(args.source, args.start,
                                          args.duration, Path(tmp))

    print(f"\nimage motion: dx std {motion[:, 0].std():.2f} px   "
          f"dy std {motion[:, 1].std():.2f} px   "
          f"tilt std {roll.std():.2f} deg/frame")
    best, rs_all = infer_mapping(motion, roll, gyro)
    print("\ncorrelation of each gyro axis against image motion:")
    for target, rs in rs_all.items():
        pretty = "  ".join(f"axis{a}:{r:+.3f}" for a, r in enumerate(rs))
        print(f"  {target:13s} {pretty}   -> axis {best[target]}")

    mapping = (f"yaw=axis{best['dx (yaw)']}, "
               f"pitch=axis{best['dy (pitch)']}, "
               f"roll=axis{best['tilt (roll)']}")
    print(f"\nthis camera's mapping: {mapping}")
    print("expected on an X4:       yaw=axis1, pitch=axis2, roll=axis0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
