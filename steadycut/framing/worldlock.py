"""Closed-loop world lock: hold the landscape still by watching the landscape.

The effect the project wants - landscape stationary, bike sweeping across
frame - is a WORLD-LOCK requirement, and the IMU cannot provide the needed
low-frequency reference: the gyro drifts, and the accelerometer is contaminated
by ride acceleration.

So the reference comes from the image instead. Render once, measure how the
distant scene actually drifts in frame, then drive the virtual camera to null
that drift. The IMU path stays underneath as a fast, short-term bridge; this
correction handles only the slow error the IMU cannot see.

Transfer, measured earlier: 14.2 px per degree of yaw, 17.8 px per degree of
pitch, so a shift can be converted to the rotation needed to undo it.
"""
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from steadycut.metrics import parallax
from steadycut.stabilization import path as P
from steadycut.render.reframe import render
from steadycut.ingest.telemetry import read_telemetry
# The window this study used, as defaults for its own runs. The RECORDING is
# not here: it is an argument, because this instrument answers a question about
# whatever file it is pointed at.
START, DUR, FOV, PITCH = 400.0, 5.0, 100.0, -35.0
S = ""                      # set from the command line; see run()
FPS = 29.97
PX_PER_DEG_YAW = 14.2      # measured: yaw +10 deg moves content +142 px
PX_PER_DEG_PITCH = 17.8    # measured: pitch +10 deg moves content -178 px


def far_field(video):
    """Per-frame shift of the distant scene, as (n, 2) pixels."""
    f = parallax.frames(str(video))
    out = []
    for i in range(len(f) - 1):
        dx, dy = parallax.shift(parallax.crop(f[i], parallax.FAR),
                                parallax.crop(f[i + 1], parallax.FAR))
        out.append((dx, dy))
    return np.array(out)


def drift(per_frame):
    """Accumulated drift of the scene within the frame, in pixels."""
    return np.cumsum(per_frame, axis=0)


def main():
    if not S:
        raise SystemExit("usage: worldlock.py <file.insv>")
    imu = read_telemetry(S)
    win = (imu.time_s >= START - 1) & (imu.time_s <= START + DUR + 1)
    t = imu.time_s[win] - START
    base = [replace(q, pitch=q.pitch + PITCH)
            for q in P.build_path_quat(t, imu.gyro[win], fov=FOV)]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        first = tmp / "base.mp4"
        render(S, first, base, start=START, duration=DUR, size=(960, 540),
               preset="ultrafast")
        pf = far_field(first)
        print(f"before: far-field motion {np.hypot(pf[:,0], pf[:,1]).mean():6.2f}"
              f" px/frame, drift {np.abs(drift(pf)).max():6.1f} px")

        for sign in (+1.0, -1.0):
            acc = drift(pf)
            times = np.arange(acc.shape[0]) / FPS
            fixed = []
            for q in base:
                i = int(np.clip(round(q.t * FPS), 0, acc.shape[0] - 1))
                yaw_c = sign * (-acc[i, 0] / PX_PER_DEG_YAW)
                pit_c = sign * (+acc[i, 1] / PX_PER_DEG_PITCH)
                fixed.append(replace(q, yaw=q.yaw + yaw_c, pitch=q.pitch + pit_c))
            out = tmp / f"lock{sign}.mp4"
            render(S, out, fixed, start=START, duration=DUR, size=(960, 540),
                   preset="ultrafast")
            pf2 = far_field(out)
            print(f"sign {sign:+.0f}: far-field motion"
                  f" {np.hypot(pf2[:,0], pf2[:,1]).mean():6.2f} px/frame,"
                  f" drift {np.abs(drift(pf2)).max():6.1f} px")


def run(argv):
    """Usage: python -m steadycut.framing.worldlock <file.insv>

    Sets the study's source from the command line and runs its sweep.
    """
    global S
    import sys
    if len(argv) < 1:
        print("usage: worldlock.py <file.insv>", file=sys.stderr)
        return 2
    S = argv[0]
    main()
    return 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(run(_sys.argv[1:]))