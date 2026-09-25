"""Map the gaze/world-lock tradeoff.

`cutoff_s` decides what counts as shake rather than intentional motion: larger
values cancel more of the head's movement, which steadies the landscape but
stops the view following where the rider looks. This measures both ends of that
tradeoff so the choice can be made from numbers rather than by feel.

Judged with the shake metric, which has agreed with itself across every run,
rather than with the rotation estimate, which does not.
"""
import tempfile
from pathlib import Path

import numpy as np

from steadycut.harness import evaluate
from steadycut.stabilization import path as P
from steadycut.ingest.telemetry import read_telemetry


def sweep(source, start=400.0, duration=4.0, cutoffs=(0.1, 0.3, 0.5, 1.0, 2.0, 5.0)):
    imu = read_telemetry(str(source))
    win = (imu.time_s >= start - 1) & (imu.time_s <= start + duration + 1)
    t = imu.time_s[win] - start
    gyro = imu.gyro[win]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        out = tmp / "raw.mp4"
        from steadycut.render.reframe import ControlPoint, render
        render(source, out, [ControlPoint(t=0.0, fov=evaluate.FOV)],
               start=start, duration=duration, size=(960, 540), preset="ultrafast")
        base_shake, base_roll = evaluate.metrics(out)
        print(f"raw         shake {base_shake:6.3f}   roll {base_roll:6.3f}")
        for c in cutoffs:
            pts = P.build_path(t, gyro, None, cutoff_s=c)
            pts = P.thin(pts, 0.05)
            out = tmp / f"c{c}.mp4"
            render(source, out, pts, start=start, duration=duration,
                   size=(960, 540), preset="ultrafast")
            sh, ro = evaluate.metrics(out)
            print(f"cutoff {c:4.1f}s  shake {sh:6.3f}   roll {ro:6.3f}"
                  f"   shake {100 * (sh / base_shake - 1):+5.1f}%")


if __name__ == "__main__":
    import sys

    sweep(sys.argv[1])
