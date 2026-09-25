"""Can the world vertical be recovered from the IMAGE?

Requested as the fix for the world-referenced framing, after three accelerometer
attempts failed to establish the absolute reference. The idea is sound:
use the image for the absolute reference at coarse anchors and the gyro to interpolate
between them, so the gyro only has to be good short-term.

The cue is the forest itself. Tree trunks grow gravity-aligned, so in a perspective
view they converge on the VERTICAL VANISHING POINT, and the position of that point
relative to the principal point gives the camera's gravity-relative orientation. For a
flat view rendered at a known horizontal field of view the focal length in pixels is
known exactly, so the geometry inverts cleanly:

    f  = (w / 2) / tan(h_fov / 2)
    the vanishing point (vx, vy) maps to a direction in the camera frame:
        d = ((vx - cx) / f, (vy - cy) / f, 1)
    which, when the trunks really are gravity-aligned, IS the upward vertical.

This doubles as a cross-check the accelerometer never had: two independent instruments
measuring the same direction. If they agree the accelerometer may be sound after all;
if they disagree, the image is the more trustworthy of the two.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from steadycut.stabilization import path as P
from steadycut.render.reframe import render
from steadycut.ingest.telemetry import read_telemetry

import cv2

W, H = 1440, 1080          # 4:3, wide enough that the trunks converge measurably
H_FOV = 120.0
PITCH = -20.0


def frame_at(source, moment):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "f.mp4"
        render(source, out, [P.PathPoint(t=0.0, fov=H_FOV, yaw=0.0, pitch=PITCH)],
               start=moment, duration=0.1, size=(W, H), preset="ultrafast")
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(out), "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            capture_output=True, check=True).stdout
    return np.frombuffer(raw[: W * H], np.uint8).reshape(H, W)


def vertical_vanishing_point(gray, max_tilt_deg=35.0):
    """Least-squares intersection of near-vertical line segments."""
    edges = cv2.Canny(gray, 60, 160)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=80,
                           minLineLength=int(H * 0.25), maxLineGap=int(H * 0.05))
    if segs is None:
        return None, 0, 0
    segs = np.asarray(segs).reshape(-1, 4)   # (N,1,4) or (N,4) by version
    lines = []
    for x1, y1, x2, y2 in segs:
        dx, dy = x2 - x1, y2 - y1
        n = np.hypot(dx, dy)
        if n < 1e-6:
            continue
        tilt = np.degrees(np.arctan2(abs(dx), abs(dy)))     # 0 = vertical
        if tilt <= max_tilt_deg:
            lines.append((x1, y1, x2, y2))
    if len(lines) < 6:
        return None, len(lines), len(np.asarray(segs).reshape(-1, 4))
    # Each line ax + by = c; solve for the point satisfying all of them.
    A, b = [], []
    for x1, y1, x2, y2 in lines:
        dx, dy = x2 - x1, y2 - y1
        # line normal = (dy, -dx); through the point
        a, bb = dy, -dx
        A.append([a, bb])
        b.append(a * x1 + bb * y1)
    A, b = np.array(A, float), np.array(b, float)
    # Robust: drop the worst-fitting quarter and re-solve once.
    for _ in range(2):
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        resid = np.abs(A @ sol - b) / np.maximum(np.linalg.norm(A, axis=1), 1e-9)
        keep = resid <= np.percentile(resid, 75)
        if keep.sum() < 6:
            break
        A, b = A[keep], b[keep]
    return sol, len(lines), len(segs)


def main(argv=None):
    """Usage: python -m steadycut.framing.visual_vertical <file> <moment_s> ...

    The recording is an argument: this instrument answers a question about
    whatever file it is pointed at, so no file is named in the code.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print("usage: visual_vertical.py <file.insv> <moment_s> [moment_s ...]",
              file=sys.stderr)
        return 2
    source = argv[0]
    imu = read_telemetry(source)
    for moment in [float(x) for x in argv[1:]]:
        gray = frame_at(source, moment)
        vp, used, total = vertical_vanishing_point(gray)
        print(f"\nt={moment:.1f}s   h_fov {H_FOV:.0f}  view pitch {PITCH:.0f}")
        print(f"   Hough segments {total}, near-vertical accepted {used}")
        if vp is None:
            print("   no usable vanishing point")
            continue
        f = (W / 2.0) / np.tan(np.radians(H_FOV / 2.0))
        cx, cy = W / 2.0, H / 2.0
        d = np.array([(vp[0] - cx) / f, (vp[1] - cy) / f, 1.0])
        d /= np.linalg.norm(d)
        print(f"   vanishing point at ({vp[0]:.0f}, {vp[1]:.0f})"
              f"  focal {f:.0f} px")
        # The world vertical in the camera frame. Camera y is DOWN in this
        # convention, so a downward camera makes the UP vanishing point appear ABOVE
        # the centre, i.e. a negative image y offset.
        pitch_from_image = np.degrees(np.arctan2(-d[1], d[2]))
        roll_from_image = np.degrees(np.arctan2(d[0], -d[1]))
        print(f"   from the image:  vertical {pitch_from_image:+.1f} deg"
              f"  (positive = camera aimed UP from horizontal? see note)"
              f"  lateral {roll_from_image:+.1f} deg")
        # Compare with the accelerometer, whose absolute value is the open question.
        w = (imu.time_s >= moment - 1) & (imu.time_s <= moment + 1)
        a = imu.accel[w].mean(axis=0)
        mag = np.linalg.norm(a)
        print(f"   accelerometer mean vector {np.round(a,2)}  |a| {mag:.2f}"
              f"  (g = 9.81)")
        pitch_accel = np.degrees(np.arctan2(a[0], np.hypot(a[1], a[2])))
        print(f"   from the accelerometer: axis0/hypot(1,2) = {pitch_accel:+.1f} deg")


if __name__ == "__main__":
    sys.exit(main())