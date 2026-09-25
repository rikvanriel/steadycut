"""Calibrate v360 yaw/pitch/roll transfer functions."""
import numpy as np
from pathlib import Path

from steadycut.metrics import rotate
from steadycut.metrics import shake
from steadycut.render.reframe import ControlPoint, render

W, H, FOV = 960, 540, 100.0


def still(source, out, start, **angles):
    pt = ControlPoint(t=0.0, fov=FOV, **angles)
    render(source, out, [pt], start=start, duration=0.2,
           size=(W, H), preset="ultrafast")


def first_frame(video):
    # Full frame at full resolution: shake.frames() crops to the top half at
    # 480x270, where a 96 px shift cannot be measured reliably.
    return rotate.frames(str(video))[0]


def translate(a, b):
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    fa = np.fft.rfft2((a - a.mean()) * win)
    fb = np.fft.rfft2((b - b.mean()) * win)
    r = fa * np.conj(fb)
    r /= np.maximum(np.abs(r), 1e-9)
    peak = np.fft.irfft2(r, s=(h, w))
    y, x = np.unravel_index(np.argmax(peak), peak.shape)
    if y > h // 2:
        y -= h
    if x > w // 2:
        x -= w
    return float(x), float(y)


def main(source, start=400.0, delta=10.0):
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base = tmp / "base.mp4"
        still(source, base, start)
        ref = first_frame(base)
        px = delta / FOV * W
        for axis in ("yaw", "pitch", "roll"):
            out = tmp / f"{axis}.mp4"
            still(source, out, start, **{axis: delta})
            img = first_frame(out)
            dx, dy = translate(ref, img)
            rot = rotate.rotation_between(ref, img) or 0.0
            print(f"  {axis:5s} +{delta:.0f} deg -> dx {dx:+7.1f} dy {dy:+7.1f} "
                  f"rot {rot:+7.1f}   (expected translation {px:.1f} px)")


if __name__ == "__main__":
    import sys

    main(sys.argv[1])
