"""Measure near-field versus far-field apparent motion in a render.

The effect the project exists for is that the front wheel sweeps across the
frame far faster than the landscape moves behind it. That is parallax: the wheel
is roughly a metre away, the ridge line hundreds of metres, so the same camera
motion displaces them by very different amounts.

This measures it directly. The near-field patch sits low and centre, where the
handlebars and the top of the front wheel are; the far-field patches sit high,
where the landscape is. Their ratio in apparent motion IS the speed
amplification, and it can be compared against the desired figure of about ten.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np


def frames(video, w=960, h=540):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video),
         "-vf", f"scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
        capture_output=True, check=True).stdout
    n = len(raw) // (w * h)
    return np.frombuffer(raw[:n * w * h], np.uint8).reshape(n, h, w).astype(np.float32)


def shift(a, b):
    """Peak phase-correlation shift from a to b, in pixels."""
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


# Regions as fractions of the frame: (y0, y1, x0, x1)
NEAR = (0.72, 0.98, 0.35, 0.65)     # handlebars and top of the front wheel
FAR = (0.05, 0.35, 0.30, 0.70)      # landscape, high in frame


def crop(f, box):
    y0, y1, x0, x1 = box
    h, w = f.shape
    return f[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def report(video):
    f = frames(video)
    near, far = [], []
    for i in range(len(f) - 1):
        nx, ny = shift(crop(f[i], NEAR), crop(f[i + 1], NEAR))
        fx, fy = shift(crop(f[i], FAR), crop(f[i + 1], FAR))
        near.append(np.hypot(nx, ny))
        far.append(np.hypot(fx, fy))
    near, far = np.array(near), np.array(far)
    print(f"{Path(video).name:16s} frames {len(f)}")
    print(f"  near-field motion  mean {near.mean():6.2f} px/frame")
    print(f"  far-field  motion  mean {far.mean():6.2f} px/frame")
    if far.mean() > 1e-6:
        print(f"  ratio near/far     {near.mean() / far.mean():6.2f}x")
    return near, far


if __name__ == "__main__":
    for path in sys.argv[1:]:
        report(path)