"""Throwaway: residual frame-to-frame motion via phase correlation."""
import subprocess
import sys

import numpy as np

W, H = 480, 270
TOP = 2  # use the upper half: distant content, rotation-dominated


def frames(path):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vf",
         f"crop=iw:ih/{TOP}:0:0,scale={W}:{H},format=gray",
         "-f", "rawvideo", "-"],
        capture_output=True, check=True).stdout
    n = len(raw) // (W * H)
    return np.frombuffer(raw[: n * W * H], dtype=np.uint8).reshape(n, H, W).astype(np.float32)


def shift(a, b):
    """Peak of the cross-power spectrum = translation from a to b."""
    win = np.outer(np.hanning(H), np.hanning(W))
    fa = np.fft.rfft2((a - a.mean()) * win)
    fb = np.fft.rfft2((b - b.mean()) * win)
    r = fa * np.conj(fb)
    r /= np.maximum(np.abs(r), 1e-9)
    peak = np.fft.irfft2(r, s=(H, W))
    y, x = np.unravel_index(np.argmax(peak), peak.shape)
    if y > H // 2:
        y -= H
    if x > W // 2:
        x -= W
    return float(x), float(y)


def report(path):
    f = frames(path)
    d = np.array([shift(f[i], f[i + 1]) for i in range(len(f) - 1)])
    speed = np.linalg.norm(d, axis=1)
    jerk = np.linalg.norm(np.diff(d, axis=0), axis=1)
    print(f"{path}: frames={len(f)}")
    print(f"  motion speed  mean={speed.mean():6.2f} px  std={speed.std():6.2f}")
    print(f"  shake (delta) mean={jerk.mean():6.2f} px  std={jerk.std():6.2f}")
    return jerk.std()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
