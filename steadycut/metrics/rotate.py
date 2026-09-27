"""Estimate in-plane rotation robustly, by fitting many image regions at once.

tilt.py compares two halves, which fails on real footage: the left half may be
sky and the right forest, so the two shifts describe different motion.  This
instead divides the frame into a grid of patches, keeps only the ones with
enough texture to track, and fits a single rigid rotation plus translation to
all of them by least squares.  Featureless patches are dropped rather than
allowed to drag the estimate.

Model, for a patch whose centre sits at p relative to the frame centre, a
rotation th about that centre moves the patch by

    d = t + th * perp(p),   perp(x, y) = (-y, x)

which is linear in (t, th), so one least-squares solve recovers them all.
"""
from __future__ import annotations

import subprocess

import numpy as np

GRID = 4          # GRID x GRID patches
MIN_TEXTURE = 6.0  # grey-level std below which a patch is not trackable


def frames(path, w: int = 960, h: int = 540) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf",
         f"scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
        capture_output=True, check=True).stdout
    n = len(raw) // (w * h)
    return (np.frombuffer(raw[: n * w * h], dtype=np.uint8)
            .reshape(n, h, w).astype(np.float32))


def _shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Translation from a to b, via the peak of the cross-power spectrum."""
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


def _patches(a: np.ndarray, b: np.ndarray, grid: int = GRID):
    """Return (shifts, centres) for patches with enough texture to track."""
    h, w = a.shape
    ph, pw = h // grid, w // grid
    shifts, centres = [], []
    for gy in range(grid):
        for gx in range(grid):
            y0, x0 = gy * ph, gx * pw
            pa = a[y0:y0 + ph, x0:x0 + pw]
            pb = b[y0:y0 + ph, x0:x0 + pw]
            if min(pa.std(), pb.std()) < MIN_TEXTURE:
                continue
            dx, dy = _shift(pa, pb)
            shifts.append((dx, dy))
            centres.append((x0 + pw / 2 - w / 2, y0 + ph / 2 - h / 2))
    return np.asarray(shifts), np.asarray(centres)


def fit_rotation(shifts: np.ndarray, centres: np.ndarray) -> float:
    """Least-squares in-plane rotation, in radians, from patch shifts."""
    if len(shifts) < 3:
        return float("nan")
    # d = t + th * perp(p)  ->  solve for (tx, ty, th)
    A = np.zeros((2 * len(shifts), 3))
    b = shifts.reshape(-1)
    for i, (px, py) in enumerate(centres):
        A[2 * i] = [1.0, 0.0, -py]   # x: tx + th * (-py)
        A[2 * i + 1] = [0.0, 1.0, px]  # y: ty + th * ( px)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(sol[2])


def rotation_series(path, grid: int = GRID, fps: float = 29.97,
                    method: str = "orb", size: tuple[int, int] | None = None):
    """Per-frame in-plane rotation in degrees, with the patch count used.

    `size` must match the clip's own aspect ratio. The default 960x540 is 16:9,
    and a non-uniform rescale of a 4:3 clip squeezes angles in the ratio of the
    scale factors -- about 25 percent at 960x720 -- so a roll rate measured on
    the wrong size is not the roll rate. Pass the source size, or the reading is
    a plausible number for the wrong quantity.
    """
    w, h = size if size else (960, 540)
    if method == "orb":
        obs = frames(path, w, h)
        rates = []
        for i in range(len(obs) - 1):
            got = rotation_between(obs[i], obs[i + 1])
            rates.append(got if got is not None else 0.0)
        return np.array(rates), None

    f = frames(path, w, h)
    out, used = [], []
    for i in range(len(f) - 1):
        shifts, centres = _patches(f[i], f[i + 1], grid)
        out.append(fit_rotation(shifts, centres))
        used.append(len(shifts))
    deg = np.degrees(np.asarray(out))
    return deg, np.asarray(used)


if __name__ == "__main__":
    import sys

    deg, used = rotation_series(sys.argv[1])
    finite = deg[np.isfinite(deg)]
    print(f"{sys.argv[1]}")
    print(f"  patches tracked per frame: min {used.min()} max {used.max()} "
          f"of {GRID * GRID}")
    print(f"  rotation rate: mean {finite.mean():+.4f}  std {finite.std():.4f} "
          f"deg/frame  (range {finite.min():+.3f}..{finite.max():+.3f})")


def rotation_between(a: np.ndarray, b: np.ndarray) -> float | None:
    """In-plane rotation taking image `a` to image `b`, in degrees.

    Feature-based rather than phase-correlation based, because phase
    correlation only survives a couple of degrees: with a larger rotation the
    patches near the frame edge move further than they are wide and the
    estimate aliases. ORB features plus a partial-affine RANSAC fit handle
    large rotations and tolerate moderate parallax.
    """
    import cv2

    orb = cv2.ORB_create(nfeatures=3000, fastThreshold=7)
    ka, da = orb.detectAndCompute(a.astype("uint8"), None)
    kb, db = orb.detectAndCompute(b.astype("uint8"), None)
    if da is None or db is None or len(ka) < 12 or len(kb) < 12:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(da, db)
    if len(matches) < 12:
        return None

    pa = np.float32([ka[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pb = np.float32([kb[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    model, _ = cv2.estimateAffinePartial2D(
        pa, pb, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if model is None:
        return None
    return float(np.degrees(np.arctan2(model[1, 0], model[0, 0])))
