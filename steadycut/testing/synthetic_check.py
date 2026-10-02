"""Projection, detection and matching: the measurement half of the test.

Everything here is convention-sensitive and therefore calibrated against the
real renderer before it is trusted (see tests/test_e2e_calibration.py). The
conventions, established by that calibration:

    equirectangular  lat -> row via (90 - lat)/180 * ROWS
                     lon -> col via (lon + 180)/360 * COLS
    sphere direction d = (cos(lat)sin(lon), sin(lat), cos(lat)cos(lon))
    camera frame     x right, y up, z forward
    output           rectilinear perspective, h_fov across the WIDTH

A convention error here does not fail loudly -- it reports itself as rotation
error in the pipeline. The calibration gate caught exactly that: a 48.6 degree
"pipeline error" that was `argmax` of the x-y norm selecting the marker
furthest from the view axis instead of `argmax` of z selecting the nearest.
"""
from __future__ import annotations

import cv2
import numpy as np

# One pixel of angular size, in degrees, at the output's vertical field.
# Frozen before the run: retuning this to a measured value would destroy its
# ability to catch a regression.
CAL_GATE_PX = 2.0
TOL_PX = 0.5


def deg_per_px(half_v_fov_deg: float, height: int) -> float:
    return 2.0 * half_v_fov_deg / height


def dirs_from_latlon(lat, lon) -> np.ndarray:
    """Unit directions in the sphere's frame. Accepts scalars or arrays."""
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.stack([np.cos(la) * np.sin(lo), np.sin(la),
                     np.cos(la) * np.cos(lo)], axis=-1)


def half_v_fov(h_fov_deg: float, width: int, height: int) -> float:
    return float(np.degrees(np.arctan(np.tan(np.radians(h_fov_deg / 2.0))
                                      * (height / width))))


def project(dirs_cam: np.ndarray, h_fov_deg: float, half_v_fov_deg: float,
            width: int, height: int):
    """Rectilinear projection to pixels.

    The two axes use DIFFERENT tangents: `h_fov` is measured across the WIDTH,
    so the horizontal scale is tan(h_fov/2) while the vertical scale is
    tan(half_v_fov). Using the vertical tangent for both is a uniform scale
    error of H/W -- 33% at 960x720 -- which leaves the centre marker perfect
    (it is at the origin of both scales) and puts every off-axis marker out by
    tens of pixels. That is invisible in a centre-marker check and obvious in
    the whole-grid one, which is why both exist.

    Returns (px, py, in_front). Points behind the camera have no image and are
    marked False rather than projected to a mirrored position -- mirroring is
    what a naive implementation does, and it puts spurious markers at the edges
    that then drag the median error around.
    """
    dirs_cam = np.asarray(dirs_cam, float)
    z = dirs_cam[:, 2]
    in_front = z > 1e-6
    safe = np.where(in_front, z, 1.0)
    t_h = np.tan(np.radians(h_fov_deg / 2.0))
    t_v = np.tan(np.radians(half_v_fov_deg))
    px = width / 2.0 + (dirs_cam[:, 0] / safe) / t_h * (width / 2.0)
    py = height / 2.0 - (dirs_cam[:, 1] / safe) / t_v * (height / 2.0)
    return px, py, in_front


def perspective_to_xyz(i, j, width, height, h=1.0):
    """v360's own perspective mapping, ported from vf_v360.c:3283-3317.

    Returns the unit direction on the sphere for frame position (i, j), or
    None where the mapping is undefined.

    This is NOT a rectilinear pinhole projection. v360 uses an azimuthal
    (stereographic-family) mapping whose shape depends on `h = 1 + v_fov`:

        uf = (2i+1)/W - 1,  vf = (2j+1)/H - 1,  rh = hypot(uf, vf)
        sinz = (h - sqrt(1 - rh^2)) / (h/rh + rh/h)

    and then phi = atan2(uf, vf), vec = (cos(t)sin(phi), cos(t)cos(phi),
    sin(t)) with t = asin(cosz). Assuming a pinhole here leaves the centre
    marker correct -- it sits at the origin of every scale -- and puts the
    surrounding markers out by more than 100 px, which is precisely what the
    whole-grid calibration check exists to catch.
    """
    uf = (2.0 * i + 1.0) / width - 1.0
    vf = (2.0 * j + 1.0) / height - 1.0
    rh = float(np.hypot(uf, vf))
    if rh >= 1.0:
        return None
    h = float(h)
    sinzz = 1.0 - rh * rh
    den = h / rh + rh / h
    sinz = (h - np.sqrt(sinzz)) / den if den else 0.0
    sinz2 = sinz * sinz
    if sinz2 > 1.0:
        return None
    cosz = float(np.sqrt(1.0 - sinz2))
    theta = float(np.arcsin(cosz))
    phi = float(np.arctan2(uf, vf))
    st, ct = np.sin(theta), np.cos(theta)
    return np.array([ct * np.sin(phi), ct * np.cos(phi), st])


def _solve_rh(sinz: float, h: float, lo=0.0, hi=0.999999) -> float:
    """Invert the sinz relation for rh, by bisection.

    Closed-form inversion is messy and error-prone; the relation is monotone
    decreasing in rh over [0, 1), so bisection is exact to machine precision
    and cannot silently return a wrong branch the way a rearranged formula can.
    """
    def f(r):
        den = h / r + r / h if r > 0 else float("inf")
        return (h - np.sqrt(max(0.0, 1.0 - r * r))) / den - sinz if den != float("inf") else np.inf
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        val = (h - np.sqrt(max(0.0, 1.0 - mid * mid))) / (h / mid + mid / h)
        if val < sinz:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def project_v360(dirs_cam, width, height, h=1.0):
    """Inverse of perspective_to_xyz: camera-frame direction -> pixel.

    Solved in the same (uf, vf) space the filter uses, then mapped back with
    the rescale the filter applies. `dirs_cam` must already be in the sphere's
    frame with the same handedness v360 expects.
    """
    dirs = np.asarray(dirs_cam, float)
    phi = np.arctan2(dirs[:, 0], dirs[:, 1])
    sinz = np.hypot(dirs[:, 0], dirs[:, 1])      # = cos(theta) in the filter
    out = np.full((len(dirs), 2), np.nan)
    for k in range(len(dirs)):
        if dirs[k, 2] <= 0.0:
            continue
        rh = _solve_rh(float(sinz[k]), h)
        uf = rh * np.sin(phi[k])
        vf = rh * np.cos(phi[k])
        out[k, 0] = ((uf + 1.0) * width - 1.0) / 2.0
        out[k, 1] = ((vf + 1.0) * height - 1.0) / 2.0
    inside = np.isfinite(out[:, 0]) & (out[:, 0] >= 0) & (out[:, 0] < width) \
        & (out[:, 1] >= 0) & (out[:, 1] < height)
    return out[:, 0], out[:, 1], inside


def detect_markers(frame_bgr: np.ndarray) -> np.ndarray:
    """Marker centroids (N, 2) in the rendered frame.

    Shape-filtered rather than area-only: a marker clipped by the frame edge
    keeps a plausible area but loses its circularity, and its centroid drifts
    inward, which shows up as a systematic edge bias in the error.
    """
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) \
        if frame_bgr.ndim == 3 else frame_bgr
    _, bw = cv2.threshold(g, 200, 255, cv2.THRESH_BINARY)
    n, _, stats, cents = cv2.connectedComponentsWithStats(bw)
    keep = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if 8 <= a <= 600 and 2 < w <= 40 and 2 < h <= 40:
            keep.append(cents[i])
    return np.asarray(keep, float).reshape(-1, 2)


def match(observed: np.ndarray, predicted: np.ndarray,
          edge_margin: float = 6.0, size=(960, 720)):
    """Distance from each predicted point to its nearest observed marker.

    Predictions within `edge_margin` of a frame edge are dropped: a marker
    clipped there has a centroid pulled inward, and keeping those points biases
    the error distribution for reasons that have nothing to do with the
    pipeline.
    """
    observed = np.asarray(observed, float).reshape(-1, 2)
    predicted = np.asarray(predicted, float).reshape(-1, 2)
    if len(observed) == 0 or len(predicted) == 0:
        return np.zeros(0)
    w, h = size
    inside = ((predicted[:, 0] > edge_margin) & (predicted[:, 0] < w - edge_margin)
              & (predicted[:, 1] > edge_margin) & (predicted[:, 1] < h - edge_margin))
    pred = predicted[inside]
    if len(pred) == 0:
        return np.zeros(0)
    d = np.linalg.norm(observed[:, None, :] - pred[None, :, :], axis=2)
    return d.min(axis=0)