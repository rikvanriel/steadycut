"""Discrete feature tracking for the translation-bounce instrument.

Whole-strip phase correlation failed as a pitch instrument (rotation fields
are not uniform translations), so this tracks discrete corners instead and
predicts each one's motion individually. Lucas-Kanade on good-corners with a
forward-backward consistency check: track prev->curr, then curr->prev, and
drop anything that does not come back to within half a pixel. On forest
texture that rejection is load-bearing, not cosmetic -- leaves and
repetitive bark track plausibly and wrongly.
"""
from __future__ import annotations

import cv2
import numpy as np


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def backproject(px, py, h_fov_deg: float, v_fov_deg: float,
                width: int, height: int) -> np.ndarray:
    """Image pixel -> unit ray in the camera frame. Inverse of project()."""
    px, py = np.asarray(px, float), np.asarray(py, float)
    t_h = float(np.tan(np.radians(h_fov_deg / 2.0)))
    t_v = float(np.tan(np.radians(v_fov_deg / 2.0)))
    x = (px - width / 2.0) / (width / 2.0) * t_h
    y = (height / 2.0 - py) / (height / 2.0) * t_v
    d = np.stack([x, y, np.ones_like(x)], axis=-1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def track(prev: np.ndarray, curr: np.ndarray,
          max_corners: int = 400, fb_tol: float = 0.5) -> np.ndarray:
    """Track features prev -> curr. Returns (N, 4) rows of (x, y, dx, dy).

    Empty array when nothing survives -- the caller decides what that means,
    this function does not guess. Forward-backward check drops inconsistent
    tracks rather than averaging them: a track that does not return is
    mistracked, and its midpoint is not a measurement.
    """
    g0, g1 = to_gray(prev), to_gray(curr)
    pts = cv2.goodFeaturesToTrack(g0, maxCorners=max_corners, qualityLevel=0.02,
                                  minDistance=8, blockSize=5,
                                  useHarrisDetector=False)
    if pts is None or len(pts) == 0:
        return np.zeros((0, 4))
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts, None,
                                          winSize=(21, 21), maxLevel=3,
                                          criteria=(cv2.TERM_CRITERIA_EPS
                                                    | cv2.TERM_CRITERIA_COUNT,
                                                    20, 0.01))
    ok1 = st1.reshape(-1).astype(bool)
    if not ok1.any():
        return np.zeros((0, 4))
    p0_back, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None,
                                               winSize=(21, 21), maxLevel=3,
                                               criteria=(cv2.TERM_CRITERIA_EPS
                                                         | cv2.TERM_CRITERIA_COUNT,
                                                         20, 0.01))
    ok2 = st2.reshape(-1).astype(bool)
    fb = np.linalg.norm((p0_back - pts).reshape(-1, 2), axis=1)
    keep = ok1 & ok2 & (fb <= fb_tol)
    if not keep.any():
        return np.zeros((0, 4))
    a = pts[keep].reshape(-1, 2)
    b = p1[keep].reshape(-1, 2)
    return np.column_stack([a[:, 0], a[:, 1], b[:, 0] - a[:, 0],
                            b[:, 1] - a[:, 1]])
