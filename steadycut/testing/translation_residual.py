"""Gyro-anchored translation residual, per tracked feature.

The Han et al. reduction in miniature: plug the gyro-estimated rotation into
the projection model FIRST, leaving translation (+ noise) as the only
unknown. For each tracked feature at pixel (x, y) with measured displacement
(dx, dy): back-project to a ray, rotate the ray by the known inter-frame
rotation, re-project, and report residual = measured - predicted.

On synthetic footage the camera never translates, so the residual there IS
the instrument floor -- and the test below asserts it (median under 2 px).
On real footage the same residual, split far vs near, is the bounce signal:
far features barely move under translation, near ones do.
"""
from __future__ import annotations

import numpy as np

from steadycut.testing import synthetic_check as chk
from steadycut.testing.synthetic_profile import OUT_H, OUT_W


def predict_displacement(x, y, rotation, h_fov_deg: float, v_fov_deg: float,
                         width: int = OUT_W, height: int = OUT_H):
    """Where a feature at pixel (x, y) should land after `rotation` (3x3).

    `rotation` maps old camera-frame rays to new ones. Accepts scalars or
    arrays for x, y.
    """
    from steadycut.testing.synthetic_track import backproject
    ray = backproject(x, y, h_fov_deg, v_fov_deg, width, height)
    single = ray.ndim == 1
    rays = ray[None, :] if single else ray
    moved = rays @ np.asarray(rotation, float).T
    px, py, ok = chk.project(moved, h_fov_deg, v_fov_deg, width, height)
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = np.column_stack([px - x, py - y])
    return out, ok


def residual(tracks: np.ndarray, rotation, h_fov_deg: float, v_fov_deg: float,
             width: int = OUT_W, height: int = OUT_H) -> np.ndarray:
    """Per-track residual displacement in px after removing rotation.

    `tracks` are (N, 4) rows of (x, y, dx, dy). Returns (M, 3) rows of
    (x, y, |residual|) for tracks whose prediction lands in frame; tracks
    rotating out of frame are dropped, not extrapolated.
    """
    tracks = np.asarray(tracks, float).reshape(-1, 4)
    if len(tracks) == 0:
        return np.zeros((0, 3))
    pred, ok = predict_displacement(tracks[:, 0], tracks[:, 1], rotation,
                                    h_fov_deg, v_fov_deg, width, height)
    valid = ok & np.isfinite(pred).all(axis=1)
    if not valid.any():
        return np.zeros((0, 3))
    t = tracks[valid]
    r = np.hypot(t[:, 2] - pred[valid, 0], t[:, 3] - pred[valid, 1])
    return np.column_stack([t[:, 0], t[:, 1], r])
