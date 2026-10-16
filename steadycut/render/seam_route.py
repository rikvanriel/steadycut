"""Seam routing, both seams independently (Tasks 3.1, 3.2b): Viterbi path.

Cost v2 (measured): the v1 |A-B| cost threads BETWEEN doubled projections
(a 37 px trunk ghost survived routing — both copies stay fully visible).
Blending cannot un-double (averaging preserves both copies); only selection
can, by placing the seam outside both projections. So cost[row][x] =
texture_A(x-r) + texture_B(x+r): cut where NEITHER side has structure
within ramp distance. States clamp to joint-valid columns (never route
into a lens void), plus the vertical smoothness penalty.

Tried before v1: cv2.detail GraphCut/DpSeamFinder. Rejected, measured: on
coincident overlap strips both cut at the strip edge regardless of content,
and one family member segfaults on float input.
"""
from __future__ import annotations

import numpy as np

STEP_PENALTY = 8.0  # DN per px of vertical seam step
RAMP_PX = 11  # blend half-width at 4 deg: structure inside stays visible


def _texture(img):
    cv2 = __import__("cv2")
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    ty = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.blur(np.hypot(tx, ty), (5, 5))


def route_seam(eqA, eqB, seam_x, halfw=64, fov_a=185.0, fov_b=183.5):
    """Route one seam. eqA/eqB: full equirects (BGR uint8) in SHARED sphere
    coordinates — production B tracks (yaw=180) need NO roll: v360 yaw
    rotates the output image (measured: same lens at yaw180 lands exactly
    -1024 px, score 1.0), so lens0@yaw0 and lens1@yaw180 share coordinates.
    Rolling by half width compares opposite globe sides (that bug fed the
    router garbage cost).
    States clamp to joint-valid columns (analytic, FOVs given). Returns
    polyline (pano-x per row float)."""
    from steadycut.render.sphere_maps import joint_valid_strip
    x0 = max(0, seam_x - halfw)
    x1 = min(eqA.shape[1], seam_x + halfw)
    a = eqA[:, x0:x1]
    b = eqB[:, x0:x1]
    h, w = a.shape[:2]
    H, W = eqA.shape[:2]
    ta = _texture(a)
    tb = _texture(b)
    r = min(RAMP_PX, w // 4)
    cost = np.full((h, w), np.inf)
    inner = np.arange(r, w - r)
    # cut at x shows A left / B right: penalize structure within ramp
    # distance on the SHOWN side of each lens.
    cost[:, inner] = (ta[:, :-2 * r][:, :len(inner)]
                      + tb[:, 2 * r:][:, :len(inner)])
    valid = joint_valid_strip(seam_x, halfw, fov_a, fov_b, h, W)
    vw = valid[:, :w] if valid.shape[1] >= w else valid
    cost[~vw] = np.inf
    finite = np.isfinite(cost).any(axis=1)
    lam = float(np.median(ta) + np.median(tb)) + STEP_PENALTY
    prev = np.where(np.isfinite(cost[0]), cost[0], 1e9)
    back = np.zeros((h, w), np.int32)
    idx = np.arange(w)
    for y in range(1, h):
        row = np.where(np.isfinite(cost[y]), cost[y], 1e9)
        cand = prev[None, :] + lam * np.abs(idx[:, None] - idx[None, :])
        back[y] = np.argmin(cand, axis=1)
        prev = row + cand[np.arange(w), back[y]]
    path = np.zeros(h, np.int32)
    path[h - 1] = int(np.argmin(prev))
    for y in range(h - 2, -1, -1):
        path[y] = back[y + 1, path[y + 1]]
    got = (x0 + path).astype(float)
    # rows with no valid state hold the meridian (never invent positions).
    bad = ~finite
    got[bad] = seam_x
    return np.clip(got, x0, x1 - 1)


def route_both(eqA, eqB, seams=(512, 1536), halfw=64, width=2048,
               fov_a=185.0, fov_b=183.5):
    """Route each seam independently. Returns {seam_x: polyline}."""
    out = {}
    for s in seams:
        x = int(s * eqA.shape[1] / width)
        out[s] = route_seam(eqA, eqB, x, halfw, fov_a, fov_b)
    return out
