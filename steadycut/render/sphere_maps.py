"""Analytic sphere maps: the one place sphere geometry is built.

Promoted from experiments/scratch/o5/select_source.py (Task 0.0): the old
geq-built blend mask wandered ~220 px off-meridian by row 256 (measured on
the cached mask), because it drew a radial ramp on a non-square fisheye
canvas while v360 maps iv_fov over the height. Everything here is pure
numpy on the equirect grid, both lenses, version-tagged cache keys.

Conventions (match the serving graph in stitch.sphere_filtergraph):
equirect x maps linearly to yaw, x=0 front; sphere weight wgt is 1.0 on
lens A's hemisphere. The mask PNG written from it is 255 at sphere front —
where the graph, whose B track carries yaw=180 and whose maskedmerge white
selects the second input, shows B. Both halves pinned in tests.
"""
from __future__ import annotations

import numpy as np

C = 1440.0  # lens center px (2880 frame)
MASK_VERSION = "v2"


def lens_geometry(w=2048, h=1024, fov_a=182.2, fov_b=180.6):
    """Per-pixel lens incidence + validity for both lenses.

    Returns dict with inc_a, inc_b (radians from each lens axis), valid_a,
    valid_b (inside the lens circle), and yaw (radians, 0 front).
    Pure equidistant model, centered, untilted — the mismatch budget is
    stated in test_fov_fit, not hidden here.
    """
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    az = (xs / w) * 2 * np.pi
    el = (0.5 - ys / h) * np.pi
    dx = np.cos(el) * np.sin(az)
    dy = np.sin(el)
    dz = np.cos(el) * np.cos(az)
    inc_a = np.arccos(np.clip(dz, -1, 1))
    inc_b = np.arccos(np.clip(-dz, -1, 1))
    fa = C / (np.radians(fov_a) / 2)
    fb = (C / (np.radians(fov_b) / 2))
    valid_a = (fa * inc_a) <= C
    valid_b = (fb * inc_b) <= C
    return {"inc_a": inc_a, "inc_b": inc_b, "valid_a": valid_a,
            "valid_b": valid_b, "yaw": np.arctan2(dx, dz)}


def blend_weight(w=2048, h=1024, ramp_deg=4.0, fov_a=182.2, fov_b=180.6,
                 incidence_cap_deg=100.0):
    """Blend weight 0..1 (1 = lens A) + validity mask.

    Ramp of constant width in sphere yaw around both seam meridians
    (+-90 deg) — constant pixel width at every latitude, so the pole rows
    get a hard meridian switch instead of a full-row 0.5 smear. Above the
    incidence cap the better lens is hard-selected (no rim-vs-rim blend).
    Where a lens has no pixels the other wins outright (never blend black).
    """
    g = lens_geometry(w, h, fov_a, fov_b)
    yaw = g["yaw"]
    half = np.radians(ramp_deg) / 2.0
    # signed distance to nearest seam: seams at +-pi/2; A owns |yaw| < pi/2.
    d = np.abs(np.abs(yaw) - np.pi / 2)
    side = np.where(np.abs(yaw) < np.pi / 2, 1.0, -1.0)
    # 0.5 at the seam, full ownership a half-width out.
    wgt = np.clip(0.5 + side * d / (2 * half), 0, 1)
    wgt = np.where(d >= half, (side + 1) / 2, wgt)
    # incidence cap: hard-select where both lenses are deep rim.
    cap = np.radians(incidence_cap_deg)
    better_a = g["inc_a"] <= g["inc_b"]
    wgt = np.where(np.minimum(g["inc_a"], g["inc_b"]) > cap,
                   better_a.astype(float), wgt)
    # validity: never blend pixels a lens does not have.
    wgt = np.where(~g["valid_a"], 0.0, wgt)
    wgt = np.where(~g["valid_b"], 1.0, wgt)
    wgt = np.where(~g["valid_a"] & ~g["valid_b"], 0.0, wgt)
    valid = g["valid_a"] | g["valid_b"]
    return wgt.astype(np.float32), valid


def build_maps(w=2048, h=1024, fov_a=182.2, fov_b=180.6, feather_deg=3.0):
    """Per-lens fisheye remap tables + incidence weight (lifts select_source).

    Returns (map_a, map_b, weight, inc_a, inc_b) with maps as (mx, my)
    float32 for cv2.remap. Kept for the FOV fit and scoring; the serving
    v360 graph uses blend_weight's mask, not these tables.
    """
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    az = (xs / w) * 2 * np.pi
    el = (0.5 - ys / h) * np.pi
    dx = np.cos(el) * np.sin(az)
    dy = np.sin(el)
    dz = np.cos(el) * np.cos(az)
    inc_a = np.arccos(np.clip(dz, -1, 1))
    inc_b = np.arccos(np.clip(-dz, -1, 1))
    fa = C / (np.radians(fov_a) / 2)
    fb = (C / (np.radians(fov_b) / 2))
    out = {}
    for tag, inc, f in (("A", inc_a, fa), ("B", inc_b, fb)):
        r = f * inc
        lx, ly = (dx, -dy) if tag == "A" else (-dx, -dy)
        norm = np.hypot(lx, ly) + 1e-12
        mx = (C + r * lx / norm).astype(np.float32)
        my = (C + r * ly / norm).astype(np.float32)
        out[tag] = (mx, my, inc)
    gap = inc_b - inc_a
    fw = np.radians(feather_deg)
    wgt = np.clip(0.5 + gap / (2 * fw), 0, 1).astype(np.float32)
    return out["A"][:2], out["B"][:2], wgt, inc_a, inc_b
