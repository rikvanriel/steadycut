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

Track identity (measured, do not re-derive from prose): "A"/"B" below are
GRAPH roles, not hardware front/rear. Track 0:v:0 is served with yaw=0
(content centered x=1024) and calibrated by spec.lens_fov_a; track 0:v:1
with yaw=180 (centered x=0) by spec.lens_fov_b. The FOV fit assigns in the
same order (its A/B inputs are track 0/1), and lens_to_sphere(back=True)
maps track-1 pixels for the leveling frames. All three agree; only the
words "front"/"rear" are untrusted. Order-swap probe (185,183.5) vs
(183.5,185) scores 3.5/11.6 vs 4.0/11.0 — second-order either way.
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


def lens_to_sphere(fov, size=2880, w=2048, h=1024, back=False):
    """Sphere coords per lens pixel (inverse of build_maps).

    For each fisheye pixel: equidistant-inverse to ray, then equirectangular
    projection. Returns (sx, sy) float32 + valid (inside the lens circle).
    Lets sphere-space corrections (seam ramps) ride on lens tracks. C scales
    with the raster (angular geometry is size-independent); production lens
    frames are always 2880, matching build_maps' constant.
    """
    C = size / 2.0
    f = C / (np.radians(fov) / 2)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    ox, oy = xs - C, ys - C
    n = np.hypot(ox, oy)
    inc = n / f
    valid = n <= C
    s = np.sin(np.minimum(inc, np.pi)) / np.maximum(n, 1e-12)
    dx = np.where(back, -ox * s, ox * s)
    dy = -oy * s
    dz = np.where(back, -np.cos(np.minimum(inc, np.pi)),
                  np.cos(np.minimum(inc, np.pi)))
    norm = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-12
    az = np.arctan2(dx / norm, dz / norm)
    el = np.arcsin(np.clip(dy / norm, -1, 1))
    sx = ((az / (2 * np.pi)) % 1.0 * w).astype(np.float32)
    sy = ((0.5 - el / np.pi) * h).astype(np.float32)
    return sx, sy, valid


def routed_weight(w=2048, h=1024, ramp_deg=4.0, fov_a=182.2, fov_b=180.6,
                  incidence_cap_deg=100.0, polylines=None, corridor_px=64):
    """Incidence weight gated by routed corridors (Task 3.2).

    Routing gates incidence, never replaces it: the ramp is recentered on
    each seam's polyline (same half-width), and OUTSIDE the corridor the
    better-hemisphere lens is hard-selected. Meridian polylines reproduce
    blend_weight byte-identically (pinned). polylines: {512: arr(1024,),
    1536: arr} in pano-x at width w, or None for both-meridian.
    """
    g = lens_geometry(w, h, fov_a, fov_b)
    yaw = g["yaw"]
    half = np.radians(ramp_deg) / 2.0
    yaw_px = ((yaw / (2 * np.pi)) % 1.0) * w  # 0..w, 0 = front
    side = np.where(np.abs(yaw) < np.pi / 2, 1.0, -1.0)
    if polylines is None:
        # meridian distances in yaw radians: identical to blend_weight
        # (pinned byte-identical); pixel distances would need a 2π/w scale.
        d = np.minimum(np.abs(np.abs(yaw) - np.pi / 2),
                       np.pi - np.abs(np.abs(yaw) - np.pi / 2))
    else:
        d = np.full((h, w), np.inf)
        for s, arr in polylines.items():
            rows = np.clip((np.arange(h) * len(arr) // h), 0, len(arr) - 1)
            dd = np.abs(yaw_px - arr[rows][:, None])
            dd = np.minimum(dd, w - dd) * (2 * np.pi / w)
            d = np.minimum(d, dd)
    wgt = np.clip(0.5 + side * d / (2 * half), 0, 1)
    wgt = np.where(d >= half, (side + 1) / 2, wgt)
    cap = np.radians(incidence_cap_deg)
    better_a = g["inc_a"] <= g["inc_b"]
    wgt = np.where(np.minimum(g["inc_a"], g["inc_b"]) > cap,
                   better_a.astype(float), wgt)
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


def joint_valid_strip(seam_x, halfw, fov_a=185.0, fov_b=183.5, h=1024,
                      w=2048):
    """Joint-valid mask for a sphere-coordinate corridor strip, in V360
    OUTPUT frame (measured: track0/yaw0 centers output-1024, track1/yaw180
    centers output-0; lens_geometry's internal az=0 sits at output-1024, so
    its validity is 180 deg off here — this function does NOT reuse it).

    Analytic incidence vs half-FOV per lens, no black-pixel thresholds
    (scene confounds those). The overlap is only ~24 px wide at fitted
    FOVs — strips wider than that are mostly one-sided, and DIS matched
    against black edges there (measured blue-smear failure)."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    az = (xs / w) * 2 * np.pi
    el = (0.5 - ys / h) * np.pi
    dx = np.cos(el) * np.sin(az)
    dy = np.sin(el)
    dz = np.cos(el) * np.cos(az)
    # output-frame axes: track0 (fov_a) at az=pi (x=1024), track1 at az=0.
    inc_a = np.arccos(np.clip(-dz, -1, 1))
    inc_b = np.arccos(np.clip(dz, -1, 1))
    va = inc_a <= np.radians(fov_a) / 2
    vb = inc_b <= np.radians(fov_b) / 2
    x0 = max(0, seam_x - halfw)
    x1 = min(w, seam_x + halfw)
    return (va & vb)[:, x0:x1]
