"""P0 edge inventory for plumb-line calibration.

Two detectors (BOTH run; per-detector counts expose bias):
- tiles: HoughLinesP in seed-undistorted gnomonic patches per direction
- edgels: Canny edgel chaining with angle-continuity tolerance
Finger mask in lens-angular coordinates (bottom rim sector from the
circle fit). Per line: rho_min/R, sagitta, orientation class, radial
third. World-line dedupe across frames (stub: within-frame groups).
"""
from __future__ import annotations

import numpy as np


def angular_finger_mask(h, w, cx, cy, r, sector_deg=50.0):
    """Bottom rim sector mask (True = masked). Lens-angular coordinates:
    pixels whose angle from straight-down is < sector/2 AND whose radius
    is in the outer band. Never rows."""
    ys, xs = np.mgrid[0:h, 0:w].astype(float)
    dx, dy = xs - cx, ys - cy
    ang = np.degrees(np.arctan2(dx, dy))  # 0 = straight down
    rad = np.hypot(dx, dy) / r
    return (np.abs(ang) < sector_deg / 2) & (rad > 0.55)


def hough_tiles(gray, cx, cy, f, tile_deg=30.0, min_len=400):
    """HoughLinesP per gnomonic tile. Returns segments + per-tile counts.

    Each tile: rays within tile_deg of the tile axis are projected
    gnomonically (undistort with the SEED model), Hough run on the tile,
    segments mapped back to fisheye pixels with their tile id.
    """
    cv2 = __import__('cv2')
    h, w = gray.shape
    segs = []
    counts = {}
    step = int(tile_deg)
    for az in range(0, 360, step):
        a0 = np.radians(az - tile_deg / 2)
        # gnomonic tile: sample directions, undistort via seed equidistant
        n = 400
        th = np.linspace(0.05, 1.2, n)
        ph = np.linspace(a0, a0 + np.radians(tile_deg), n)
        TH, PH = np.meshgrid(th, ph)
        # tile image: direction (TH, PH) -> tangent-plane coords
        tx = np.tan(TH) * np.cos(PH - np.radians(az))
        ty = np.tan(TH) * np.sin(PH - np.radians(az))
        # ... to fisheye pixels via seed model r = f*theta
        rr = f * TH
        mx = (cx + rr * np.cos(PH)).astype(np.float32)
        my = (cy + rr * np.sin(PH)).astype(np.float32)
        tile = cv2.remap(gray, mx, my, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
        edges = cv2.Canny(tile, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80,
                                minLineLength=min_len, maxLineGap=12)
        counts[az] = 0 if lines is None else len(lines)
        if lines is None:
            continue
        # map segment endpoints back to fisheye pixels
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            def back(x, y):
                txx = x / n * 1.2 + 0.05
                pph = a0 + y / n * np.radians(tile_deg)
                rrr = f * txx
                return (cx + rrr * np.cos(pph), cy + rrr * np.sin(pph))
            segs.append((back(x1, y1), back(x2, y2), az))
    return segs, counts


def chain_edgels(gray, mask=None, tol_deg=8.0, min_len=400):
    """Canny edgels chained with angle-continuity tolerance. Returns
    polylines (lists of pixels) longer than min_len."""
    cv2 = __import__('cv2')
    edges = cv2.Canny(gray, 50, 150)
    if mask is not None:
        edges = np.where(mask, 0, edges)
    # contours as edgel chains; filter by length + angle continuity
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_NONE)
    tol = np.radians(tol_deg)
    out = []
    for c in contours:
        p = c[:, 0, :].astype(float)
        if len(p) < min_len:
            continue
        d = np.diff(p, axis=0)
        ang = np.arctan2(d[:, 1], d[:, 0])
        da = np.abs(np.diff(np.unwrap(ang)))
        if float(np.median(da)) < tol:
            out.append(p)
    return out


def line_info(p0, p1, cx, cy, r):
    """rho_min/R, sagitta proxy (endpoint-midpoint deviation needs the
    arc; here chord geometry), length, orientation class."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    length = float(np.hypot(*(p1 - p0)))
    # closest approach of the chord to the center
    mid = (p0 + p1) / 2
    d = p1 - p0
    t = -np.dot(mid - np.array([cx, cy]), d) / (np.dot(d, d) + 1e-12)
    closest = mid + np.clip(t, -0.5, 0.5) * d
    rho = float(np.hypot(*(closest - np.array([cx, cy]))) / r)
    ang = float(np.degrees(np.arctan2(d[1], d[0]))) % 180
    return {'length': length, 'rho_min': rho, 'angle': ang}


def sagitta(pts):
    """Max deviation of chain points from their PCA axis, in px. The axis
    (not the endpoint chord) is the reference so closed loops don't make
    the chord degenerate. Deep arcs constrain radial distortion;
    near-zero values are rotation-only witnesses."""
    p = np.asarray(pts, float)
    mu, d, L = chain_axis(pts)
    n = np.array([-d[1], d[0]])
    return float(np.abs((p - mu) @ n).max()), L


def unproject(pts, cx, cy, f, k1=0.0):
    """Fisheye pixels -> unit rays (equidistant + k1 radial). Lens frame:
    x right, y down, z forward (out of the lens)."""
    p = np.asarray(pts, float)
    dx, dy = (p[:, 0] - cx) / f, (p[:, 1] - cy) / f
    r = np.hypot(dx, dy)
    th = r * (1 + k1 * r * r)
    s = np.sinc(th / np.pi)  # sin(th)/th, safe at 0
    return np.stack([dx * s, dy * s, np.cos(th)], axis=1)


def plane_normal(rays):
    """Best-fit plane normal through camera center for unit rays; also
    RMS angular residual (great-circle agreement within one lens)."""
    _, _, vt = np.linalg.svd(rays, full_matrices=False)
    n = vt[-1] / (np.linalg.norm(vt[-1]) + 1e-12)
    res = np.degrees(np.arcsin(np.clip(np.abs(rays @ n), 0, 1)))
    return n, float(np.sqrt((res ** 2).mean()))


def plumb_residuals(params, chains_data):
    """Joint residual vector. params = [cxA, cyA, cxB, cyB, f, k1].
    chains_data = list of (lens, pts). Per chain: unproject with current
    params, fit plane, stack ray-to-plane angular residuals (radians).
    Rotation-free per lens: constrains centers + radial model only."""
    cxA, cyA, cxB, cyB, f, k1 = params
    out = []
    for lens, pts in chains_data:
        cx, cy = (cxA, cyA) if lens == 0 else (cxB, cyB)
        rays = unproject(pts, cx, cy, f, k1)
        n, _ = plane_normal(rays)
        out.append(np.arcsin(np.clip(np.abs(rays @ n), 0, 1)))
    return np.concatenate(out)


def fit_plumb(chains_data, x0):
    """Levenberg-Marquardt joint fit over centers + (f, k1). Returns
    (result, rms_deg). Report-only: caller decides gates. Bounds keep
    the solver off the degenerate at-infinity zero (parallel rays are
    trivially coplanar): centers ±60px of circle fit, f ±12%, |k1|≤0.3."""
    from scipy.optimize import least_squares
    x0 = np.asarray(x0, float)
    lo = np.array([x0[0] - 60, x0[1] - 60, x0[2] - 60, x0[3] - 60,
                   x0[4] * 0.88, -0.3])
    hi = np.array([x0[0] + 60, x0[1] + 60, x0[2] + 60, x0[3] + 60,
                   x0[4] * 1.12, 0.3])
    res = least_squares(plumb_residuals, x0, args=(chains_data,),
                        bounds=(lo, hi), method='trf',
                        max_nfev=200, verbose=0)
    rms = float(np.sqrt((res.fun ** 2).mean()))
    return res, np.degrees(rms)


def open_curve(pts, bin_px=4.0):
    """Collapse a there-and-back contour loop into a single open curve:
    project on the PCA axis, median perpendicular per bin."""
    p = np.asarray(pts, float)
    mu, d, _ = chain_axis(p)
    t = (p - mu) @ d
    n = np.array([-d[1], d[0]])
    s = (p - mu) @ n
    lo, hi = t.min(), t.max()
    nb = max(8, int((hi - lo) / bin_px))
    edges = np.linspace(lo, hi, nb + 1)
    idx = np.clip(np.digitize(t, edges) - 1, 0, nb - 1)
    curve = []
    for b in range(nb):
        m = idx == b
        if m.sum() >= 2:
            curve.append(mu + d * t[m].mean() + n * np.median(s[m]))
    return np.array(curve)


def chain_axis(pts):
    """PCA axis of a chain: (center, direction, extent). Works for closed
    loops where endpoints coincide."""
    p = np.asarray(pts, float)
    mu = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - mu, full_matrices=False)
    d = vt[0] / (np.linalg.norm(vt[0]) + 1e-12)
    t = (p - mu) @ d
    return mu, d, float(t.max() - t.min())


def group_fragments(items, cx, cy, r, angle_tol=6.0, rho_tol=0.03):
    """Cluster segments/chains into world-line groups by (angle, rho).

    items: list of (p0, p1) endpoint pairs. Returns groups (lists of
    indices) with merged info. Both edges of one jamb land in one group
    only if within rho_tol — else they stay separate constraints (safe).
    """
    infos = [line_info(a, b, cx, cy, r) for a, b in items]
    groups, used = [], [False] * len(items)
    for i, inf in enumerate(infos):
        if used[i]:
            continue
        g = [i]
        used[i] = True
        for j in range(i + 1, len(items)):
            if used[j]:
                continue
            d_ang = abs(infos[j]['angle'] - inf['angle'])
            d_ang = min(d_ang, 180 - d_ang)
            if d_ang < angle_tol and abs(infos[j]['rho_min'] -
                                        inf['rho_min']) < rho_tol:
                g.append(j)
                used[j] = True
        groups.append(g)
    return groups, infos
