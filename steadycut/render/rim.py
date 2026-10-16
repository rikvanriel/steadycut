"""Rim-blue correction (fisheye domain, per recording).

Both X4 lens frames carry an intensely blue ring at the image-circle
edge (measured snow-only: median excess +43..+60 at r 0.985-0.995, max
+137; +52..+85 at the edge, max +193). v360 magnifies that ring into
the blue bands flanking the seam. The ring sits at a fixed radius
(sensor boundary), but its amplitude scales with exposure/AWB, so it
is measured per recording from its own lens frames — same pattern as
per-clip gain — not stored per body.

rim_blue_correct() subtracts the above-baseline excess radially.
"""
from __future__ import annotations

import numpy as np


def rim_profile(lens_bgr, nbins=96, pct=75):
    """B-R excess vs normalized radius on bright pixels.

    Returns (centers, excess) with excess[k] the pct-th percentile over
    the annulus (75: the edge spike is narrow; the median dilutes it).
    Bright mask (min channel > 120) keeps sky/snow, drops black and dark
    content whose chroma is scene, not ring.
    """
    img = np.asarray(lens_bgr).astype(float)
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    r = np.sqrt((xs - w / 2) ** 2 + (ys - h / 2) ** 2) / (w / 2)
    bright = img.min(axis=2) > 120
    exc = img[:, :, 0] - img[:, :, 2]
    edges = np.linspace(0, 1.02, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    prof = np.full(nbins, np.nan)
    for k in range(nbins):
        m = bright & (r >= edges[k]) & (r < edges[k + 1])
        if m.sum() > 500:
            prof[k] = float(np.percentile(exc[m], pct))
    return centers, prof


def rim_blue_correct(lens_bgr, strength=1.0):
    """Subtract the rim ring above its mid-field baseline. Returns BGR."""
    cv2 = __import__("cv2")
    img = np.asarray(lens_bgr).astype(float)
    h, w = img.shape[:2]
    centers, prof = rim_profile(img)
    ok = ~np.isnan(prof)
    if ok.sum() < 8:
        return np.asarray(lens_bgr)
    base = float(np.median(prof[(centers > 0.45) & (centers < 0.7) & ok]))
    # baseline stays a median (stable mid-field); only the ring uses pct.
    corr = np.clip(prof - base, 0, None)
    corr[~ok] = 0.0
    # smooth radially (ring is smooth; pixel noise is not). Narrow kernel:
    # the edge spike is real and must survive.
    corr = cv2.GaussianBlur(corr.reshape(-1, 1), (1, 3), 0)[:, 0]
    ys, xs = np.mgrid[0:h, 0:w]
    r = np.sqrt((xs - w / 2) ** 2 + (ys - h / 2) ** 2) / (w / 2)
    sub = np.interp(r.flat, centers, corr).reshape(h, w) * strength
    out = img.copy()
    out[:, :, 0] = np.clip(out[:, :, 0] - sub, 0, 255)
    return out.astype(np.uint8)
