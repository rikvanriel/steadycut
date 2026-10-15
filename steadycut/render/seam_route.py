"""Seam routing, both seams independently (Task 3.1): Viterbi min-cost path.

Tried first: cv2.detail GraphCut/DpSeamFinder. Rejected, measured: on
coincident overlap strips both cut at the strip edge regardless of content
(even with minimum cost central), and one family member segfaults on float
input. At our corridor widths (24-64 px) Viterbi over per-pixel BGR
difference + vertical continuity penalty is the exact optimum, deterministic,
and crash-free — same cost family Studio minimizes (color difference),
without the binding.
"""
from __future__ import annotations

import numpy as np

STEP_PENALTY = 8.0  # DN per px of vertical seam step; see _lam()


def _lam(diff):
    return float(np.median(diff)) + STEP_PENALTY


def route_seam(eqA, eqB, seam_x, halfw=64):
    """Route one seam. eqA/eqB: full equirects (BGR uint8). Returns polyline
    (pano-x per row float) lying inside [seam_x-halfw, seam_x+halfw]."""
    x0 = max(0, seam_x - halfw)
    x1 = min(eqA.shape[1], seam_x + halfw)
    a = eqA[:, x0:x1].astype(np.float32)
    b = eqB[:, x0:x1].astype(np.float32)
    h, w = a.shape[:2]
    diff = np.abs(a - b).sum(axis=2)
    lam = _lam(diff)
    prev = diff[0].copy()
    back = np.zeros((h, w), np.int32)
    idx = np.arange(w)
    for y in range(1, h):
        # transition penalty |x - x'| via broadcast; w <= ~128 keeps it cheap.
        cand = prev[None, :] + lam * np.abs(idx[:, None] - idx[None, :])
        back[y] = np.argmin(cand, axis=1)
        prev = diff[y] + cand[np.arange(w), back[y]]
    path = np.zeros(h, np.int32)
    path[h - 1] = int(np.argmin(prev))
    for y in range(h - 2, -1, -1):
        path[y] = back[y + 1, path[y + 1]]
    return (x0 + path).astype(float)


def route_both(eqA, eqB, seams=(512, 1536), halfw=64, width=2048):
    """Route each seam independently. Returns {seam_x: polyline}."""
    out = {}
    for s in seams:
        x = int(s * eqA.shape[1] / width)
        out[s] = route_seam(eqA, eqB, x, halfw)
    return out
